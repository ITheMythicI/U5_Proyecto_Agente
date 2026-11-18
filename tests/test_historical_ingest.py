"""
Suite de Pruebas Automatizadas para el Motor de Ingesta Histórica.
Utiliza exclusivamente respuestas simuladas en memoria (Mocks) y directorios temporales aislados.
NUNCA realiza llamadas a la red ni afecta los datos reales del proyecto.
"""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

# Añadir raíz de U5_Proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.historical_ingest import HistoricalIngestManager


class TestHistoricalIngestLogic(unittest.TestCase):

    def setUp(self):
        """Crea un entorno temporal aislado para cada prueba."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.test_dir.name)

    def tearDown(self):
        """Elimina el entorno temporal."""
        self.test_dir.cleanup()

    @patch("src.historical_ingest.REQUEST_INTERVAL_SECONDS", 0.001)
    def test_01_season_parameter_and_cursor_pagination(self):
        """Verifica que se envíen los parámetros season correctos y se pagine por cursor."""
        manager = HistoricalIngestManager(seasons=[2022], raw_data_dir=self.raw_dir)
        manager.api_key = "fake_key_for_testing"

        # Simular respuestas de API para 2 páginas
        page1_response = {
            "data": [{"id": i, "season": 2022, "home_score": 1, "away_score": 0} for i in range(1, 101)],
            "meta": {"next_cursor": 101}
        }
        page2_response = {
            "data": [{"id": i, "season": 2022, "home_score": 2, "away_score": 1} for i in range(101, 151)],
            "meta": {"next_cursor": None}
        }

        mock_session = MagicMock()
        mock_resp1 = MagicMock(status_code=200, json=lambda: page1_response)
        mock_resp2 = MagicMock(status_code=200, json=lambda: page2_response)
        mock_session.get.side_effect = [mock_resp1, mock_resp2]

        with patch("requests.Session", return_value=mock_session), \
             patch.object(manager, "_probe_matches_access", return_value=True), \
             patch.object(manager, "_check_disk_space", return_value=True):
            
            res = manager.run_ingestion()
            self.assertEqual(res["status"], "success")

            # Comprobar llamadas HTTP realizadas
            calls = mock_session.get.call_args_list
            self.assertEqual(len(calls), 2)
            
            # Llamada 1: season=2022, per_page=100, sin cursor
            self.assertEqual(calls[0][1]["params"]["season"], 2022)
            self.assertNotIn("cursor", calls[0][1]["params"])
            
            # Llamada 2: season=2022, per_page=100, cursor=101
            self.assertEqual(calls[1][1]["params"]["season"], 2022)
            self.assertEqual(calls[1][1]["params"]["cursor"], 101)

            # Verificar que el archivo consolidado matches_2022.json contiene 150 partidos
            consolidated = self.raw_dir / "matches_2022.json"
            self.assertTrue(consolidated.exists())
            with open(consolidated) as f:
                saved = json.load(f)
                self.assertEqual(len(saved["data"]), 150)

    @patch("src.historical_ingest.REQUEST_INTERVAL_SECONDS", 0.001)
    def test_02_resumption_after_interruption(self):
        """Verifica que una descarga interrumpida reanude desde la página y cursor pendientes sin repetir las previas."""
        manager = HistoricalIngestManager(seasons=[2023], raw_data_dir=self.raw_dir)
        manager.api_key = "fake_key_for_testing"

        page1_response = {
            "data": [{"id": 1, "season": 2023}],
            "meta": {"next_cursor": 50}
        }
        page2_response = {
            "data": [{"id": 2, "season": 2023}],
            "meta": {"next_cursor": None}
        }

        # Intento 1: Página 1 tiene éxito, página 2 falla con corte de red
        mock_session_1 = MagicMock()
        mock_session_1.get.side_effect = [
            MagicMock(status_code=200, json=lambda: page1_response),
            Exception("Simulación de corte de conexión")
        ]

        with patch("requests.Session", return_value=mock_session_1), \
             patch.object(manager, "_probe_matches_access", return_value=True), \
             patch.object(manager, "_check_disk_space", return_value=True):
            
            res1 = manager.run_ingestion()
            self.assertEqual(res1["status"], "interrupted")
            
            # Verificar que el checkpoint registró la página 1 y el cursor 50
            cp_data = manager.checkpoint["2023"]
            self.assertEqual(cp_data["status"], "IN_PROGRESS")
            self.assertEqual(cp_data["pages_downloaded"], 1)
            self.assertEqual(cp_data["next_cursor"], 50)

        # Intento 2: Reanudación con una nueva instancia del manager
        manager_resume = HistoricalIngestManager(seasons=[2023], raw_data_dir=self.raw_dir)
        manager_resume.api_key = "fake_key_for_testing"

        mock_session_2 = MagicMock()
        mock_session_2.get.side_effect = [
            MagicMock(status_code=200, json=lambda: page2_response)
        ]

        with patch("requests.Session", return_value=mock_session_2), \
             patch.object(manager_resume, "_probe_matches_access", return_value=True), \
             patch.object(manager_resume, "_check_disk_space", return_value=True):
            
            res2 = manager_resume.run_ingestion()
            self.assertEqual(res2["status"], "success")

            # Debe haber hecho exactamente 1 llamada (la página 2 con cursor 50), omitiendo la página 1
            calls = mock_session_2.get.call_args_list
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["params"]["cursor"], 50)
            self.assertEqual(manager_resume.checkpoint["2023"]["status"], "COMPLETED")
            self.assertEqual(manager_resume.checkpoint["2023"]["unique_matches"], 2)

    @patch("src.historical_ingest.REQUEST_INTERVAL_SECONDS", 0.001)
    def test_03_deduplication_during_consolidation(self):
        """Verifica que si un partido aparece en dos páginas, se consolide una sola vez."""
        manager = HistoricalIngestManager(seasons=[2024], raw_data_dir=self.raw_dir)
        manager.api_key = "fake_key_for_testing"

        # ID 2 aparece en ambas páginas
        page1 = {"data": [{"id": 1}, {"id": 2}], "meta": {"next_cursor": 20}}
        page2 = {"data": [{"id": 2}, {"id": 3}], "meta": {"next_cursor": None}}

        mock_session = MagicMock()
        mock_session.get.side_effect = [
            MagicMock(status_code=200, json=lambda: page1),
            MagicMock(status_code=200, json=lambda: page2)
        ]

        with patch("requests.Session", return_value=mock_session), \
             patch.object(manager, "_probe_matches_access", return_value=True), \
             patch.object(manager, "_check_disk_space", return_value=True):
            
            res = manager.run_ingestion()
            self.assertEqual(res["status"], "success")
            self.assertEqual(manager.checkpoint["2024"]["unique_matches"], 3)  # 1, 2, 3 (no 4)

    @patch("src.historical_ingest.REQUEST_INTERVAL_SECONDS", 0.001)
    def test_04_no_premature_completed_status(self):
        """Verifica que si una temporada queda vacía o falla, NUNCA se marque como COMPLETED."""
        manager = HistoricalIngestManager(seasons=[2022], raw_data_dir=self.raw_dir)
        manager.api_key = "fake_key_for_testing"

        # Respuesta vacía sin partidos
        empty_page = {"data": [], "meta": {"next_cursor": None}}

        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock(status_code=200, json=lambda: empty_page)

        with patch("requests.Session", return_value=mock_session), \
             patch.object(manager, "_probe_matches_access", return_value=True), \
             patch.object(manager, "_check_disk_space", return_value=True):
            
            manager.run_ingestion()
            # Debe quedar como ERROR_EMPTY y nunca como COMPLETED
            self.assertNotEqual(manager.checkpoint.get("2022", {}).get("status"), "COMPLETED")


if __name__ == "__main__":
    unittest.main()
