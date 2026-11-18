"""
Suite de Pruebas de Integración y Calidad Post-ETL.
Valida la integridad de matches.parquet, teams.parquet, matches_features.parquet y DuckDB.
"""

import sys
import unittest
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAKEHOUSE_DB_PATH, PROCESSED_DATA_DIR


class TestPostETLIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.con = duckdb.connect(str(LAKEHOUSE_DB_PATH), read_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    def test_01_total_records_preservation(self):
        """Verifica que no se haya perdido ni un solo partido de los 1,140 originales."""
        total = self.con.execute("SELECT count(*) FROM matches").fetchone()[0]
        self.assertEqual(total, 1140, "El total de partidos en matches.parquet debe ser exactamente 1,140.")

        counts_by_season = self.con.execute("""
            SELECT season, count(*) FROM matches GROUP BY season ORDER BY season
        """).fetchall()
        self.assertEqual(counts_by_season, [(2022, 380), (2023, 380), (2024, 380)])

    def test_02_preservation_of_zero_scores(self):
        """Comprueba que los marcadores 0 se conserven exactamente igual que en el JSON crudo."""
        ceros = self.con.execute("""
            SELECT 
                sum(CASE WHEN home_score = 0 THEN 1 ELSE 0 END),
                sum(CASE WHEN away_score = 0 THEN 1 ELSE 0 END),
                sum(CASE WHEN home_score = 0 AND away_score = 0 THEN 1 ELSE 0 END)
            FROM matches
        """).fetchone()
        self.assertEqual(ceros[0], 241, "Total de ceros local debe ser 241")
        self.assertEqual(ceros[1], 301, "Total de ceros visita debe ser 301")
        self.assertEqual(ceros[2], 50, "Total de empates 0-0 debe ser 50")

    def test_03_teams_catalog(self):
        """Verifica que el catálogo de clubes contenga exactamente los 24 equipos únicos."""
        total_teams = self.con.execute("SELECT count(*) FROM teams").fetchone()[0]
        self.assertEqual(total_teams, 24)

    def test_04_features_dataset_alignment(self):
        """Comprueba que el dataset de características tenga 1,140 filas y división temporal correcta."""
        total_feat = self.con.execute("SELECT count(*) FROM matches_features").fetchone()[0]
        self.assertEqual(total_feat, 1140)

        train_count = self.con.execute("SELECT count(*) FROM matches_features WHERE split_set = 'train'").fetchone()[0]
        test_count = self.con.execute("SELECT count(*) FROM matches_features WHERE split_set = 'test'").fetchone()[0]
        self.assertEqual(train_count, 760)
        self.assertEqual(test_count, 380)

    def test_05_no_temporal_data_leakage(self):
        """Verifica que la frontera temporal entre train y test no presente solapamientos."""
        max_train_date = self.con.execute("SELECT max(match_date) FROM matches_features WHERE split_set = 'train'").fetchone()[0]
        min_test_date = self.con.execute("SELECT min(match_date) FROM matches_features WHERE split_set = 'test'").fetchone()[0]
        self.assertLess(max_train_date, min_test_date, "La fecha máxima de train debe ser anterior a la mínima de test.")


if __name__ == "__main__":
    unittest.main()
