"""
Prueba de integración técnica para PySpark y DuckDB.
Genera un archivo JSON crudo representativo de la respuesta de BALLDONTLIE EPL V2,
ejecuta el pipeline de PySpark hacia Parquet y verifica las consultas analíticas en DuckDB.
"""

import sys
import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LAKEHOUSE_DB_PATH
from src.etl.spark_pipeline import (
    get_spark_session,
    process_teams_json_to_parquet,
    process_standings_json_to_parquet,
    process_matches_json_to_parquet,
    register_parquet_in_duckdb,
    run_duckdb_test_queries
)
import duckdb


class TestSparkDuckDBPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepara archivos JSON de prueba bajo la estructura de BALLDONTLIE EPL V2."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Equipos de prueba
        teams_data = {
            "data": [
                {"id": 1, "name": "Arsenal FC", "short_name": "Arsenal", "abbreviation": "ARS", "location": "London"},
                {"id": 2, "name": "Manchester City FC", "short_name": "Man City", "abbreviation": "MCI", "location": "Manchester"},
                {"id": 3, "name": "Liverpool FC", "short_name": "Liverpool", "abbreviation": "LIV", "location": "Liverpool"}
            ]
        }
        cls.teams_file = RAW_DATA_DIR / "teams_test_20261116.json"
        with open(cls.teams_file, "w", encoding="utf-8") as f:
            json.dump(teams_data, f)

        # 2. Clasificaciones de prueba (con consistencia aritmética)
        standings_data = {
            "data": [
                {
                    "season": 2023, "rank": 1,
                    "team": {"id": 2, "name": "Manchester City FC", "abbreviation": "MCI"},
                    "games_played": 38, "wins": 28, "draws": 7, "losses": 3,
                    "points": 91, "goals_for": 96, "goals_against": 34, "goal_differential": 62,
                    "points_per_game": 2.39, "note": "Champions"
                },
                {
                    "season": 2023, "rank": 2,
                    "team": {"id": 1, "name": "Arsenal FC", "abbreviation": "ARS"},
                    "games_played": 38, "wins": 28, "draws": 5, "losses": 5,
                    "points": 89, "goals_for": 91, "goals_against": 29, "goal_differential": 62,
                    "points_per_game": 2.34, "note": "UCL"
                }
            ]
        }
        cls.standings_file = RAW_DATA_DIR / "standings_2023_test.json"
        with open(cls.standings_file, "w", encoding="utf-8") as f:
            json.dump(standings_data, f)

        # 3. Partidos de prueba (Demostrando distinción NULL vs 0)
        matches_data = {
            "data": [
                {
                    "id": 901, "season": 2023, "date": "2023-10-08T15:30:00Z",
                    "status": "FT", "status_state": "post",
                    "home_team_id": 1, "away_team_id": 2,
                    "home_team": {"name": "Arsenal FC"}, "away_team": {"name": "Manchester City FC"},
                    "home_score": 1, "away_score": 0,  # Marcador 0 de visita (clean sheet local)
                    "venue_name": "Emirates Stadium", "attendance": 60233
                },
                {
                    "id": 902, "season": 2023, "date": "2024-03-31T15:30:00Z",
                    "status": "FT", "status_state": "post",
                    "home_team_id": 2, "away_team_id": 1,
                    "home_team": {"name": "Manchester City FC"}, "away_team": {"name": "Arsenal FC"},
                    "home_score": 0, "away_score": 0,  # Ambos en 0 (empate a ceros)
                    "venue_name": "Etihad Stadium", "attendance": 54900
                },
                {
                    "id": 903, "season": 2023, "date": "2024-05-25T15:00:00Z",
                    "status": "Scheduled", "status_state": "pre",
                    "home_team_id": 1, "away_team_id": 3,
                    "home_team": {"name": "Arsenal FC"}, "away_team": {"name": "Liverpool FC"},
                    "home_score": None, "away_score": None,  # NULL estricto: partido no disputado
                    "venue_name": "Emirates Stadium", "attendance": None
                }
            ]
        }
        cls.matches_file = RAW_DATA_DIR / "matches_2023_test.json"
        with open(cls.matches_file, "w", encoding="utf-8") as f:
            json.dump(matches_data, f)

    def test_spark_etl_and_duckdb_queries(self):
        """Ejecuta el pipeline PySpark completo y valida el comportamiento en DuckDB."""
        spark = get_spark_session("TestEPLPipeline")
        try:
            t_path = process_teams_json_to_parquet(spark)
            self.assertIsNotNone(t_path)
            self.assertTrue(Path(t_path).exists())

            s_path = process_standings_json_to_parquet(spark)
            self.assertIsNotNone(s_path)
            self.assertTrue(Path(s_path).exists())

            m_path = process_matches_json_to_parquet(spark)
            self.assertIsNotNone(m_path)
            self.assertTrue(Path(m_path).exists())
        finally:
            spark.stop()

        # Registro en DuckDB
        registered = register_parquet_in_duckdb()
        self.assertIn("teams", registered)
        self.assertIn("standings", registered)
        self.assertIn("matches", registered)
        self.assertEqual(registered["teams"], 3)
        self.assertEqual(registered["standings"], 2)
        self.assertEqual(registered["matches"], 3)

        # Verificación estricta en DuckDB: NULL vs 0
        con = duckdb.connect(str(LAKEHOUSE_DB_PATH), read_only=True)
        try:
            # 1. Comprobar que el partido no jugado tiene home_score NULL (no 0)
            null_match = con.execute("SELECT home_score, away_score, resultado_1x2 FROM matches WHERE match_id = 903").fetchone()
            self.assertIsNone(null_match[0], "home_score debe ser NULL para partido no jugado")
            self.assertIsNone(null_match[1], "away_score debe ser NULL para partido no jugado")
            self.assertIsNone(null_match[2], "resultado_1x2 debe ser NULL para partido no jugado")

            # 2. Comprobar que el partido 902 tiene marcador 0-0 y resultado 'Empate'
            zero_match = con.execute("SELECT home_score, away_score, resultado_1x2 FROM matches WHERE match_id = 902").fetchone()
            self.assertEqual(zero_match[0], 0, "home_score debe ser 0")
            self.assertEqual(zero_match[1], 0, "away_score debe ser 0")
            self.assertEqual(zero_match[2], "Empate", "resultado debe ser Empate")

            # 3. Comprobar que el partido 901 tiene resultado 'Local'
            local_win = con.execute("SELECT resultado_1x2 FROM matches WHERE match_id = 901").fetchone()
            self.assertEqual(local_win[0], "Local")
        finally:
            con.close()

    @classmethod
    def tearDownClass(cls):
        """Limpia los archivos de prueba para dejar el repositorio limpio."""
        for f in [cls.teams_file, cls.standings_file, cls.matches_file]:
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    unittest.main()
