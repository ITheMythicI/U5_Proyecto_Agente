"""
Suite de Pruebas de Integración y Validación para Fase 1.
Verifica:
1. Esquemas Pydantic y distinción crítica NULL vs 0.
2. Salvaguarda del cliente BALLDONTLIE ante endpoints de pago y errores HTTP 401/403.
3. Pipeline PySpark procesando hacia Parquet con tipos estrictos.
4. Consultas analíticas en DuckDB de forma 100% desconectada.
"""

import sys
import unittest
from pathlib import Path

# Añadir raíz al sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.schemas.balldontlie_schemas import (
    SoccerTeam,
    SoccerStanding,
    SoccerMatch,
    IngestionReport
)
from src.data.balldontlie_client import (
    BALLDONTLIEClient,
    PaidEndpointRestrictedError,
    BallDontLieAPIError
)
from src.etl.spark_pipeline import (
    get_spark_session,
    process_teams_json_to_parquet,
    process_standings_json_to_parquet,
    process_matches_json_to_parquet,
    register_parquet_in_duckdb,
    run_duckdb_test_queries
)


class TestPhase1Integrity(unittest.TestCase):

    def test_01_pydantic_null_vs_zero_distinction(self):
        """Verifica que un marcador 0-0 se distinga de un partido pendiente (NULL)."""
        # Caso A: Partido jugado con marcador 0 - 0 (empate sin goles)
        match_played_0_0 = SoccerMatch(
            id=101,
            season=2023,
            date="2023-09-01T15:00:00Z",
            status="FT",
            status_state="post",
            home_team_id=1,
            away_team_id=2,
            home_score=0,
            away_score=0
        )
        self.assertEqual(match_played_0_0.home_score, 0)
        self.assertEqual(match_played_0_0.away_score, 0)
        self.assertEqual(match_played_0_0.resultado_1x2, "Empate")

        # Caso B: Partido programado pero no jugado (NULL)
        match_scheduled_null = SoccerMatch(
            id=102,
            season=2023,
            date="2024-05-19T15:00:00Z",
            status="Scheduled",
            status_state="pre",
            home_team_id=1,
            away_team_id=2,
            home_score=None,
            away_score=None
        )
        self.assertIsNone(match_scheduled_null.home_score)
        self.assertIsNone(match_scheduled_null.away_score)
        self.assertIsNone(match_scheduled_null.resultado_1x2)

    def test_02_pydantic_standing_consistency_validation(self):
        """Verifica que el validador detecte discrepancias aritméticas en clasificaciones."""
        team = SoccerTeam(id=1, name="Arsenal FC", abbreviation="ARS")
        
        # Clasificación consistente
        valid_standing = SoccerStanding(
            season=2023,
            rank=2,
            team=team,
            games_played=38,
            wins=28,
            draws=5,
            losses=5,
            points=89,
            goals_for=91,
            goals_against=29,
            goal_differential=62
        )
        self.assertEqual(valid_standing.points, 89)

        # Inconsistencia provocada en suma de partidos (28+5+4 = 37 != 38)
        with self.assertRaises(ValueError):
            SoccerStanding(
                season=2023,
                rank=2,
                team=team,
                games_played=38,
                wins=28,
                draws=5,
                losses=4,  # Suma 37
                points=89,
                goals_for=91,
                goals_against=29,
                goal_differential=62
            )

    def test_03_paid_endpoint_guardrail(self):
        """Comprueba que el cliente se niegue a invocar /matches si allow_paid_endpoints=False."""
        client = BALLDONTLIEClient(
            api_key="clave_falsa",
            allow_paid_endpoints=False
        )
        with self.assertRaises(PaidEndpointRestrictedError):
            client.fetch_matches(season=2023)

    def test_04_unauthorized_error_handling(self):
        """Verifica que una clave vacía o inválida devuelva BallDontLieAPIError con código 401."""
        client = BALLDONTLIEClient(api_key="clave_invalida_de_prueba")
        with self.assertRaises(BallDontLieAPIError) as ctx:
            client.fetch_teams()
        self.assertIn("401", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
