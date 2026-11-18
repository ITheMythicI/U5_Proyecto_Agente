"""
Punto de entrada para la Ingesta Oficial y Pipeline ETL (Fase 1).
Ejecuta la descarga desde BALLDONTLIE EPL V2 (endpoints autorizados),
valida con Pydantic, procesa con PySpark hacia Parquet y registra en DuckDB.
"""

import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    BALLDONTLIE_API_KEY,
    ALLOW_PAID_ENDPOINTS,
    LAKEHOUSE_DB_PATH
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MainIngest")


def main():
    logger.info("=" * 70)
    logger.info("INICIO DE INGESTA Y PROCESAMIENTO: FASE 1 (BALLDONTLIE EPL V2 + PYSPARK)")
    logger.info("=" * 70)

    client = BALLDONTLIEClient()

    # 1. Comprobación de API Key
    if not client.api_key or client.api_key == "tu_api_key_aqui":
        logger.warning(
            "AVISO: BALLDONTLIE_API_KEY no configurada o usa valor por defecto en .env.\n"
            "Para descargar datos reales desde la API, ingresa tu clave en el archivo .env."
        )

    # 2. Descarga de Equipos (/epl/v2/teams - Endpoint Free)
    teams_count = 0
    if client.api_key and client.api_key != "tu_api_key_aqui":
        try:
            teams, t_report = client.fetch_teams()
            teams_count = t_report.registros_validos
            logger.info(f"Equipos descargados: {teams_count} (Duplicados: {t_report.duplicados_descartados})")
        except BallDontLieAPIError as e:
            logger.error(f"Fallo en descarga de equipos: {e}")

        # 3. Descarga de Clasificaciones (/epl/v2/standings - Endpoint Free)
        for season in [2022, 2023, 2024]:
            try:
                standings, s_report = client.fetch_standings(season=season)
                logger.info(f"Clasificación {season} descargada: {s_report.registros_validos} equipos.")
            except BallDontLieAPIError as e:
                logger.error(f"Fallo en descarga de clasificación para temporada {season}: {e}")

        # 4. Descarga de Partidos (/epl/v2/matches - Requiere ALL-STAR / Trial)
        if ALLOW_PAID_ENDPOINTS:
            logger.info("ALLOW_PAID_ENDPOINTS=True: Iniciando descarga de partidos autorizada...")
            for season in [2022, 2023, 2024]:
                try:
                    matches, m_report = client.fetch_matches(season=season)
                    logger.info(f"Partidos temporada {season}: {m_report.registros_validos} válidos.")
                except BallDontLieAPIError as e:
                    logger.error(f"Error descargando partidos de temporada {season}: {e}")
                    logger.info("Deteniendo descarga de partidos conforme a política de integridad.")
                    break
        else:
            logger.info(
                "SALVAGUARDA ACTIVA: ALLOW_PAID_ENDPOINTS=False.\n"
                "No se realizan llamadas al endpoint de partidos (/epl/v2/matches) hasta confirmación explícita."
            )
    else:
        logger.info("Modo demostración / verificación sin clave de red.")

    # 5. Ejecución del pipeline PySpark (si existen archivos en data/raw)
    spark = get_spark_session("EPL_Ingest_Run")
    try:
        t_path = process_teams_json_to_parquet(spark)
        s_path = process_standings_json_to_parquet(spark)
        m_path = process_matches_json_to_parquet(spark)
    finally:
        spark.stop()

    # 6. Registro y verificación analítica en DuckDB
    registered = register_parquet_in_duckdb()
    test_results = run_duckdb_test_queries()

    logger.info("=" * 70)
    logger.info("RESUMEN DE FASE 1 COMPLETADA")
    logger.info(f"Tablas registradas en DuckDB ({LAKEHOUSE_DB_PATH.name}): {registered}")
    logger.info(f"Resultados de prueba offline: {test_results}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
