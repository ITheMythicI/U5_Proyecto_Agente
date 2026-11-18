"""
Pipeline ETL con PySpark (Modo Local).
Transforma respuestas JSON crudas oficiales de BALLDONTLIE EPL V2 hacia Parquet
particionado por temporada y registra tablas analíticas en DuckDB.
Preserva estrictamente tipos de datos, marcadores 0 y valores NULL sin imputaciones arbitrarias.
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ajustar PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
    TimestampType
)
from pyspark.sql.window import Window
import duckdb

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LAKEHOUSE_DB_PATH
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkPipeline")


def get_spark_session(app_name: str = "EPL_Data_Lakehouse") -> SparkSession:
    """
    Crea o recupera una sesión local de PySpark optimizada para entorno mononodo.
    Utiliza local[*] para procesamiento multi-hilo en memoria local.
    """
    logger.info("Inicializando SparkSession local...")
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def process_matches_json_to_parquet(spark: SparkSession) -> str:
    """
    Procesa los 1,140 partidos oficiales consolidados (2022, 2023, 2024) hacia Parquet.
    Normaliza nombres y abreviaturas de clubes, preserva marcadores 0 y escribe
    particionado por temporada.
    """
    consolidated_files = [
        RAW_DATA_DIR / f"matches_{s}.json" for s in [2022, 2023, 2024]
    ]

    for cf in consolidated_files:
        if not cf.exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {cf}")

    logger.info(f"Leyendo 3 archivos consolidados de partidos ({len(consolidated_files)} temporadas)...")

    # Esquema estricto de entrada para lectura de JSON
    match_schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("season", IntegerType(), False),
        StructField("date", StringType(), False),
        StructField("name", StringType(), True),
        StructField("short_name", StringType(), True),
        StructField("status", StringType(), True),
        StructField("status_state", StringType(), True),
        StructField("status_detail", StringType(), True),
        StructField("home_team_id", IntegerType(), False),
        StructField("away_team_id", IntegerType(), False),
        StructField("home_score", IntegerType(), True),
        StructField("away_score", IntegerType(), True),
        StructField("attendance", IntegerType(), True),
        StructField("venue_name", StringType(), True),
        StructField("venue_city", StringType(), True)
    ])

    dfs: List[Any] = []
    for cf in consolidated_files:
        raw_df = spark.read.option("multiline", "true").json(str(cf))
        exploded = raw_df.select(F.explode("data").alias("m")).select("m.*")
        dfs.append(exploded)

    union_df = dfs[0]
    for other in dfs[1:]:
        union_df = union_df.unionByName(other)

    logger.info(f"Registros leídos desde JSON: {union_df.count()}")

    # Deduplicación por clave primaria oficial
    dedup_df = union_df.dropDuplicates(["id"])

    # Normalización y extracción de nombres y abreviaturas
    # name format: "Away Team at Home Team"
    # short_name format: "AWAY @ HOME"
    transformed_df = dedup_df.select(
        F.col("id").cast(IntegerType()).alias("match_id"),
        F.col("season").cast(IntegerType()).alias("season"),
        F.to_timestamp("date", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").alias("match_date"),
        F.col("name").alias("match_name"),
        F.col("short_name").alias("match_short_name"),
        F.col("home_team_id").cast(IntegerType()).alias("home_team_id"),
        F.trim(F.split(F.col("name"), " at ").getItem(1)).alias("home_team_name"),
        F.trim(F.split(F.col("short_name"), " @ ").getItem(1)).alias("home_team_abbr"),
        F.col("away_team_id").cast(IntegerType()).alias("away_team_id"),
        F.trim(F.split(F.col("name"), " at ").getItem(0)).alias("away_team_name"),
        F.trim(F.split(F.col("short_name"), " @ ").getItem(0)).alias("away_team_abbr"),
        F.col("home_score").cast(IntegerType()).alias("home_score"),
        F.col("away_score").cast(IntegerType()).alias("away_score"),
        F.col("status").alias("status"),
        F.col("status_state").alias("status_state"),
        F.col("status_detail").alias("status_detail"),
        F.col("attendance").cast(IntegerType()).alias("attendance"),
        F.col("venue_name").alias("venue_name"),
        F.col("venue_city").alias("venue_city")
    )

    # Cálculo de columnas analíticas
    # resultado_1x2: Local, Empate, Visita
    final_df = transformed_df.withColumn(
        "resultado_1x2",
        F.when(F.col("home_score").isNull() | F.col("away_score").isNull(), None)
         .when(F.col("home_score") > F.col("away_score"), "Local")
         .when(F.col("home_score") < F.col("away_score"), "Visita")
         .when(F.col("home_score") == F.col("away_score"), "Empate")
         .otherwise(None)
    ).withColumn(
        "total_goles",
        F.col("home_score") + F.col("away_score")
    ).withColumn(
        "diferencia_goles",
        F.col("home_score") - F.col("away_score")
    )

    output_path = str(PROCESSED_DATA_DIR / "matches.parquet")
    logger.info(f"Escribiendo partidos en Parquet particionado: {output_path}...")
    final_df.write.mode("overwrite").partitionBy("season").parquet(output_path)

    # Extracción y guardado del catálogo de clubes (teams.parquet)
    home_teams = final_df.select(
        F.col("home_team_id").alias("team_id"),
        F.col("home_team_name").alias("team_name"),
        F.col("home_team_abbr").alias("abbreviation")
    ).distinct()

    teams_path = str(PROCESSED_DATA_DIR / "teams.parquet")
    logger.info(f"Escribiendo catálogo de clubes en: {teams_path}...")
    home_teams.write.mode("overwrite").parquet(teams_path)

    logger.info("Transformación PySpark completada exitosamente.")
    return output_path


def register_parquet_in_duckdb() -> Dict[str, Any]:
    """
    Registra los archivos Parquet procesados como tablas en DuckDB (Lakehouse local).
    """
    logger.info(f"Conectando a DuckDB en: {LAKEHOUSE_DB_PATH}")
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH))
    registered = {}

    matches_parquet = PROCESSED_DATA_DIR / "matches.parquet"
    if matches_parquet.exists():
        con.execute(f"""
            CREATE OR REPLACE TABLE matches AS 
            SELECT * FROM read_parquet('{matches_parquet}/**/*.parquet', hive_partitioning=true)
        """)
        count = con.execute("SELECT count(*) FROM matches").fetchone()[0]
        registered["matches"] = count
        logger.info(f"DuckDB: Tabla 'matches' registrada con {count} registros.")

    teams_parquet = PROCESSED_DATA_DIR / "teams.parquet"
    if teams_parquet.exists():
        con.execute(f"""
            CREATE OR REPLACE TABLE teams AS 
            SELECT * FROM read_parquet('{teams_parquet}/*.parquet')
        """)
        count = con.execute("SELECT count(*) FROM teams").fetchone()[0]
        registered["teams"] = count
        logger.info(f"DuckDB: Tabla 'teams' registrada con {count} registros.")

    # Registrar matches_features si ya fue generado
    features_parquet = PROCESSED_DATA_DIR / "matches_features.parquet"
    if features_parquet.exists():
        con.execute(f"""
            CREATE OR REPLACE TABLE matches_features AS 
            SELECT * FROM read_parquet('{features_parquet}/**/*.parquet', hive_partitioning=true)
        """)
        count = con.execute("SELECT count(*) FROM matches_features").fetchone()[0]
        registered["matches_features"] = count
        logger.info(f"DuckDB: Tabla 'matches_features' registrada con {count} registros.")

    con.close()
    return registered


def run_duckdb_lakehouse_audit() -> Dict[str, Any]:
    """
    Ejecuta consultas de verificación analítica sobre DuckDB comprobando consistencia,
    nulos, distribuciones e integridad.
    """
    logger.info("Ejecutando consultas analíticas de verificación en DuckDB...")
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH), read_only=True)
    results = {}

    # 1. Conteo por temporada
    season_counts = con.execute("""
        SELECT season, count(*) AS total_partidos,
               count(home_score) AS partidos_con_marcador,
               sum(CASE WHEN home_score = 0 THEN 1 ELSE 0 END) AS local_cero,
               sum(CASE WHEN away_score = 0 THEN 1 ELSE 0 END) AS visita_cero,
               sum(CASE WHEN home_score = 0 AND away_score = 0 THEN 1 ELSE 0 END) AS empates_0_0
        FROM matches
        GROUP BY season
        ORDER BY season ASC
    """).fetchdf()
    results["conteo_temporadas"] = season_counts.to_dict(orient="records")

    # 2. Distribución de resultados 1X2
    dist_1x2 = con.execute("""
        SELECT season, resultado_1x2, count(*) AS total,
               round(count(*) * 100.0 / 380, 2) AS porcentaje
        FROM matches
        GROUP BY season, resultado_1x2
        ORDER BY season ASC, total DESC
    """).fetchdf()
    results["distribucion_1x2"] = dist_1x2.to_dict(orient="records")

    # 3. Comprobación de anomalías
    anomalies = con.execute("""
        SELECT count(*) 
        FROM matches
        WHERE home_team_id = away_team_id
           OR home_score < 0 
           OR away_score < 0
           OR resultado_1x2 IS NULL
    """).fetchone()[0]
    results["anomalias_detectadas"] = anomalies

    # 4. Tabla de posiciones recalculada con SQL (Comprobación de agregación)
    standings_2023 = con.execute("""
        WITH participaciones AS (
            SELECT season, home_team_name AS team, home_score AS gf, away_score AS ga,
                   CASE WHEN resultado_1x2 = 'Local' THEN 3 WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS pts,
                   CASE WHEN resultado_1x2 = 'Local' THEN 1 ELSE 0 END AS w,
                   CASE WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS d,
                   CASE WHEN resultado_1x2 = 'Visita' THEN 1 ELSE 0 END AS l
            FROM matches
            WHERE season = 2023
            UNION ALL
            SELECT season, away_team_name AS team, away_score AS gf, home_score AS ga,
                   CASE WHEN resultado_1x2 = 'Visita' THEN 3 WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS pts,
                   CASE WHEN resultado_1x2 = 'Visita' THEN 1 ELSE 0 END AS w,
                   CASE WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS d,
                   CASE WHEN resultado_1x2 = 'Local' THEN 1 ELSE 0 END AS l
            FROM matches
            WHERE season = 2023
        )
        SELECT team, count(*) as pj, sum(w) as v, sum(d) as e, sum(l) as d,
               sum(gf) as gf, sum(ga) as ga, sum(gf) - sum(ga) as dg, sum(pts) as pts
        FROM participaciones
        GROUP BY team
        ORDER BY pts DESC, dg DESC, gf DESC
        LIMIT 5
    """).fetchdf()
    results["top5_standings_2023"] = standings_2023.to_dict(orient="records")

    con.close()
    return results


def run_full_etl_pipeline() -> Dict[str, Any]:
    """Ejecuta el pipeline ETL completo con PySpark y valida en DuckDB."""
    spark = get_spark_session()
    try:
        matches_path = process_matches_json_to_parquet(spark)
        registered = register_parquet_in_duckdb()
        audit_results = run_duckdb_lakehouse_audit()
        return {
            "status": "success",
            "matches_parquet": matches_path,
            "registered": registered,
            "audit_results": audit_results
        }
    finally:
        spark.stop()
        logger.info("SparkSession detenida correctamente.")


if __name__ == "__main__":
    res = run_full_etl_pipeline()
    print("\n" + "=" * 80)
    print("ETL PYSPARK Y REGISTRO LAKEHOUSE COMPLETADO")
    print(f"Tablas registradas: {res['registered']}")
    print("=" * 80)
