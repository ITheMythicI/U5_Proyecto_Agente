"""
Pipeline de Ingeniería de Características y Preparación para ML Supervisado.
Genera variables temporales libres de fuga de información (Zero Data Leakage),
diseña la partición temporal estricta y evalúa el desbalance de clases.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple

import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# Ajustar PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, LAKEHOUSE_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeaturePipeline")


def generate_match_features() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Genera el dataset de características a partir de matches.parquet en DuckDB.
    Garantiza causalidad estricta: para predecir el partido M_t, solo se utilizan
    partidos ocurridos en tiempos t_prev < t.
    """
    logger.info("Cargando partidos ordenados cronológicamente desde DuckDB...")
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH), read_only=True)
    query = """
        SELECT match_id, season, match_date, home_team_id, away_team_id,
               home_team_name, away_team_name, home_score, away_score,
               resultado_1x2, attendance
        FROM matches
        ORDER BY match_date ASC, match_id ASC
    """
    df = con.execute(query).fetchdf()
    con.close()

    total_matches = len(df)
    logger.info(f"Total partidos a procesar: {total_matches}")

    # Estructuras de seguimiento de estado acumulado por equipo
    team_history = defaultdict(lambda: deque(maxlen=5))  # Últimos 5 partidos: (pts, gf, ga)
    team_season_stats = defaultdict(lambda: defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "pj": 0, "wins": 0}))
    team_last_date = {}  # Última fecha jugada por equipo

    feature_rows = []

    for idx, row in df.iterrows():
        mid = row["match_id"]
        season = row["season"]
        mdate = row["match_date"]
        hid = row["home_team_id"]
        aid = row["away_team_id"]
        h_score = row["home_score"]
        a_score = row["away_score"]
        target = row["resultado_1x2"]

        # -------------------------------------------------------------
        # 1. FORMA RECIENTE (ÚLTIMOS 5 PARTIDOS JUGADOS)
        # -------------------------------------------------------------
        h_hist = list(team_history[hid])
        a_hist = list(team_history[aid])

        home_pts_ult5 = sum(m[0] for m in h_hist) if h_hist else 0
        home_gf_ult5 = sum(m[1] for m in h_hist) if h_hist else 0
        home_ga_ult5 = sum(m[2] for m in h_hist) if h_hist else 0
        home_gd_ult5 = home_gf_ult5 - home_ga_ult5
        home_games_tracked = len(h_hist)

        away_pts_ult5 = sum(m[0] for m in a_hist) if a_hist else 0
        away_gf_ult5 = sum(m[1] for m in a_hist) if a_hist else 0
        away_ga_ult5 = sum(m[2] for m in a_hist) if a_hist else 0
        away_gd_ult5 = away_gf_ult5 - away_ga_ult5
        away_games_tracked = len(a_hist)

        diff_pts_ult5 = home_pts_ult5 - away_pts_ult5
        diff_gd_ult5 = home_gd_ult5 - away_gd_ult5

        # -------------------------------------------------------------
        # 2. RENDIMIENTO ACUMULADO EN LA TEMPORADA ACTUAL
        # -------------------------------------------------------------
        h_s_stat = team_season_stats[season][hid]
        a_s_stat = team_season_stats[season][aid]

        home_cum_pts = h_s_stat["pts"]
        home_cum_gf = h_s_stat["gf"]
        home_cum_ga = h_s_stat["ga"]
        home_cum_pj = h_s_stat["pj"]
        home_cum_ppg = round(home_cum_pts / home_cum_pj, 3) if home_cum_pj > 0 else 0.0

        away_cum_pts = a_s_stat["pts"]
        away_cum_gf = a_s_stat["gf"]
        away_cum_ga = a_s_stat["ga"]
        away_cum_pj = a_s_stat["pj"]
        away_cum_ppg = round(away_cum_pts / away_cum_pj, 3) if away_cum_pj > 0 else 0.0

        diff_cum_ppg = round(home_cum_ppg - away_cum_ppg, 3)

        # -------------------------------------------------------------
        # 3. DÍAS DE DESCANSO (REST DAYS)
        # -------------------------------------------------------------
        # Si es el primer partido de la temporada, se asigna 14 días estándar de pretemporada
        if hid in team_last_date:
            home_rest_days = min((mdate - team_last_date[hid]).days, 30)
        else:
            home_rest_days = 14

        if aid in team_last_date:
            away_rest_days = min((mdate - team_last_date[aid]).days, 30)
        else:
            away_rest_days = 14

        diff_rest_days = home_rest_days - away_rest_days

        # -------------------------------------------------------------
        # 4. PARTICIÓN TEMPORAL ESTRICTA (ZERO DATA LEAKAGE)
        # -------------------------------------------------------------
        # Train: Temporadas 2022 y 2023 (760 partidos)
        # Test: Temporada 2024 (380 partidos)
        split_set = "train" if season in (2022, 2023) else "test"

        feature_row = {
            "match_id": mid,
            "season": season,
            "match_date": mdate,
            "home_team_id": hid,
            "home_team_name": row["home_team_name"],
            "away_team_id": aid,
            "away_team_name": row["away_team_name"],
            # Variables predictoras (Features)
            "home_pts_ult5": home_pts_ult5,
            "home_gf_ult5": home_gf_ult5,
            "home_ga_ult5": home_ga_ult5,
            "home_gd_ult5": home_gd_ult5,
            "home_games_tracked": home_games_tracked,
            "away_pts_ult5": away_pts_ult5,
            "away_gf_ult5": away_gf_ult5,
            "away_ga_ult5": away_ga_ult5,
            "away_gd_ult5": away_gd_ult5,
            "away_games_tracked": away_games_tracked,
            "diff_pts_ult5": diff_pts_ult5,
            "diff_gd_ult5": diff_gd_ult5,
            "home_cum_pts": home_cum_pts,
            "home_cum_gf": home_cum_gf,
            "home_cum_ga": home_cum_ga,
            "home_cum_pj": home_cum_pj,
            "home_cum_ppg": home_cum_ppg,
            "away_cum_pts": away_cum_pts,
            "away_cum_gf": away_cum_gf,
            "away_cum_ga": away_cum_ga,
            "away_cum_pj": away_cum_pj,
            "away_cum_ppg": away_cum_ppg,
            "diff_cum_ppg": diff_cum_ppg,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days,
            "diff_rest_days": diff_rest_days,
            # Variable Objetivo y Marcadores
            "home_score": h_score,
            "away_score": a_score,
            "resultado_1x2": target,
            "split_set": split_set
        }
        feature_rows.append(feature_row)

        # -------------------------------------------------------------
        # ACTUALIZACIÓN DE ESTADO POST-PARTIDO (Para partidos futuros)
        # -------------------------------------------------------------
        if h_score is not None and a_score is not None:
            if h_score > a_score:
                h_pts, a_pts = 3, 0
            elif h_score < a_score:
                h_pts, a_pts = 0, 3
            else:
                h_pts, a_pts = 1, 1

            team_history[hid].append((h_pts, h_score, a_score))
            team_history[aid].append((a_pts, a_score, h_score))

            h_s_stat["pts"] += h_pts
            h_s_stat["gf"] += h_score
            h_s_stat["ga"] += a_score
            h_s_stat["pj"] += 1
            if h_pts == 3: h_s_stat["wins"] += 1

            a_s_stat["pts"] += a_pts
            a_s_stat["gf"] += a_score
            a_s_stat["ga"] += h_score
            a_s_stat["pj"] += 1
            if a_pts == 3: a_s_stat["wins"] += 1

            team_last_date[hid] = mdate
            team_last_date[aid] = mdate

    feat_df = pd.DataFrame(feature_rows)

    # -----------------------------------------------------------------
    # EVALUACIÓN DE DESBALANCE DE CLASES
    # -----------------------------------------------------------------
    train_df = feat_df[feat_df["split_set"] == "train"]
    test_df = feat_df[feat_df["split_set"] == "test"]

    train_dist = train_df["resultado_1x2"].value_counts()
    test_dist = test_df["resultado_1x2"].value_counts()

    # Cálculo formal de pesos de clase: w_c = N_train / (K * N_c)
    n_train = len(train_df)
    k_classes = len(train_dist)
    class_weights = {}
    for c_name, c_count in train_dist.items():
        w = round(n_train / (k_classes * c_count), 4)
        class_weights[c_name] = w

    balance_eval = {
        "n_train_total": n_train,
        "n_test_total": len(test_df),
        "train_distribution": {
            k: {"count": int(v), "pct": round(v / n_train * 100, 2)} for k, v in train_dist.items()
        },
        "test_distribution": {
            k: {"count": int(v), "pct": round(v / len(test_df) * 100, 2)} for k, v in test_dist.items()
        },
        "calculated_class_weights": class_weights,
        "ratio_local_vs_empate": round(train_dist.get("Local", 1) / train_dist.get("Empate", 1), 2),
        "dictamen_ponderacion": (
            "Se justifica técnicamente el uso de ponderación de clases (class_weight='balanced') "
            f"debido a que la clase 'Empate' representa solo el {round(train_dist.get('Empate', 0)/n_train*100, 1)}% "
            f"del conjunto de entrenamiento (ratio de desbalance {round(train_dist.get('Local', 1)/train_dist.get('Empate', 1), 1)}:1 frente a Local)."
        )
    }

    # -----------------------------------------------------------------
    # EXPORTACIÓN A PARQUET PARTICIONADO
    # -----------------------------------------------------------------
    output_path = PROCESSED_DATA_DIR / "matches_features.parquet"
    logger.info(f"Guardando dataset de características en: {output_path}...")
    table = pa.Table.from_pandas(feat_df)
    pq.write_to_dataset(
        table,
        root_path=str(output_path),
        partition_cols=["season"]
    )
    logger.info(f"Dataset de características exportado exitosamente con {len(feat_df)} registros.")

    # Registrar en DuckDB
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH))
    con.execute(f"""
        CREATE OR REPLACE TABLE matches_features AS 
        SELECT * FROM read_parquet('{output_path}/**/*.parquet', hive_partitioning=true)
    """)
    con.close()
    logger.info("Tabla 'matches_features' registrada en DuckDB.")

    return feat_df, balance_eval


if __name__ == "__main__":
    df_feat, eval_data = generate_match_features()
    print("\n" + "=" * 80)
    print("RESUMEN DE PREPARACIÓN DE DATASET (SUPERVISADO)")
    print("=" * 80)
    print(f"Total registros: {len(df_feat)}")
    print(f"Columnas generadas ({len(df_feat.columns)}): {list(df_feat.columns)}")
    print("\nDistribución Conjunto Entrenamiento (Train - 2022 y 2023):")
    for k, v in eval_data["train_distribution"].items():
        print(f"  • {k:8s}: {v['count']} ({v['pct']}%) | Peso de clase calculado: {eval_data['calculated_class_weights'][k]}")
    print("\nDistribución Conjunto Prueba (Test - 2024):")
    for k, v in eval_data["test_distribution"].items():
        print(f"  • {k:8s}: {v['count']} ({v['pct']}%)")
    print(f"\nDictamen: {eval_data['dictamen_ponderacion']}")
    print("=" * 80)
