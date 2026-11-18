"""
Pipeline de Aprendizaje No Supervisado: PCA y K-Means Clustering.
Segmenta el perfil de rendimiento de los clubes de Premier League (60 observaciones: 20 clubes x 3 temporadas).
Justifica la elección de k mediante Inercia (Codo) y Coeficiente de Silueta,
analiza las cargas factoriales de los componentes principales y genera conclusiones futbolísticas.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import duckdb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ajustar PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAKEHOUSE_DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("UnsupervisedModeling")

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

UNSUPERVISED_FEATURES = [
    "puntos", "goles_favor", "goles_contra", "diferencia_goles",
    "victorias", "empates", "derrotas"
]


def load_team_season_table() -> pd.DataFrame:
    """Extrae y calcula la tabla de rendimiento agregada de clubes por temporada desde DuckDB."""
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH), read_only=True)
    query = """
        WITH participaciones AS (
            SELECT season, home_team_id AS team_id, home_team_name AS team_name,
                   home_score AS gf, away_score AS ga,
                   CASE WHEN resultado_1x2 = 'Local' THEN 3 WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS pts,
                   CASE WHEN resultado_1x2 = 'Local' THEN 1 ELSE 0 END AS w,
                   CASE WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS d,
                   CASE WHEN resultado_1x2 = 'Visita' THEN 1 ELSE 0 END AS l
            FROM matches
            UNION ALL
            SELECT season, away_team_id AS team_id, away_team_name AS team_name,
                   away_score AS gf, home_score AS ga,
                   CASE WHEN resultado_1x2 = 'Visita' THEN 3 WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS pts,
                   CASE WHEN resultado_1x2 = 'Visita' THEN 1 ELSE 0 END AS w,
                   CASE WHEN resultado_1x2 = 'Empate' THEN 1 ELSE 0 END AS d,
                   CASE WHEN resultado_1x2 = 'Local' THEN 1 ELSE 0 END AS l
            FROM matches
        )
        SELECT 
            season,
            team_id,
            team_name,
            count(*) as partidos_jugados,
            sum(w) as victorias,
            sum(d) as empates,
            sum(l) as derrotas,
            sum(gf) as goles_favor,
            sum(ga) as goles_contra,
            sum(gf) - sum(ga) as diferencia_goles,
            sum(pts) as puntos,
            round(sum(pts) * 1.0 / count(*), 3) as puntos_por_partido
        FROM participaciones
        GROUP BY season, team_id, team_name
        ORDER BY season ASC, puntos DESC, diferencia_goles DESC
    """
    df = con.execute(query).fetchdf()
    con.close()
    logger.info(f"Tabla agregada de clubes cargada: {len(df)} observaciones (20 equipos x 3 temporadas = 60)")
    return df


def run_unsupervised_analysis() -> Dict[str, Any]:
    """Ejecuta el análisis completo de PCA y K-Means."""
    df = load_team_season_table()
    X = df[UNSUPERVISED_FEATURES].copy()

    # 1. Escalado con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA: Análisis de Componentes Principales
    pca_full = PCA()
    pca_full.fit(X_scaled)
    var_exp = pca_full.explained_variance_ratio_
    var_cum = np.cumsum(var_exp)

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    # Cargas factoriales (Loadings)
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        columns=["PC1", "PC2"],
        index=UNSUPERVISED_FEATURES
    )

    # 3. K-Means: Selección óptima de k
    k_range = list(range(2, 7))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        silhouettes.append(sil)

    # Entrenar K-Means con k=3 óptimo
    optimal_k = 3
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["cluster"] = kmeans_optimal.fit_predict(X_scaled)

    # Interpretación futbolística de clústeres
    cluster_profiles = df.groupby("cluster")[UNSUPERVISED_FEATURES].mean().reset_index()

    # Identificar nombres descriptivos por jerarquía de puntos
    cluster_rank_order = cluster_profiles.sort_values(by="puntos", ascending=False)["cluster"].tolist()
    cluster_labels_map = {
        cluster_rank_order[0]: "Contendientes al Título / Top 4",
        cluster_rank_order[1]: "Mitad de Tabla Consolidada",
        cluster_rank_order[2]: "Zona de Descenso / Lucha por Permanencia"
    }
    df["cluster_name"] = df["cluster"].map(cluster_labels_map)

    # Guardar artefactos
    joblib.dump(scaler, MODELS_DIR / "unsupervised_scaler.joblib")
    joblib.dump(pca_2d, MODELS_DIR / "pca_2d_model.joblib")
    joblib.dump(kmeans_optimal, MODELS_DIR / "kmeans_cluster_model.joblib")

    # Guardar tabla con clústeres en parquet y DuckDB
    output_parquet = PROJECT_ROOT / "data" / "processed" / "teams_clustered.parquet"
    df.to_parquet(output_parquet, index=False)

    con = duckdb.connect(str(LAKEHOUSE_DB_PATH))
    con.execute(f"CREATE OR REPLACE TABLE teams_clustered AS SELECT * FROM read_parquet('{output_parquet}')")
    con.close()
    logger.info(f"Tabla teams_clustered guardada en DuckDB ({len(df)} registros).")

    # Guardar métricas en JSON
    metrics = {
        "total_observaciones": len(df),
        "variables_utilizadas": UNSUPERVISED_FEATURES,
        "pca": {
            "varianza_explicada_pc1": round(float(var_exp[0]), 4),
            "varianza_explicada_pc2": round(float(var_exp[1]), 4),
            "varianza_acumulada_2_componentes": round(float(var_cum[1]), 4),
            "cargas_factoriales": loadings.to_dict()
        },
        "kmeans": {
            "evaluacion_k": [
                {"k": k, "inercia": round(inertias[i], 2), "silhouette": round(silhouettes[i], 4)}
                for i, k in enumerate(k_range)
            ],
            "k_optimo_seleccionado": optimal_k,
            "silhouette_k_optimo": round(silhouettes[k_range.index(optimal_k)], 4),
            "perfiles_promedio_cluster": cluster_profiles.to_dict(orient="records"),
            "etiquetas_cualitativas": cluster_labels_map
        }
    }

    metrics_path = MODELS_DIR / "unsupervised_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Generar gráficos diagnósticos
    plot_unsupervised_visualizations(df, var_exp, var_cum, k_range, inertias, silhouettes, loadings)

    return metrics


def plot_unsupervised_visualizations(df: pd.DataFrame, var_exp: np.ndarray, var_cum: np.ndarray,
                                     k_range: List[int], inertias: List[float],
                                     silhouettes: List[float], loadings: pd.DataFrame) -> None:
    """Genera gráficos profesionales no redundantes de PCA y K-Means."""
    sns.set_theme(style="whitegrid", palette="deep")

    # Figura 1: Codo (Inercia) y Silueta para justificación de k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, marker="o", color="#1f77b4", linewidth=2)
    ax1.set_title("Método del Codo (Inercia vs Número de Clústeres k)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Número de Clústeres (k)")
    ax1.set_ylabel("Inercia (Suma de Cuadrados Intra-Clúster)")
    ax1.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3 Óptimo")
    ax1.legend()

    ax2.plot(k_range, silhouettes, marker="s", color="#2ca02c", linewidth=2)
    ax2.set_title("Coeficiente de Silueta por Valor de k", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Número de Clústeres (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3 Óptimo")
    ax2.legend()

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "pca_variance_elbow.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Gráfico de selección de k guardado en: {fig1_path}")

    # Figura 2: Dispersión 2D PCA con Clústeres K-Means y Nombres de Clubes
    plt.figure(figsize=(12, 8))
    colors = {"Contendientes al Título / Top 4": "#2ca02c", 
              "Mitad de Tabla Consolidada": "#1f77b4", 
              "Zona de Descenso / Lucha por Permanencia": "#d62728"}

    for cname, grp in df.groupby("cluster_name"):
        plt.scatter(grp["PC1"], grp["PC2"], label=cname, s=90, alpha=0.8, color=colors.get(cname, "gray"))

    # Anotaciones selectivas de clubes emblemáticos para evitar sobrecarga visual
    sample_teams = ["Manchester City", "Arsenal", "Liverpool", "Aston Villa",
                    "Sheffield United", "Luton Town", "Burnley", "Everton", "Fulham", "Chelsea"]
    for _, row in df[df["team_name"].isin(sample_teams)].iterrows():
        plt.annotate(
            f"{row['team_name']} ('{str(row['season'])[-2:]})",
            (row["PC1"] + 0.12, row["PC2"] + 0.08),
            fontsize=8, alpha=0.85
        )

    pc1_pct = var_exp[0] * 100
    pc2_pct = var_exp[1] * 100
    plt.title(f"Proyección PCA 2D y Clústeres K-Means (k=3)\nVarianza Explicada Acumulada: {pc1_pct+pc2_pct:.1f}% (PC1: {pc1_pct:.1f}%, PC2: {pc2_pct:.1f}%)",
              fontsize=13, fontweight="bold")
    plt.xlabel(f"Componente Principal 1 (Dominancia Ofensiva y Puntos - {pc1_pct:.1f}%)", fontsize=11)
    plt.ylabel(f"Componente Principal 2 (Equilibrio Defensivo y Empates - {pc2_pct:.1f}%)", fontsize=11)
    plt.axhline(0, color="gray", linestyle=":", alpha=0.6)
    plt.axvline(0, color="gray", linestyle=":", alpha=0.6)
    plt.legend(title="Perfil Competitivo", loc="upper right")
    plt.tight_layout()

    fig2_path = FIGURES_DIR / "pca_team_clusters.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Dispersión PCA de clubes guardada en: {fig2_path}")


if __name__ == "__main__":
    res = run_unsupervised_analysis()
    print("\n" + "=" * 80)
    print("RESUMEN DE APRENDIZAJE NO SUPERVISADO (PCA + K-MEANS)")
    print("=" * 80)
    pca_m = res["pca"]
    print(f"Varianza PC1: {pca_m['varianza_explicada_pc1']*100:.2f}% | PC2: {pca_m['varianza_explicada_pc2']*100:.2f}% | Acumulada: {pca_m['varianza_acumulada_2_componentes']*100:.2f}%")
    km_m = res["kmeans"]
    print(f"\nK-Means Óptimo: k = {km_m['k_optimo_seleccionado']} (Silhouette: {km_m['silhouette_k_optimo']:.4f})")
    print("\nPerfiles Promedio de Clústeres:")
    for prof in km_m["perfiles_promedio_cluster"]:
        cid = prof["cluster"]
        cname = km_m["etiquetas_cualitativas"].get(cid, str(cid))
        print(f"  • Clúster {cid} ({cname}):")
        print(f"    Puntos: {prof['puntos']:.1f} | Victorias: {prof['victorias']:.1f} | Empates: {prof['empates']:.1f} | Derrotas: {prof['derrotas']:.1f} | Dif. Goles: {prof['diferencia_goles']:+.1f}")
    print("=" * 80)
