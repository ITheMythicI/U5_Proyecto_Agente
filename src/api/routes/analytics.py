"""
Rutas de analítica descriptiva global, métricas por temporada y generación de gráficos.
"""

import io
import base64
from fastapi import APIRouter, HTTPException
import duckdb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ..schemas import (
    AnalyticsOverviewResponse, SeasonsSummaryResponse,
    SeasonSummaryItem, ChartRequest, ChartResponse
)
from ...agent.tools import generate_chart
from ...agent.config import config

router = APIRouter(prefix="/api/v1/analytics", tags=["Analítica y Visualizaciones"])


@router.get("/overview", response_model=AnalyticsOverviewResponse, summary="Resumen global de los 1,140 partidos")
def get_analytics_overview():
    """Retorna indicadores agregados sobre los 1,140 partidos de las tres temporadas analizadas."""
    con = duckdb.connect(str(config.lakehouse_db_path), read_only=True)
    df_matches = con.execute("""
        SELECT 
            count(*) as total_matches,
            list(DISTINCT season ORDER BY season) as seasons,
            sum(home_score + away_score) as total_goals,
            avg(home_score + away_score) as avg_goals,
            sum(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins,
            sum(CASE WHEN home_score = away_score THEN 1 ELSE 0 END) as draws,
            sum(CASE WHEN home_score < away_score THEN 1 ELSE 0 END) as away_wins
        FROM matches
        WHERE status IN ('STATUS_FULL_TIME', 'Final')
    """).fetchdf()

    teams_cnt = con.execute("SELECT count(DISTINCT team_name) FROM teams").fetchone()[0]
    con.close()

    total = int(df_matches["total_matches"][0])
    total_goals = int(df_matches["total_goals"][0])
    avg_goals = round(float(df_matches["avg_goals"][0]), 2)
    h_wins = int(df_matches["home_wins"][0])
    draws = int(df_matches["draws"][0])
    a_wins = int(df_matches["away_wins"][0])

    return AnalyticsOverviewResponse(
        status="success",
        total_matches=total,
        seasons=list(df_matches["seasons"][0]),
        total_goals=total_goals,
        avg_goals_per_match=avg_goals,
        home_wins_pct=round((h_wins / total) * 100, 2),
        draws_pct=round((draws / total) * 100, 2),
        away_wins_pct=round((a_wins / total) * 100, 2),
        teams_count=int(teams_cnt)
    )


@router.get("/seasons", response_model=SeasonsSummaryResponse, summary="Desglose comparativo por temporada")
def get_seasons_summary():
    """Retorna estadísticas clave y el campeón de cada temporada analizada (2022, 2023, 2024)."""
    con = duckdb.connect(str(config.lakehouse_db_path), read_only=True)

    df_seasons = con.execute("""
        SELECT 
            season,
            count(*) as matches_count,
            sum(home_score + away_score) as total_goals,
            round(avg(home_score + away_score), 2) as avg_goals,
            sum(CASE WHEN home_score > away_score THEN 1 ELSE 0 END) as home_wins,
            sum(CASE WHEN home_score = away_score THEN 1 ELSE 0 END) as draws,
            sum(CASE WHEN home_score < away_score THEN 1 ELSE 0 END) as away_wins
        FROM matches
        WHERE status IN ('STATUS_FULL_TIME', 'Final')
        GROUP BY season
        ORDER BY season ASC
    """).fetchdf()

    champions_df = con.execute("""
        SELECT season, team_name as champion_team, puntos as champion_points
        FROM (
            SELECT season, team_name, puntos,
                   row_number() OVER (PARTITION BY season ORDER BY puntos DESC, diferencia_goles DESC) as rn
            FROM teams_clustered
        )
        WHERE rn = 1
        ORDER BY season ASC
    """).fetchdf().set_index("season")

    con.close()

    items = []
    for _, row in df_seasons.iterrows():
        s = int(row["season"])
        champ_info = champions_df.loc[s] if s in champions_df.index else None
        champ_name = champ_info["champion_team"] if champ_info is not None else "Desconocido"
        champ_pts = int(champ_info["champion_points"]) if champ_info is not None else 0

        items.append(SeasonSummaryItem(
            season=s,
            matches_count=int(row["matches_count"]),
            total_goals=int(row["total_goals"]),
            avg_goals=float(row["avg_goals"]),
            home_wins=int(row["home_wins"]),
            draws=int(row["draws"]),
            away_wins=int(row["away_wins"]),
            champion_team=champ_name,
            champion_points=champ_pts
        ))

    return SeasonsSummaryResponse(
        status="success",
        seasons=items
    )


# Plantillas de consultas SQL analíticas de confianza pre-aprobadas
# El endpoint NO acepta cadenas SQL libres del cliente por razones de seguridad.
TRUSTED_PRESET_QUERIES = {
    "goals_by_season": """
        SELECT CAST(season AS VARCHAR) as season_label, sum(home_score + away_score) as total_goles
        FROM matches
        WHERE status IN ('STATUS_FULL_TIME', 'Final')
        GROUP BY season
        ORDER BY season ASC
    """,
    "top_scoring_teams": """
        SELECT team_name, sum(goles_favor) as total_gf
        FROM teams_clustered
        WHERE season = 2024
        GROUP BY team_name
        ORDER BY total_gf DESC
        LIMIT 10
    """,
    "points_distribution": """
        SELECT team_name, puntos
        FROM teams_clustered
        WHERE season = 2024
        ORDER BY puntos DESC
    """,
    "cluster_summary": """
        SELECT cluster_name, count(*) as num_clubes
        FROM teams_clustered
        WHERE season = 2024
        GROUP BY cluster_name
    """
}


@router.post("/chart", response_model=ChartResponse, summary="Generar gráfico estadístico en Base64")
def create_chart_endpoint(payload: ChartRequest):
    """
    Genera un gráfico estadístico seguro (barras, dispersión o línea) codificado en Base64.
    Por directriz de seguridad, este endpoint RECHAZA cualquier SQL arbitrario.
    Admite exclusivamente:
    1. Una plantilla analítica autorizada (`preset`).
    2. O un conjunto de datos numéricos estructurados (`data`).
    """
    # Caso 1: Generación a partir de plantilla autorizada (preset seguro)
    if payload.preset:
        if payload.preset not in TRUSTED_PRESET_QUERIES:
            raise HTTPException(
                status_code=400,
                detail=f"Preset '{payload.preset}' no autorizado. Opciones válidas: {list(TRUSTED_PRESET_QUERIES.keys())}"
            )
        
        query = TRUSTED_PRESET_QUERIES[payload.preset]
        filename = f"chart_{payload.preset}_{payload.chart_type}.png"
        res = generate_chart(
            chart_type=payload.chart_type,
            sql_query=query,
            title=payload.title,
            output_filename=filename
        )
        if res.get("status") != "success":
            raise HTTPException(status_code=400, detail=res.get("message", "Error generando el gráfico."))
        
        with open(res["output_path"], "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        return ChartResponse(
            status="success",
            chart_type=payload.chart_type,
            format="png",
            image_base64=b64_img
        )

    # Caso 2: Generación a partir de datos estructurados en payload.data
    if payload.data:
        data = payload.data
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            if payload.chart_type == "bar":
                cats = data.get("categories", list(data.keys()))
                vals = data.get("values", list(data.values()))
                sns.barplot(x=cats, y=vals, ax=ax, palette="Blues_d")
                plt.xticks(rotation=45, ha="right")
            elif payload.chart_type == "line":
                cats = data.get("categories", list(data.keys()))
                vals = data.get("values", list(data.values()))
                sns.lineplot(x=cats, y=vals, ax=ax, marker="o", color="#00ff87")
                plt.xticks(rotation=45, ha="right")
            elif payload.chart_type == "scatter":
                x = data.get("x", [])
                y = data.get("y", [])
                sns.scatterplot(x=x, y=y, ax=ax, s=70, color="#38bdf8")
            else:
                plt.close(fig)
                raise HTTPException(status_code=400, detail=f"Tipo de gráfico '{payload.chart_type}' no soportado.")

            ax.set_title(payload.title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel(payload.x_label or "")
            ax.set_ylabel(payload.y_label or "")
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160)
            plt.close(fig)
            buf.seek(0)
            b64_img = base64.b64encode(buf.read()).decode("utf-8")

            return ChartResponse(
                status="success",
                chart_type=payload.chart_type,
                format="png",
                image_base64=b64_img
            )
        except HTTPException:
            raise
        except Exception as e:
            plt.close(fig)
            raise HTTPException(status_code=400, detail=f"Error renderizando gráfico: {str(e)}")

    raise HTTPException(status_code=400, detail="Debe proporcionar un 'preset' autorizado o un diccionario 'data'.")
