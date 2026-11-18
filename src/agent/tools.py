"""
Módulo de Herramientas Analíticas del Agente Deportivo.
Proporciona funciones estructuradas para DuckDB, predicción con modelos entrenados,
perfiles de clubes (PCA + K-Means) y generación de gráficos estadísticos.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import duckdb
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para entornos de servidor
import matplotlib.pyplot as plt
import seaborn as sns

from .config import config

# Catálogo canónico de equipos y sinónimos comunes
CANONICAL_TEAMS = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton",
    "brighton & hove albion": "Brighton",
    "burnley": "Burnley",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    "liverpool": "Liverpool",
    "luton": "Luton Town",
    "luton town": "Luton Town",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "city": "Manchester City",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "united": "Manchester United",
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "nottingham": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    "sheffield": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "sheffield united": "Sheffield United",
    "southampton": "Southampton",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "west ham": "West Ham United",
    "west ham united": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers"
}

FEATURE_COLS = [
    "home_pts_ult5", "home_gf_ult5", "home_ga_ult5", "home_gd_ult5", "home_games_tracked",
    "away_pts_ult5", "away_gf_ult5", "away_ga_ult5", "away_gd_ult5", "away_games_tracked",
    "diff_pts_ult5", "diff_gd_ult5",
    "home_cum_pts", "home_cum_gf", "home_cum_ga", "home_cum_pj", "home_cum_ppg",
    "away_cum_pts", "away_cum_gf", "away_ga_cum" if False else "away_cum_ga",
    "away_cum_pj", "away_cum_ppg",
    "diff_cum_ppg",
    "home_rest_days", "away_rest_days", "diff_rest_days"
]

CLASSES = ["Empate", "Local", "Visita"]


def resolve_team_name(name: str) -> Optional[str]:
    """Resuelve un nombre informal de club a su nomenclatura canónica."""
    cleaned = name.strip().lower()
    if cleaned in CANONICAL_TEAMS:
        return CANONICAL_TEAMS[cleaned]
    # Búsqueda por subcadena
    for alias, canonical in CANONICAL_TEAMS.items():
        if alias in cleaned or cleaned in alias:
            return canonical
    return None


def query_duckdb(sql_query: str) -> Dict[str, Any]:
    """
    Ejecuta una consulta SQL en modo de solo lectura sobre el Lakehouse DuckDB.
    Bloquea sentencias DDL/DML destructivas y limita los registros devueltos.
    """
    # Validación de seguridad: solo lectura, sin DDL/DML, sin acceso a archivos externos ni extensiones
    forbidden_exact = [
        "insert", "update", "delete", "drop", "truncate", "alter",
        "create", "attach", "detach", "copy", "export", "pragma",
        "install", "load", "checkpoint", "vacuum"
    ]
    forbidden_patterns = [
        "read_parquet", "read_csv", "read_json", "scan_parquet",
        "parquet_scan", "httpfs", "glob("
    ]

    query_lower = sql_query.lower()
    for kw in forbidden_exact:
        if re.search(rf"\b{kw}\b", query_lower):
            return {
                "status": "error",
                "message": f"Operación no permitida: el comando '{kw.upper()}' está estrictamente restringido por política de seguridad del Lakehouse. Solo se permiten consultas de lectura (SELECT) sobre las tablas registradas.",
                "sql_query": sql_query
            }

    for pat in forbidden_patterns:
        if pat in query_lower:
            return {
                "status": "error",
                "message": f"Operación no permitida: la función '{pat}' está restringida para evitar acceso no controlado a archivos fuera del Lakehouse.",
                "sql_query": sql_query
            }

    try:
        con = duckdb.connect(str(config.lakehouse_db_path), read_only=True)
        # Limitar a 50 filas si no incluye LIMIT
        executed_query = sql_query.strip()
        if not re.search(r"\blimit\b", query_lower):
            executed_query = f"{executed_query.rstrip(';')} LIMIT 50"

        df = con.execute(executed_query).fetchdf()
        con.close()

        # Convertir timestamps a ISO strings
        records = df.to_dict(orient="records")
        for r in records:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()

        return {
            "status": "success",
            "sql_query": executed_query,
            "row_count": len(df),
            "columns": list(df.columns),
            "rows": records
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error ejecutando SQL: {str(e)}",
            "sql_query": sql_query
        }


def predict_match(home_team: str, away_team: str, model_type: str = "champion") -> Dict[str, Any]:
    """
    Ejecuta una predicción deportiva 1X2 utilizando los modelos supervisados entrenados.
    Calcula probabilidades para Local, Empate y Visita, y explica los factores determinantes.
    
    Args:
        home_team: Nombre del equipo local.
        away_team: Nombre del equipo visitante.
        model_type: 'champion' (Regresión Logística Balanceada) o 'automl' (FLAML LRL1).
    """
    c_home = resolve_team_name(home_team)
    c_away = resolve_team_name(away_team)

    if not c_home or not c_away:
        return {
            "status": "error",
            "message": f"No se reconocen los nombres de los equipos. Local detectado: '{c_home}', Visita detectado: '{c_away}'."
        }

    if c_home == c_away:
        return {
            "status": "error",
            "message": f"Un equipo no puede jugar contra sí mismo ({c_home})."
        }

    # Seleccionar modelo
    if model_type == "automl":
        model_file = config.models_dir / "flaml_automl_best.joblib"
        model_name = "FLAML AutoML (LRL1)"
    else:
        model_file = config.models_dir / "logistic_regression_balanced.joblib"
        model_name = "Regresión Logística Multinomial Balanceada (Campeón)"

    if not model_file.exists():
        return {"status": "error", "message": f"Archivo de modelo no encontrado: {model_file}"}

    model = joblib.load(model_file)

    # Extraer las últimas características registradas en el Lakehouse para ambos clubes
    con = duckdb.connect(str(config.lakehouse_db_path), read_only=True)

    # Métricas más recientes jugando de local para c_home
    home_stats = con.execute(f"""
        SELECT home_pts_ult5, home_gf_ult5, home_ga_ult5, home_gd_ult5, home_games_tracked,
               home_cum_pts, home_cum_gf, home_cum_ga, home_cum_pj, home_cum_ppg, home_rest_days
        FROM matches_features
        WHERE home_team_name = '{c_home}'
        ORDER BY match_date DESC
        LIMIT 1
    """).fetchall()

    # Métricas más recientes jugando de visitante para c_away
    away_stats = con.execute(f"""
        SELECT away_pts_ult5, away_gf_ult5, away_ga_ult5, away_gd_ult5, away_games_tracked,
               away_cum_pts, away_cum_gf, away_cum_ga, away_cum_pj, away_cum_ppg, away_rest_days
        FROM matches_features
        WHERE away_team_name = '{c_away}'
        ORDER BY match_date DESC
        LIMIT 1
    """).fetchall()
    con.close()

    if not home_stats or not away_stats:
        return {
            "status": "error",
            "message": f"No se encontraron suficientes datos históricos recientes para {c_home} o {c_away}."
        }

    h = home_stats[0]
    a = away_stats[0]

    # Construir el vector de 26 variables pre-partido
    h_pts_5, h_gf_5, h_ga_5, h_gd_5, h_tracked = h[0], h[1], h[2], h[3], h[4]
    h_pts_cum, h_gf_cum, h_ga_cum, h_pj_cum, h_ppg_cum, h_rest = h[5], h[6], h[7], h[8], h[9], h[10]

    a_pts_5, a_gf_5, a_ga_5, a_gd_5, a_tracked = a[0], a[1], a[2], a[3], a[4]
    a_pts_cum, a_gf_cum, a_ga_cum, a_pj_cum, a_ppg_cum, a_rest = a[5], a[6], a[7], a[8], a[9], a[10]

    diff_pts_5 = h_pts_5 - a_pts_5
    diff_gd_5 = h_gd_5 - a_gd_5
    diff_ppg_cum = h_ppg_cum - a_ppg_cum
    diff_rest = h_rest - a_rest

    feature_dict = {
        "home_pts_ult5": h_pts_5,
        "home_gf_ult5": h_gf_5,
        "home_ga_ult5": h_ga_5,
        "home_gd_ult5": h_gd_5,
        "home_games_tracked": h_tracked,
        "away_pts_ult5": a_pts_5,
        "away_gf_ult5": a_gf_5,
        "away_ga_ult5": a_ga_5,
        "away_gd_ult5": a_gd_5,
        "away_games_tracked": a_tracked,
        "diff_pts_ult5": diff_pts_5,
        "diff_gd_ult5": diff_gd_5,
        "home_cum_pts": h_pts_cum,
        "home_cum_gf": h_gf_cum,
        "home_cum_ga": h_ga_cum,
        "home_cum_pj": h_pj_cum,
        "home_cum_ppg": h_ppg_cum,
        "away_cum_pts": a_pts_cum,
        "away_cum_gf": a_gf_cum,
        "away_cum_ga": a_ga_cum,
        "away_cum_pj": a_pj_cum,
        "away_cum_ppg": a_ppg_cum,
        "diff_cum_ppg": diff_ppg_cum,
        "home_rest_days": h_rest,
        "away_rest_days": a_rest,
        "diff_rest_days": diff_rest
    }

    X_pred = pd.DataFrame([feature_dict])

    # Inferencia
    y_proba = model.predict_proba(X_pred)[0]
    classes_order = list(model.classes_)  # ['Empate', 'Local', 'Visita']

    prob_dict = {
        cls_name: round(float(y_proba[classes_order.index(cls_name)]), 4)
        for cls_name in ["Local", "Empate", "Visita"]
    }

    pred_class = max(prob_dict, key=prob_dict.get)

    # Explicación analítica de factores determinantes
    factors = []
    if diff_ppg_cum > 0.4:
        factors.append(f"Ventaja acumulada de temporada a favor de {c_home} (+{diff_ppg_cum:.2f} pts por partido).")
    elif diff_ppg_cum < -0.4:
        factors.append(f"Ventaja acumulada de temporada a favor de {c_away} ({-diff_ppg_cum:.2f} pts por partido más).")
    else:
        factors.append("Rendimiento acumulado de temporada muy parejo entre ambos clubes.")

    if diff_pts_5 >= 4:
        factors.append(f"{c_home} llega en mejor racha reciente (+{diff_pts_5} puntos en los últimos 5 partidos).")
    elif diff_pts_5 <= -4:
        factors.append(f"{c_away} llega en mejor racha reciente ({-diff_pts_5} puntos más en últimos 5 partidos).")

    if abs(diff_ppg_cum) < 0.2 and abs(diff_pts_5) <= 2:
        factors.append("Alta paridad en forma reciente y acumulada, lo que eleva la probabilidad estimada de Empate.")

    return {
        "status": "success",
        "home_team": c_home,
        "away_team": c_away,
        "modelo_utilizado": model_name,
        "probabilidades": prob_dict,
        "prediccion_estimada": pred_class,
        "confianza_maxima": prob_dict[pred_class],
        "factores_clave": factors,
        "metricas_prepartido": {
            "home_ppg_acumulado": round(h_ppg_cum, 2),
            "away_ppg_acumulado": round(a_ppg_cum, 2),
            "home_pts_ult5": h_pts_5,
            "away_pts_ult5": a_pts_5,
            "diferencial_puntos_ult5": diff_pts_5
        },
        "advertencia_metodologica": "Las probabilidades calculadas son estimaciones estadísticas derivadas de resultados pre-partido históricos y no representan certezas de resultado ni implican causalidad determinista."
    }


def get_team_profile(team_name: str, season: Optional[int] = None) -> Dict[str, Any]:
    """
    Recupera el perfil taxonómico y analítico de un club (asignación de Clúster K-Means y PCA).
    """
    canonical = resolve_team_name(team_name)
    if not canonical:
        return {"status": "error", "message": f"No se reconoce el club '{team_name}'."}

    con = duckdb.connect(str(config.lakehouse_db_path), read_only=True)
    season_filter = f"AND season = {season}" if season else ""

    query = f"""
        SELECT team_name, season, puntos, goles_favor, goles_contra, diferencia_goles,
               victorias, empates, derrotas, cluster, cluster_name as etiqueta_cluster
        FROM teams_clustered
        WHERE team_name = '{canonical}' {season_filter}
        ORDER BY season DESC
    """
    rows = con.execute(query).fetchdf().to_dict(orient="records")
    con.close()

    if not rows:
        return {"status": "error", "message": f"No se encontraron registros de clúster para {canonical}."}

    return {
        "status": "success",
        "team_name": canonical,
        "perfiles_por_temporada": rows,
        "ultimo_perfil": rows[0]
    }


def get_standings(season: int = 2024) -> Dict[str, Any]:
    """
    Devuelve la tabla general de posiciones de la Premier League para una temporada específica.
    """
    con = duckdb.connect(str(config.lakehouse_db_path), read_only=True)
    query = f"""
        SELECT team_name, puntos, victorias, empates, derrotas, goles_favor, goles_contra,
               diferencia_goles, cluster, cluster_name as etiqueta_cluster
        FROM teams_clustered
        WHERE season = {season}
        ORDER BY puntos DESC, diferencia_goles DESC
    """
    rows = con.execute(query).fetchdf().to_dict(orient="records")
    con.close()

    if not rows:
        return {"status": "error", "message": f"No hay datos para la temporada {season}."}

    return {
        "status": "success",
        "season": season,
        "tabla": rows
    }


def generate_chart(chart_type: str, sql_query: str, title: str, output_filename: str) -> Dict[str, Any]:
    """
    Genera una visualización estadística (barras, dispersión, línea) y la almacena en disco.
    """
    res = query_duckdb(sql_query)
    if res["status"] != "success":
        return res

    df = pd.DataFrame(res["rows"])
    if df.empty:
        return {"status": "error", "message": "La consulta no devolvió datos para graficar."}

    # Asegurar extensión png
    filename = output_filename if output_filename.endswith(".png") else f"{output_filename}.png"
    out_path = config.reports_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cols = df.columns

    try:
        if chart_type == "bar":
            x_col, y_col = cols[0], cols[1]
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax, palette="Blues_d")
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "scatter":
            x_col, y_col = cols[0], cols[1]
            hue_col = cols[2] if len(cols) > 2 else None
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, s=80)
        elif chart_type == "line":
            x_col, y_col = cols[0], cols[1]
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax, marker="o")
            plt.xticks(rotation=45, ha="right")
        else:
            plt.close(fig)
            return {"status": "error", "message": f"Tipo de gráfico '{chart_type}' no soportado (use 'bar', 'scatter', 'line')."}

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        return {
            "status": "success",
            "chart_type": chart_type,
            "title": title,
            "output_path": str(out_path),
            "rows_plotted": len(df)
        }
    except Exception as e:
        plt.close(fig)
        return {"status": "error", "message": f"Error generando gráfico: {str(e)}"}
