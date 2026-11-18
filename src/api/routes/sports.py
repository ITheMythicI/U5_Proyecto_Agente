"""
Rutas de datos deportivos: tablas de clasificación, perfiles de equipos y comparaciones directas.
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException, Path
import duckdb

from ..schemas import (
    StandingsResponse, TeamStandingItem,
    TeamsListResponse, TeamListItem,
    TeamProfileResponse, TeamCompareResponse
)
from ...agent.tools import get_standings, get_team_profile, resolve_team_name, CANONICAL_TEAMS
from ...agent.config import config

router = APIRouter(prefix="/api/v1", tags=["Datos Deportivos"])


@router.get("/standings", response_model=StandingsResponse, summary="Tabla de posiciones por temporada")
def get_league_standings(season: int = Query(2024, ge=2022, le=2024, description="Año inicial de la temporada (2022, 2023, 2024)")):
    """Retorna la tabla general de posiciones de la Premier League con clasificación taxonómica por clústeres."""
    res = get_standings(season=season)
    if res.get("status") != "success":
        raise HTTPException(status_code=404, detail=res.get("message", "No se encontraron datos de clasificación."))

    rows = res["tabla"]
    items = []
    for idx, r in enumerate(rows, start=1):
        items.append(TeamStandingItem(
            posicion=idx,
            team_name=r["team_name"],
            season=season,
            partidos_jugados=int(r["victorias"] + r["empates"] + r["derrotas"]),
            puntos=int(r["puntos"]),
            victorias=int(r["victorias"]),
            empates=int(r["empates"]),
            derrotas=int(r["derrotas"]),
            goles_favor=int(r["goles_favor"]),
            goles_contra=int(r["goles_contra"]),
            diferencia_goles=int(r["diferencia_goles"]),
            cluster=int(r["cluster"]),
            etiqueta_cluster=str(r["etiqueta_cluster"])
        ))

    return StandingsResponse(
        status="success",
        season=season,
        total_teams=len(items),
        tabla=items
    )


@router.get("/teams", response_model=TeamsListResponse, summary="Listado completo de clubes")
def list_teams():
    """Retorna los 24 clubes catalogados en el Lakehouse, sus temporadas disponibles y su clúster más reciente."""
    con = duckdb.connect(str(config.lakehouse_db_path), read_only=True)
    df = con.execute("""
        SELECT team_name, list(season ORDER BY season) as seasons,
               arg_max(cluster_name, season) as latest_cluster
        FROM teams_clustered
        GROUP BY team_name
        ORDER BY team_name ASC
    """).fetchdf()
    con.close()

    teams_list = []
    for _, row in df.iterrows():
        t_name = row["team_name"]
        canonical = resolve_team_name(t_name) or t_name
        seasons_avail = list(row["seasons"])
        teams_list.append(TeamListItem(
            team_name=t_name,
            canonical_name=canonical,
            available_seasons=seasons_avail,
            latest_cluster=row["latest_cluster"]
        ))

    return TeamsListResponse(
        status="success",
        total_teams=len(teams_list),
        teams=teams_list
    )


@router.get("/teams/{team_name}/profile", response_model=TeamProfileResponse, summary="Perfil táctico y taxonómico de un club")
def get_team_profile_endpoint(
    team_name: str = Path(..., description="Nombre del equipo (e.g. 'Arsenal', 'Manchester City')"),
    season: Optional[int] = Query(None, ge=2022, le=2024, description="Temporada específica (opcional)")
):
    """Retorna las métricas históricas, clúster K-Means y balance de goles de un club."""
    canonical = resolve_team_name(team_name)
    if not canonical:
        raise HTTPException(status_code=404, detail=f"Club '{team_name}' no reconocido.")

    res = get_team_profile(canonical, season=season)
    if res.get("status") != "success":
        raise HTTPException(status_code=404, detail=res.get("message"))

    return TeamProfileResponse(
        status="success",
        team_name=canonical,
        ultimo_perfil=res["ultimo_perfil"],
        perfiles_historicos=res.get("perfiles_por_temporada", [])
    )


@router.get("/teams/compare", response_model=TeamCompareResponse, summary="Comparación cara a cara entre dos clubes")
def compare_teams_endpoint(
    team_a: str = Query(..., description="Nombre del primer club"),
    team_b: str = Query(..., description="Nombre del segundo club"),
    season: int = Query(2024, ge=2022, le=2024, description="Temporada para comparar")
):
    """Compara métricas ofensivas, defensivas y asignación taxonómica de dos clubes en una misma temporada."""
    can_a = resolve_team_name(team_a)
    can_b = resolve_team_name(team_b)

    if not can_a:
        raise HTTPException(status_code=404, detail=f"Club '{team_a}' no reconocido.")
    if not can_b:
        raise HTTPException(status_code=404, detail=f"Club '{team_b}' no reconocido.")

    res_a = get_team_profile(can_a, season=season)
    res_b = get_team_profile(can_b, season=season)

    if res_a.get("status") != "success":
        raise HTTPException(status_code=404, detail=f"Sin datos en {season} para {can_a}.")
    if res_b.get("status") != "success":
        raise HTTPException(status_code=404, detail=f"Sin datos en {season} para {can_b}.")

    return TeamCompareResponse(
        status="success",
        season=season,
        team_a=res_a["ultimo_perfil"],
        team_b=res_b["ultimo_perfil"]
    )
