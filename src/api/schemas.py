"""
Esquemas Pydantic v2 para validación estricta de solicitudes y respuestas de la API REST.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sistema y Estado
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    timestamp: str = Field(..., json_schema_extra={"example": "2026-11-18T00:30:00"})
    app_name: str = Field(..., json_schema_extra={"example": "Premier League AI Sports Analyst API"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    database: str = Field(..., json_schema_extra={"example": "connected"})


class StatusResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "operational"})
    gemini_configured: bool = Field(..., json_schema_extra={"example": True})
    gemini_model_name: str = Field(..., json_schema_extra={"example": "gemini-2.5-flash"})
    active_model: str = Field(..., json_schema_extra={"example": "gemini-3.5-flash-lite"})
    rag_status: Dict[str, Any]
    duckdb_tables: List[str]
    cached_models: List[str]


# ---------------------------------------------------------------------------
# Agente Conversacional (Chat)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Consulta del usuario en lenguaje natural")
    session_id: Optional[str] = Field(None, description="Identificador único de sesión conversacional")
    forced_rag_mode: Optional[str] = Field(None, description="Modo RAG forzado: 'gemini_neural_3072d' o 'local_lsa_32d'")


class ChatResponse(BaseModel):
    session_id: str
    status: str
    mode: str
    model_name: Optional[str] = None
    response: str
    tool_calls: List[Dict[str, Any]] = []
    memory_entities: Dict[str, Any] = {}
    api_notice: Optional[str] = None


class ChatResetRequest(BaseModel):
    session_id: str = Field(..., description="Identificador de sesión a reiniciar")


class ChatResetResponse(BaseModel):
    status: str
    message: str
    session_id: str


# ---------------------------------------------------------------------------
# Datos Deportivos
# ---------------------------------------------------------------------------

class TeamStandingItem(BaseModel):
    posicion: int
    team_name: str
    season: int
    partidos_jugados: int
    puntos: int
    victorias: int
    empates: int
    derrotas: int
    goles_favor: int
    goles_contra: int
    diferencia_goles: int
    cluster: int
    etiqueta_cluster: str


class StandingsResponse(BaseModel):
    status: str
    season: int
    total_teams: int
    tabla: List[TeamStandingItem]


class TeamListItem(BaseModel):
    team_name: str
    canonical_name: str
    available_seasons: List[int]
    latest_cluster: Optional[str] = None


class TeamsListResponse(BaseModel):
    status: str
    total_teams: int
    teams: List[TeamListItem]


class TeamProfileResponse(BaseModel):
    status: str
    team_name: str
    ultimo_perfil: Dict[str, Any]
    perfiles_historicos: List[Dict[str, Any]]


class TeamCompareResponse(BaseModel):
    status: str
    season: int
    team_a: Dict[str, Any]
    team_b: Dict[str, Any]


# ---------------------------------------------------------------------------
# Predicciones
# ---------------------------------------------------------------------------

class MatchPredictionRequest(BaseModel):
    home_team: str = Field(..., description="Nombre del equipo local")
    away_team: str = Field(..., description="Nombre del equipo visitante")
    model_type: Optional[str] = Field("champion", description="Tipo de modelo: 'champion' o 'automl'")


class MatchPredictionResponse(BaseModel):
    status: str
    home_team: str
    away_team: str
    modelo_utilizado: str
    prediccion_estimada: str
    probabilidades: Dict[str, float]
    confianza_maxima: float
    factores_clave: List[str]
    advertencia_metodologica: str


# ---------------------------------------------------------------------------
# Analítica y Visualizaciones
# ---------------------------------------------------------------------------

class AnalyticsOverviewResponse(BaseModel):
    status: str
    total_matches: int
    seasons: List[int]
    total_goals: int
    avg_goals_per_match: float
    home_wins_pct: float
    draws_pct: float
    away_wins_pct: float
    teams_count: int


class SeasonSummaryItem(BaseModel):
    season: int
    matches_count: int
    total_goals: int
    avg_goals: float
    home_wins: int
    draws: int
    away_wins: int
    champion_team: str
    champion_points: int


class SeasonsSummaryResponse(BaseModel):
    status: str
    seasons: List[SeasonSummaryItem]


class ChartRequest(BaseModel):
    model_config = {"extra": "forbid"}
    chart_type: str = Field(..., description="Tipo de gráfico: 'bar', 'scatter', 'line'")
    preset: Optional[str] = Field(None, description="Plantilla analítica autorizada: 'goals_by_season', 'top_scoring_teams', 'points_distribution', 'cluster_summary'")
    data: Optional[Dict[str, Any]] = Field(None, description="Datos directos estructurados para graficar (opcional)")
    title: str = Field(..., max_length=150, description="Título principal del gráfico")
    x_label: Optional[str] = Field("Eje X", max_length=50, description="Etiqueta del eje X")
    y_label: Optional[str] = Field("Eje Y", max_length=50, description="Etiqueta del eje Y")


class ChartResponse(BaseModel):
    status: str
    chart_type: str
    format: str
    image_base64: str
