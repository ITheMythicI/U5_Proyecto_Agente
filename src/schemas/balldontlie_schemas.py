"""
Esquemas Pydantic V2 para validación de datos de BALLDONTLIE EPL API V2.
Cumple estrictamente con la distinción entre NULL (dato no registrado/ausente) y 0 (ocurrencia nula).
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class SoccerTeam(BaseModel):
    """Esquema de validación para un equipo de la Premier League."""
    id: int = Field(..., description="Identificador único del club en BALLDONTLIE")
    name: str = Field(..., description="Nombre completo del club")
    short_name: Optional[str] = Field(None, description="Nombre corto o comercial")
    abbreviation: Optional[str] = Field(None, description="Abreviatura oficial (ej. ARS, MCI)")
    location: Optional[str] = Field(None, description="Ciudad o sede del club")

    class Config:
        extra = "ignore"


class SoccerStanding(BaseModel):
    """
    Esquema de clasificación oficial de Premier League.
    Valida temporadas, estadísticas agregadas y consistencia aritmética.
    """
    season: int = Field(..., description="Año de inicio de temporada (ej. 2023 para 2023-24)")
    rank: int = Field(..., ge=1, le=20, description="Posición en la tabla (1 a 20)")
    team: SoccerTeam = Field(..., description="Objeto de información del club")
    games_played: int = Field(..., ge=0, le=38, description="Partidos jugados")
    wins: int = Field(..., ge=0, le=38, description="Victorias")
    draws: int = Field(..., ge=0, le=38, description="Empates")
    losses: int = Field(..., ge=0, le=38, description="Derrotas")
    points: int = Field(..., ge=0, description="Puntos acumulados")
    goals_for: int = Field(..., ge=0, description="Goles a favor")
    goals_against: int = Field(..., ge=0, description="Goles en contra")
    goal_differential: int = Field(..., description="Diferencia de goles (puede ser negativa)")
    points_per_game: Optional[float] = Field(None, description="Promedio de puntos por partido")
    note: Optional[str] = Field(None, description="Notas (ej. clasificación europea o descenso)")

    @model_validator(mode="after")
    def verificar_consistencia_partidos(self) -> "SoccerStanding":
        """Comprueba que la suma de victorias, empates y derrotas equivalga a los partidos jugados."""
        total = self.wins + self.draws + self.losses
        if total != self.games_played:
            raise ValueError(
                f"Inconsistencia en partidos jugados ({self.games_played}) vs W+D+L ({total}) "
                f"para equipo {self.team.name} en temporada {self.season}"
            )
        # Comprobar diferencia de goles
        calc_diff = self.goals_for - self.goals_against
        if calc_diff != self.goal_differential:
            raise ValueError(
                f"Inconsistencia en diferencia de goles: reportada={self.goal_differential}, calculada={calc_diff}"
            )
        return self

    class Config:
        extra = "ignore"


class SoccerPlayer(BaseModel):
    """Esquema de futbolista registrado en Premier League."""
    id: int = Field(..., description="Identificador único del jugador")
    first_name: Optional[str] = Field(None, description="Nombre de pila")
    last_name: Optional[str] = Field(None, description="Apellido")
    position: Optional[str] = Field(None, description="Posición en el campo (F, M, D, G)")
    nationality: Optional[str] = Field(None, description="País de origen")
    team_id: Optional[int] = Field(None, description="ID del equipo actual")

    class Config:
        extra = "ignore"


class SoccerMatch(BaseModel):
    """
    Esquema de partido oficial de la Premier League.
    DISTINCIÓN CRÍTICA:
    - home_score / away_score es None (NULL) cuando el partido no se ha jugado o el marcador no está registrado.
    - home_score / away_score es 0 cuando el equipo jugó y no marcó goles (portería a cero / clean sheet).
    """
    id: int = Field(..., description="Identificador único del partido")
    season: int = Field(..., description="Temporada")
    date: str = Field(..., description="Fecha y hora del partido en formato ISO 8601")
    status: Optional[str] = Field(None, description="Estado textual (ej. FT, Halftime, Scheduled)")
    status_state: str = Field(..., description="Estado normalizado de ciclo de vida (pre, in, post)")
    home_team_id: Optional[int] = Field(None, description="ID del equipo local")
    away_team_id: Optional[int] = Field(None, description="ID del equipo visitante")
    home_team: Optional[SoccerTeam] = Field(None, description="Detalle del equipo local si está expandido")
    away_team: Optional[SoccerTeam] = Field(None, description="Detalle del equipo visitante si está expandido")
    
    # Marcadores: estrictamente opcionales para preservar NULL vs 0
    home_score: Optional[int] = Field(
        None, 
        ge=0, 
        description="Goles del local. NULL si el partido no se ha jugado; 0 si se jugó y no marcó goles."
    )
    away_score: Optional[int] = Field(
        None, 
        ge=0, 
        description="Goles del visitante. NULL si el partido no se ha jugado; 0 si se jugó y no marcó goles."
    )
    
    venue_name: Optional[str] = Field(None, description="Nombre del estadio")
    attendance: Optional[int] = Field(None, ge=0, description="Asistencia oficial al estadio")

    @property
    def resultado_1x2(self) -> Optional[str]:
        """
        Calcula el resultado categórico (Local, Empate, Visita).
        Devuelve None si cualquiera de los marcadores es NULL o si el partido no ha concluido (status_state != 'post').
        """
        if self.home_score is None or self.away_score is None:
            return None
        if self.status_state != "post":
            return None
        if self.home_score > self.away_score:
            return "Local"
        elif self.home_score < self.away_score:
            return "Visita"
        elif self.home_score == self.away_score:
            return "Empate"
        return None

    class Config:
        extra = "ignore"


class IngestionReport(BaseModel):
    """Reporte de integridad y estadísticas de descarga."""
    endpoint: str
    total_registros_recibidos: int
    registros_validos: int
    registros_invalidos: int
    duplicados_descartados: int
    ids_unicos: List[int]
    detalles_error: List[str] = []
