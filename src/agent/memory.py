"""
Módulo de Memoria Conversacional del Agente Deportivo.
Conserva el contexto de diálogo, turnos previos, consultas históricas y entidades relevantes.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# Catálogo canónico de clubes para reconocimiento de entidades
KNOWN_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Ipswich Town", "Leeds United", "Leicester City", "Liverpool",
    "Luton Town", "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sheffield United", "Southampton", "Tottenham Hotspur",
    "West Ham United", "Wolverhampton Wanderers"
]


class ConversationTurn:
    """Representa un turno individual en el diálogo."""

    def __init__(self, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class AgentMemory:
    """Memoria de trabajo y contexto conversacional con seguimiento de entidades."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[ConversationTurn] = []
        self.entities: Dict[str, Any] = {
            "mentioned_teams": set(),
            "mentioned_seasons": set(),
            "last_prediction": None,
            "last_sql_query": None
        }

    def _extract_entities(self, text: str) -> None:
        """Identifica clubes y temporadas mencionadas en el texto respetando especificidad."""
        text_lower = text.lower()
        
        # Mapeo de términos específicos a nombres canónicos (ordenados por longitud descendente)
        alias_map = [
            ("manchester city", "Manchester City"),
            ("man city", "Manchester City"),
            ("manchester united", "Manchester United"),
            ("man united", "Manchester United"),
            ("man utd", "Manchester United"),
            ("tottenham hotspur", "Tottenham Hotspur"),
            ("tottenham", "Tottenham Hotspur"),
            ("spurs", "Tottenham Hotspur"),
            ("wolverhampton wanderers", "Wolverhampton Wanderers"),
            ("wolverhampton", "Wolverhampton Wanderers"),
            ("wolves", "Wolverhampton Wanderers"),
            ("west ham united", "West Ham United"),
            ("west ham", "West Ham United"),
            ("nottingham forest", "Nottingham Forest"),
            ("nottingham", "Nottingham Forest"),
            ("newcastle united", "Newcastle United"),
            ("newcastle", "Newcastle United"),
            ("sheffield united", "Sheffield United"),
            ("sheffield", "Sheffield United"),
            ("crystal palace", "Crystal Palace"),
            ("palace", "Crystal Palace"),
            ("brighton & hove albion", "Brighton"),
            ("brighton", "Brighton"),
            ("aston villa", "Aston Villa"),
            ("villa", "Aston Villa"),
            ("afc bournemouth", "Bournemouth"),
            ("bournemouth", "Bournemouth"),
            ("leicester city", "Leicester City"),
            ("leicester", "Leicester City"),
            ("leeds united", "Leeds United"),
            ("leeds", "Leeds United"),
            ("luton town", "Luton Town"),
            ("luton", "Luton Town"),
            ("ipswich town", "Ipswich Town"),
            ("ipswich", "Ipswich Town"),
            ("arsenal", "Arsenal"),
            ("chelsea", "Chelsea"),
            ("liverpool", "Liverpool"),
            ("everton", "Everton"),
            ("fulham", "Fulham"),
            ("brentford", "Brentford"),
            ("burnley", "Burnley"),
            ("southampton", "Southampton")
        ]

        found_teams = []
        for phrase, canonical in alias_map:
            pattern = rf"\b{re.escape(phrase)}\b"
            if re.search(pattern, text_lower):
                if canonical not in found_teams:
                    found_teams.append(canonical)

        # Actualizar lista manteniendo orden de aparición
        for t in found_teams:
            self.entities["mentioned_teams"].add(t)

        # Buscar temporadas (2022, 2023, 2024, 2025)
        seasons_found = re.findall(r"\b(202[2-5])\b", text)
        for s in seasons_found:
            self.entities["mentioned_seasons"].add(int(s))

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Registra una consulta del usuario y extrae entidades clave."""
        self._extract_entities(content)
        turn = ConversationTurn(role="user", content=content, metadata=metadata)
        self.history.append(turn)
        self._trim_history()

    def add_model_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Registra una respuesta generada por el agente o llamadas a herramientas."""
        turn = ConversationTurn(role="model", content=content, tool_calls=tool_calls, metadata=metadata)
        self.history.append(turn)
        self._trim_history()

    def add_tool_result(self, tool_name: str, result: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Registra el resultado devuelto por una herramienta."""
        meta = metadata or {}
        meta["tool_name"] = tool_name
        if tool_name == "predict_match" and isinstance(result, dict):
            self.entities["last_prediction"] = result
        elif tool_name == "query_duckdb":
            self.entities["last_sql_query"] = meta.get("sql_query")

        turn = ConversationTurn(role="tool", content=str(result), metadata=meta)
        self.history.append(turn)
        self._trim_history()

    def _trim_history(self) -> None:
        """Mantiene el tamaño de la memoria dentro del límite max_turns."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history_as_dicts(self) -> List[Dict[str, Any]]:
        """Devuelve el historial completo como lista de diccionarios serializables."""
        return [turn.to_dict() for turn in self.history]

    def get_context_summary(self) -> str:
        """Genera un resumen conciso de entidades y estado reciente para inyectar en el LLM."""
        lines = []
        if self.entities["mentioned_teams"]:
            teams_str = ", ".join(sorted(list(self.entities["mentioned_teams"])))
            lines.append(f"Equipos mencionados en la conversación: {teams_str}")
        if self.entities["mentioned_seasons"]:
            seasons_str = ", ".join(str(s) for s in sorted(list(self.entities["mentioned_seasons"])))
            lines.append(f"Temporadas bajo análisis: {seasons_str}")
        if self.entities["last_prediction"]:
            p = self.entities["last_prediction"]
            lines.append(f"Última predicción realizada: {p.get('home_team')} vs {p.get('away_team')} -> "
                         f"Resultado estimado: {p.get('prediccion_estimada')} "
                         f"(Probabilidades: L={p.get('probabilidades', {}).get('Local', 0):.2f}, "
                         f"E={p.get('probabilidades', {}).get('Empate', 0):.2f}, "
                         f"V={p.get('probabilidades', {}).get('Visita', 0):.2f})")
        return "\n".join(lines) if lines else "Sin contexto previo relevante."

    def clear(self) -> None:
        """Reinicia la memoria conversacional."""
        self.history.clear()
        self.entities = {
            "mentioned_teams": set(),
            "mentioned_seasons": set(),
            "last_prediction": None,
            "last_sql_query": None
        }
