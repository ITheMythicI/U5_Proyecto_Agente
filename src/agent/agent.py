"""
Agente de Análisis Deportivo con Inteligencia Artificial (Premier League).
Integra Cerebro (Gemini LLM vía google-genai), Memoria conversacional,
Herramientas analíticas (DuckDB, Modelos supervisados, K-Means), RAG y Ejecución de Código.
"""

from typing import Dict, Any, List, Optional
import json
import re

from .config import config, AgentConfig
from .memory import AgentMemory
from .tools import (
    query_duckdb,
    predict_match,
    get_team_profile,
    get_standings,
    generate_chart,
    resolve_team_name
)
from .rag import RAGPipeline
from .code_executor import ControlledCodeExecutor

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


SYSTEM_INSTRUCTION = """Eres Antigravity EPL Analyst, un agente de inteligencia artificial especializado en análisis deportivo, ciencia de datos y modelado predictivo para la Premier League inglesa (temporadas 2022 a 2025).

Directrices operativas y metodológicas obligatorias:
1. Honestidad Estadística: Las predicciones deportivas NUNCA son certezas. Debes presentar siempre probabilidades por resultado (Local, Empate, Visita), explicar la clase predicha y los factores causales o métricas que la respaldan. Nunca confundas importancia de variables con causalidad determinista.
2. Uso Riguroso de Herramientas:
   - Para pronósticos de partidos, invoca 'predict_match'.
   - Para consultar estadísticas históricas, resultados y métricas del Lakehouse, invoca 'query_duckdb'.
   - Para perfiles tácticos, clasificación en clústeres K-Means y análisis PCA de un club, invoca 'get_team_profile'.
   - Para tablas de posiciones oficiales, invoca 'get_standings'.
   - Para reglas de competición, contexto histórico o arquitectura del proyecto, invoca 'search_knowledge_base' (RAG).
   - Para cálculos numéricos específicos o análisis de datos personalizados, invoca 'execute_code'.
   - Para crear gráficos estadísticos, invoca 'generate_chart'.
3. Claridad y Estructura: Responde con formato Markdown limpio, destacando métricas clave, tablas comparativas y justificaciones analíticas sólidas.
"""


class SportsAnalystAgent:
    """Orquestador central del Agente de Análisis Deportivo."""

    def __init__(self, agent_config: Optional[AgentConfig] = None):
        self.config = agent_config or config
        self.memory = AgentMemory(max_turns=self.config.max_memory_turns)
        self.rag = RAGPipeline(knowledge_dir=self.config.knowledge_dir)
        self.code_executor = ControlledCodeExecutor(timeout_seconds=self.config.code_execution_timeout)
        self.rag.initialize()

        self._client: Optional[genai.Client] = None
        if self.config.is_gemini_configured and GENAI_AVAILABLE:
            try:
                self._client = genai.Client(api_key=self.config.gemini_api_key)
            except Exception:
                self._client = None

    def search_knowledge_base(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Herramienta RAG: Recupera fragmentos contextuales mediante búsqueda semántica densa."""
        results = self.rag.retrieve(query, top_k=top_k)
        return {
            "status": "success",
            "query": query,
            "fragmentos_recuperados": results
        }

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Herramienta de Agente Programador: Ejecuta código Python seguro y auditado."""
        return self.code_executor.execute_python(code)

    def _get_tools_list(self) -> List[Any]:
        """Retorna la lista de funciones llamables para function calling en Gemini."""
        return [
            query_duckdb,
            predict_match,
            get_team_profile,
            get_standings,
            self.search_knowledge_base,
            self.execute_code,
            generate_chart
        ]

    def _execute_tool_by_name(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Despacha la ejecución de una herramienta y gestiona posibles errores."""
        tool_dispatch = {
            "query_duckdb": query_duckdb,
            "predict_match": predict_match,
            "get_team_profile": get_team_profile,
            "get_standings": get_standings,
            "search_knowledge_base": self.search_knowledge_base,
            "execute_code": self.execute_code,
            "generate_chart": generate_chart
        }

        if tool_name not in tool_dispatch:
            return {"status": "error", "message": f"Herramienta '{tool_name}' no existe."}

        func = tool_dispatch[tool_name]
        try:
            return func(**args)
        except Exception as e:
            return {"status": "error", "message": f"Error ejecutando {tool_name}: {str(e)}"}

    def ask(self, user_query: str, session_memory: Optional[AgentMemory] = None) -> Dict[str, Any]:
        """
        Punto de entrada principal para interactuar con el agente.
        Procesa la consulta, administra memoria, ejecuta herramientas y genera la respuesta.
        Permite recibir una instancia opcional de AgentMemory para gestión de sesiones concurrentes.
        """
        mem = session_memory or self.memory

        # 1. Registrar consulta en la memoria conversacional activa
        mem.add_user_message(user_query)

        # 2. Si Gemini está configurado con una clave válida, usar el LLM con function calling
        if self._client is not None:
            return self._ask_gemini_live(user_query, mem=mem)

        # 3. Modo de Despacho Local / Diagnóstico (si no hay API key de Gemini configurada aún)
        return self._ask_local_dispatcher(user_query, mem=mem)

    def _ask_gemini_live(self, user_query: str, mem: Optional[AgentMemory] = None) -> Dict[str, Any]:
        """
        Flujo de inferencia real utilizando Gemini con Chat y Automatic Function Calling (AFC).
        Maneja llamadas a herramientas reales, historial conversacional y contingencias de modelo.
        """
        active_memory = mem or self.memory
        context_summary = active_memory.get_context_summary()
        augmented_system = f"{SYSTEM_INSTRUCTION}\n\n[Contexto Conversacional Previo]:\n{context_summary}"

        tools = self._get_tools_list()
        gen_config = types.GenerateContentConfig(
            system_instruction=augmented_system,
            tools=tools,
            temperature=0.2
        )

        models_to_try = []
        for m in [self.config.gemini_model_name, "gemini-3.5-flash-lite", "gemini-3.8-flash"]:
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        executed_tools = []
        final_text = ""
        active_model = self.config.gemini_model_name

        for model_name in models_to_try:
            # Reintentos controlados (máximo 2 intentos por modelo ante 429)
            for attempt in range(2):
                try:
                    active_model = model_name
                    chat = self._client.chats.create(model=model_name, config=gen_config)
                    response = chat.send_message(user_query)
                    final_text = response.text

                    # Extraer herramientas ejecutadas a partir del historial del chat
                    for msg in chat.get_history():
                        if msg.parts:
                            for part in msg.parts:
                                if hasattr(part, "function_call") and part.function_call:
                                    fc_name = part.function_call.name
                                    fc_args = dict(part.function_call.args) if part.function_call.args else {}
                                    executed_tools.append({
                                        "tool": fc_name,
                                        "args": fc_args
                                    })
                                elif hasattr(part, "function_response") and part.function_response:
                                    fr_name = part.function_response.name
                                    fr_resp = part.function_response.response
                                    active_memory.add_tool_result(fr_name, fr_resp)
                                    if executed_tools and executed_tools[-1]["tool"] == fr_name:
                                        executed_tools[-1]["output"] = fr_resp

                    last_error = None
                    break  # Éxito en este intento

                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str or "not found" in err_str.lower():
                        last_error = f"Modelo '{model_name}' descontinuado o no disponible (404)."
                        break  # Pasar al siguiente modelo
                    elif "400" in err_str or "API_KEY_INVALID" in err_str:
                        return {
                            "status": "error",
                            "error_type": "invalid_api_key",
                            "message": "La API Key de Gemini configurada en .env no es válida. Verifique la credencial en Google AI Studio.",
                            "mode": "gemini_error"
                        }
                    elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        last_error = f"Cuota excedida (429) en {model_name} (Intento {attempt+1}/2)."
                        if attempt == 0:
                            import time
                            time.sleep(5)  # Breve espera controlada antes del segundo intento
                            continue
                        else:
                            break  # Pasar al siguiente modelo disponible
                    else:
                        last_error = err_str
                        break

            if final_text:
                break  # Éxito obtenido con el modelo activo

        if last_error and not final_text:
            local_res = self._ask_local_dispatcher(user_query, mem=active_memory)
            local_res["api_notice"] = f"Aviso de Gemini API ({active_model}): {last_error}. Se recurrió al despachador analítico local."
            return local_res

        active_memory.add_model_message(final_text, tool_calls=executed_tools)
        return {
            "status": "success",
            "mode": "gemini_live",
            "model_name": active_model,
            "response": final_text,
            "tool_calls": executed_tools,
            "memory_entities": {
                "teams": list(active_memory.entities["mentioned_teams"]),
                "seasons": list(active_memory.entities["mentioned_seasons"])
            }
        }

    def _ask_local_dispatcher(self, user_query: str, mem: Optional[AgentMemory] = None) -> Dict[str, Any]:
        """
        Despachador analítico local determinista.
        Permite validar la memoria, predicciones, DuckDB, RAG y ejecución de código
        incluso cuando la clave de Gemini no ha sido configurada en el .env.
        """
        active_memory = mem or self.memory
        q_lower = user_query.lower()
        executed_tools = []
        response_text = ""

        # Identificar equipos en la consulta actual (prioridad) o en la memoria previa
        query_teams = []
        for team in [
            "Manchester City", "Manchester United", "Tottenham Hotspur", "Wolverhampton Wanderers",
            "West Ham United", "Nottingham Forest", "Newcastle United", "Sheffield United",
            "Crystal Palace", "Brighton", "Aston Villa", "Bournemouth", "Leicester City",
            "Leeds United", "Luton Town", "Ipswich Town", "Arsenal", "Chelsea", "Liverpool",
            "Everton", "Fulham", "Brentford", "Burnley", "Southampton"
        ]:
            if team.lower() in q_lower or (team.split()[0].lower() in q_lower and len(team.split()[0]) >= 4 and "manchester" not in team.lower()):
                if team not in query_teams:
                    query_teams.append(team)
            elif "man city" in q_lower and team == "Manchester City":
                query_teams.append(team)
            elif ("man utd" in q_lower or "man united" in q_lower) and team == "Manchester United":
                query_teams.append(team)

        teams_found = query_teams if query_teams else list(active_memory.entities["mentioned_teams"])

        if any(w in q_lower for w in ["predic", "pronostic", "gana", "enfrent", "vs", "versus"]) and len(teams_found) >= 2:
            h_team, a_team = teams_found[0], teams_found[1]
            res = predict_match(h_team, a_team)
            active_memory.add_tool_result("predict_match", res, metadata={"home": h_team, "away": a_team})
            executed_tools.append({"tool": "predict_match", "args": {"home_team": h_team, "away_team": a_team}, "output": res})
            
            if res.get("status") == "success":
                probs = res["probabilidades"]
                factors = "\n".join(f"- {f}" for f in res["factores_clave"])
                response_text = (
                    f"### Pronóstico Estadístico: {res['home_team']} (Local) vs {res['away_team']} (Visita)\n\n"
                    f"- **Modelo Operativo:** {res['modelo_utilizado']}\n"
                    f"- **Resultado Más Probable Estimado:** **{res['prediccion_estimada']}** ({res['confianza_maxima']*100:.1f}%)\n"
                    f"- **Distribución de Probabilidades 1X2:**\n"
                    f"  - Victoria Local ({res['home_team']}): **{probs['Local']*100:.2f}%**\n"
                    f"  - Empate: **{probs['Empate']*100:.2f}%**\n"
                    f"  - Victoria Visitante ({res['away_team']}): **{probs['Visita']*100:.2f}%**\n\n"
                    f"**Factores Clave Determinantes:**\n{factors}\n\n"
                    f"> *Nota Metodológica:* {res['advertencia_metodologica']}"
                )
            else:
                response_text = f"Error generando predicción: {res.get('message')}"

        # Detección de intención: Perfil taxonómico / Clúster de un Club
        elif any(w in q_lower for w in ["cluster", "clúster", "perfil", "estilo", "categor"]) and len(teams_found) >= 1:
            team = teams_found[0]
            res = get_team_profile(team)
            active_memory.add_tool_result("get_team_profile", res, metadata={"team": team})
            executed_tools.append({"tool": "get_team_profile", "args": {"team_name": team}, "output": res})
            if res.get("status") == "success":
                p = res["ultimo_perfil"]
                dif = int(p['diferencia_goles'])
                response_text = (
                    f"### Perfil Taxonómico de {res['team_name']} (Temporada {p['season']})\n\n"
                    f"- **Clúster Asignado (K-Means k=3):** Clúster {p['cluster']} — **{p['etiqueta_cluster']}**\n"
                    f"- **Rendimiento de Temporada:** {int(p['puntos'])} puntos en 38 partidos ({int(p['victorias'])}V, {int(p['empates'])}E, {int(p['derrotas'])}D)\n"
                    f"- **Balance de Goles:** {int(p['goles_favor'])} GF, {int(p['goles_contra'])} GC (Diferencia: {dif:+d})\n"
                )
            else:
                response_text = f"Error consultando perfil: {res.get('message')}"

        # Detección de intención: Tabla de posiciones
        elif (any(w in q_lower for w in ["tabla de posiciones", "tabla general", "standings"]) or
              (("tabla" in q_lower or "posiciones" in q_lower) and not any(r in q_lower for r in ["regla", "champions", "descenso"]))):
            season = 2024
            if active_memory.entities["mentioned_seasons"]:
                season = max(active_memory.entities["mentioned_seasons"])
            res = get_standings(season=season)
            active_memory.add_tool_result("get_standings", res, metadata={"season": season})
            executed_tools.append({"tool": "get_standings", "args": {"season": season}, "output": res})
            if res.get("status") == "success":
                top5 = res["tabla"][:5]
                top_str = "\n".join(f"{i+1}. **{t['team_name']}** — {int(t['puntos'])} pts (Dif: {int(t['diferencia_goles']):+d})" for i, t in enumerate(top5))
                response_text = f"### Tabla General de Posiciones — Premier League {season}\n\n**Top 5 Equipos:**\n{top_str}\n\n*(Consulta DuckDB para ver los 20 clubes completos)*"
            else:
                response_text = f"Error consultando tabla: {res.get('message')}"

        # Detección de intención: Consulta a Base de Conocimiento (RAG)
        else:
            rag_res = self.search_knowledge_base(user_query, top_k=2)
            active_memory.add_tool_result("search_knowledge_base", rag_res, metadata={"query": user_query})
            executed_tools.append({"tool": "search_knowledge_base", "args": {"query": user_query}, "output": rag_res})
            frags = rag_res["fragmentos_recuperados"]
            if frags:
                rag_texts = "\n\n".join(f"**De {f['source_file']} ({f['section_title']}):**\n{f['text']}" for f in frags)
                response_text = f"### Información Recuperada mediante RAG Semántico\n\n{rag_texts}"
            else:
                response_text = "No se localizó información concluyente para la consulta formulada."

        active_memory.add_model_message(response_text, tool_calls=executed_tools)

        return {
            "status": "success",
            "mode": "local_dispatcher",
            "response": response_text,
            "tool_calls": executed_tools,
            "memory_entities": {
                "teams": list(active_memory.entities["mentioned_teams"]),
                "seasons": list(active_memory.entities["mentioned_seasons"])
            }
        }
