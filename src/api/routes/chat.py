"""
Rutas del Agente Conversacional con memoria de sesión y Function Calling real.
"""

from fastapi import APIRouter, Request, HTTPException
from ..schemas import ChatRequest, ChatResponse, ChatResetRequest, ChatResetResponse
from ..session_manager import session_manager

router = APIRouter(prefix="/api/v1/chat", tags=["Agente Conversacional"])


@router.post("", response_model=ChatResponse, summary="Interactuar con el agente conversacional")
def post_chat_message(request: Request, payload: ChatRequest):
    """
    Envía una consulta en lenguaje natural al agente de análisis deportivo.
    Conserva el contexto en la sesión especificada, ejecuta herramientas en DuckDB/ML si es necesario,
    y sintetiza la respuesta mediante Gemini o el despachador analítico local.
    """
    agent = getattr(request.app.state, "agent", None)
    if not agent:
        from ...agent.agent import SportsAnalystAgent
        agent = SportsAnalystAgent()
        request.app.state.agent = agent

    # 1. Resolver sesión y memoria independiente
    session_id, mem = session_manager.get_or_create(payload.session_id)

    # 2. Configurar modo RAG si se especificó explícitamente
    if payload.forced_rag_mode in ["local_lsa_32d", "gemini_neural_3072d"]:
        agent.rag.initialize(forced_mode=payload.forced_rag_mode)

    # 3. Invocar al agente
    try:
        result = agent.ask(user_query=payload.message, session_memory=mem)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el procesamiento del agente: {str(e)}")

    if result.get("status") == "error":
        return ChatResponse(
            session_id=session_id,
            status="error",
            mode=result.get("mode", "error"),
            model_name=result.get("model_name"),
            response=result.get("message", "Error inesperado en la consulta."),
            tool_calls=[],
            memory_entities={},
            api_notice=result.get("api_notice")
        )

    return ChatResponse(
        session_id=session_id,
        status="success",
        mode=result.get("mode", "unknown"),
        model_name=result.get("model_name"),
        response=result.get("response", ""),
        tool_calls=result.get("tool_calls", []),
        memory_entities=result.get("memory_entities", {}),
        api_notice=result.get("api_notice")
    )


@router.post("/reset", response_model=ChatResetResponse, summary="Reiniciar memoria de una sesión")
def reset_chat_session(payload: ChatResetRequest):
    """Borra el historial de diálogo y entidades acumuladas para el session_id especificado."""
    success = session_manager.reset_session(payload.session_id)
    return ChatResetResponse(
        status="success" if success else "not_found",
        message="Memoria de sesión reiniciada exitosamente." if success else "La sesión no existía o ya estaba vacía.",
        session_id=payload.session_id
    )
