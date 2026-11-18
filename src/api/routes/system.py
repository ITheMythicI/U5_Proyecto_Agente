"""
Rutas de diagnóstico, salud del sistema y estado operativo.
"""

from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
import duckdb

from ..schemas import HealthResponse, StatusResponse
from ...agent.config import config

router = APIRouter(tags=["Sistema"])


@router.get("/health", response_model=HealthResponse, summary="Verificación básica de salud")
def health_check(request: Request):
    """Retorna el estado de disponibilidad general de la API y conectividad con DuckDB."""
    db_status = "connected"
    try:
        conn = duckdb.connect(str(config.lakehouse_db_path), read_only=True)
        conn.execute("SELECT 1").fetchone()
        conn.close()
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="ok" if "error" not in db_status else "degraded",
        timestamp=datetime.now().isoformat(),
        app_name="Premier League AI Sports Analyst API",
        version="1.0.0",
        database=db_status
    )


@router.get("/api/v1/status", response_model=StatusResponse, summary="Diagnóstico detallado de subsistemas")
def get_system_status(request: Request):
    """Retorna el estado de Gemini, tablas cargadas en DuckDB, modelos en caché y RAG."""
    agent = getattr(request.app.state, "agent", None)

    # 1. Comprobar tablas en DuckDB
    tables = []
    try:
        conn = duckdb.connect(str(config.lakehouse_db_path), read_only=True)
        res = conn.execute("SHOW TABLES").fetchall()
        tables = [r[0] for r in res]
        conn.close()
    except Exception:
        tables = []

    # 2. Comprobar modelos disponibles
    models_dir = config.models_dir
    cached_models = [f.name for f in models_dir.glob("*.joblib")] if models_dir.exists() else []

    # 3. Estado de RAG
    rag_info = {
        "is_indexed": agent.rag._is_indexed if agent else False,
        "indexed_mode": agent.rag._indexed_mode if agent else "none",
        "total_chunks": len(agent.rag.vector_store.chunks) if agent and agent.rag.vector_store else 0
    }

    active_model = "gemini-3.5-flash-lite"
    if agent and agent._client is not None:
        active_model = agent.config.gemini_model_name

    return StatusResponse(
        status="operational",
        gemini_configured=config.is_gemini_configured,
        gemini_model_name=config.gemini_model_name,
        active_model=active_model,
        rag_status=rag_info,
        duckdb_tables=tables,
        cached_models=cached_models
    )
