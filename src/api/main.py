"""
Aplicación Principal FastAPI para el Agente Inteligente de Análisis Deportivo (Premier League).
Incorpora ciclo de vida asíncrono (lifespan), CORS restrictivo, validación Pydantic v2,
manejadores centralizados de excepciones y montaje de archivos estáticos para el Dashboard.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import os
import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routes import system, chat, sports, predictions, analytics
from ..agent.agent import SportsAnalystAgent
from ..agent.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida controlado de la aplicación:
    Inicializa los recursos pesados (agente, vector store, modelos ML y DuckDB)
    exactamente una vez durante el arranque, manteniéndolos en memoria compartida.
    """
    logger.info("Iniciando aplicación FastAPI. Precargando subsistemas y modelos...")
    
    # 1. Inicializar agente central (DuckDB, RAG y modelos supervisados)
    agent = SportsAnalystAgent()
    app.state.agent = agent
    logger.info(f"Agente inicializado. Modo Gemini: {config.is_gemini_configured} ({config.gemini_model_name}).")
    logger.info(f"RAG inicializado con {len(agent.rag.vector_store.chunks)} fragmentos.")

    yield

    # Cleanup al apagar el servidor
    logger.info("Apagando aplicación FastAPI. Liberando recursos...")
    app.state.agent = None


# Crear instancia de la aplicación FastAPI
app = FastAPI(
    title="Premier League AI Sports Analyst API",
    description="""
    API REST para el Agente Inteligente de Análisis Deportivo de la Premier League (2022-2025).
    Desarrollado para el Instituto Tecnológico de La Laguna (TecNM).
    
    Notas de Seguridad y Despliegue:
    * Entorno Local: Enlazar a 127.0.0.1 (loopback) para evitar exposición en red local sin autenticación.
    * Entorno Productivo: Enlazar a 0.0.0.0 únicamente dentro de contenedores aislados (Docker) o detrás de un
      Reverse Proxy / API Gateway (Nginx, Traefik, AWS ALB) que provea TLS/HTTPS, autenticación y WAF.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configuración de CORS restrictivo
allowed_origins_env = os.getenv("CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Manejadores Centralizados de Excepciones
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP {exc.status_code} en {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "code": exc.status_code, "detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [{"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")} for err in exc.errors()]
    logger.warning(f"Error de validación en {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "validation_error", "detail": errors}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Excepción no controlada en {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "internal_error", "detail": "Ocurrió un error interno en el servidor analítico."}
    )


# ---------------------------------------------------------------------------
# Inclusión de Routers
# ---------------------------------------------------------------------------

app.include_router(system.router)
app.include_router(chat.router)
app.include_router(sports.router)
app.include_router(predictions.router)
app.include_router(analytics.router)


# ---------------------------------------------------------------------------
# Montaje del Frontend y Archivos Estáticos
# ---------------------------------------------------------------------------

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
static_dir = frontend_dir / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Sirve la interfaz principal del Dashboard Interactivo."""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(content={"message": "Premier League AI Sports Analyst API operational. Dashboard index.html not yet deployed."})
