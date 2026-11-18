"""
Módulo de Configuración del Agente de Análisis Deportivo.
Gestiona variables de entorno, rutas al lakehouse, modelos y parámetros de Gemini.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Cargar .env de U5_Proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    """Configuración centralizada y desacoplada del agente."""

    # Gemini API y Modelos
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    gemini_fallback_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # Rutas del Proyecto
    project_root: Path = PROJECT_ROOT
    lakehouse_db_path: Path = PROJECT_ROOT / "data" / "lakehouse" / "epl.duckdb"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports" / "figures"
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"

    # Políticas de Seguridad Operativa y Red
    enable_code_execution: bool = os.getenv("ENABLE_CODE_EXECUTION", "false").lower() == "true"
    max_memory_turns: int = 10
    code_execution_timeout: int = 10  # segundos

    # Configuración de Red del Servidor
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")

    @property
    def is_gemini_configured(self) -> bool:
        """Verifica si la API Key de Gemini está presente y no es un marcador de posición."""
        if not self.gemini_api_key:
            return False
        placeholder_tokens = ["tu_", "your_", "aqui", "dummy", "placeholder", "xxx"]
        key_lower = self.gemini_api_key.lower()
        if any(token in key_lower for token in placeholder_tokens):
            return False
        return len(self.gemini_api_key.strip()) >= 20


# Instancia por defecto
config = AgentConfig()
