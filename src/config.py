"""
Módulo de Configuración Global y Entorno.
Gestiona rutas, claves de API, límites de tasa y parámetros del proyecto.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carga de variables de entorno desde .env si existe
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# BALLDONTLIE EPL API V2
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "").strip()
BALLDONTLIE_BASE_URL = os.getenv("BALLDONTLIE_BASE_URL", "https://api.balldontlie.io/epl/v2").rstrip("/")

# Control de Tasa (Rate Limiting)
# 5 req/min -> ~12.5s entre llamadas con margen de seguridad
# 60 req/min -> ~1.1s entre llamadas
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "5"))
REQUEST_INTERVAL_SECONDS = 60.0 / RATE_LIMIT_RPM if RATE_LIMIT_RPM > 0 else 12.5

# Salvaguarda de endpoints de pago:
# True solo si la cuenta cuenta con plan ALL-STAR / GOAT o prueba activa confirmada
ALLOW_PAID_ENDPOINTS = os.getenv("ALLOW_PAID_ENDPOINTS", "False").lower() in ("true", "1", "yes")

# Gemini LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()

# Rutas de almacenamiento local (Data Lakehouse)
RAW_DATA_DIR = ROOT_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
LAKEHOUSE_DB_PATH = ROOT_DIR / os.getenv("LAKEHOUSE_DB_PATH", "data/lakehouse/epl.duckdb")

# Asegurar existencia de directorios base
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LAKEHOUSE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
