# Premier League AI Sports Analyst & REST API (Fase 4)
## Proyecto de Ciencia de Datos y Machine Learning — Instituto Tecnológico de La Laguna (TecNM)

Sistema integral de análisis deportivo para la Premier League inglesa (Temporadas 2022-2025). Integra un **Data Lakehouse** local en DuckDB (1,140 partidos), modelos supervisados y no supervisados de **Machine Learning**, un sistema **RAG Semántico Dual**, un **Agente Inteligente con Function Calling real (Gemini)** y una **API REST de alto rendimiento con FastAPI y Dashboard Interactivo**.

---

## 🚀 Arquitectura y Tecnologías Principales

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn + Pydantic v2 (Lifespan asíncrono, CORS restrictivo, gestión de sesiones).
- **Frontend:** Dashboard SPA moderno (HTML5 + Tailwind CSS + Chart.js + Marked.js), 100% desacoplado y adaptable a dispositivos móviles.
- **Data Lakehouse:** DuckDB (acceso `read_only`), Parquet particionado por temporada.
- **Machine Learning:** Scikit-Learn (`LogisticRegression` balanceada como Champion Model), FLAML (AutoML), K-Means ($k=3$) y PCA.
- **Inteligencia Artificial:** Google Gemini (`gemini-3.5-flash-lite` / `gemini-3.8-flash`) vía librería oficial `google-genai` (v3.5.0) con Automatic Function Calling (AFC).
- **RAG Dual:** Gemini Neural Embeddings (3,072d) y LSA Local (18d).

---

## 📦 Instalación y Configuración

### 1. Requisitos Previos
- Python 3.10 o superior (recomendado Python 3.12/3.14 o entorno Conda).
- Clave de API de Google AI Studio configurada para el agente conversacional.

### 2. Clonación y Entorno Virtual
```bash
cd U5_Proyecto
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows
```

### 3. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 4. Variables de Entorno (.env)
Copia el archivo de plantilla `.env.example` a `.env` y configura tus credenciales:
```bash
cp .env.example .env
```
Edita `.env` para agregar tu `GEMINI_API_KEY`:
```ini
GEMINI_API_KEY=AIzaSy...tu_clave_real...
GEMINI_MODEL_NAME=gemini-2.5-flash
```

---

## 🖥️ Ejecución de la Aplicación

### Iniciar el Servidor API y Dashboard
Ejecuta el siguiente comando desde la raíz de `U5_Proyecto`:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

> **Nota de Seguridad de Red:**  
> Por defecto en desarrollo se enlaza a `127.0.0.1` (loopback) para evitar exposición accidental a otros dispositivos en la red local. El enlace a `0.0.0.0` debe reservarse exclusivamente para contenedores Docker o entornos detrás de un Reverse Proxy con autenticación y TLS.

Una vez iniciado el servidor:
- **Dashboard Web Interactivo:** Abre tu navegador en [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Documentación Interactiva Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Documentación ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 Catálogo Principal de Endpoints REST

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Verificación de disponibilidad y conexión a DuckDB |
| `GET` | `/api/v1/status` | Diagnóstico de modelos cargados, tablas DuckDB y RAG |
| `POST` | `/api/v1/chat` | Consulta al Agente IA con memoria conversacional y AFC |
| `POST` | `/api/v1/chat/reset` | Reinicio de memoria para una sesión específica |
| `GET` | `/api/v1/standings` | Tabla general con clústeres K-Means (`?season=2024`) |
| `GET` | `/api/v1/teams` | Catálogo de los 24 clubes catalogados |
| `GET` | `/api/v1/teams/{team_name}/profile` | Perfil táctico y métricas históricas de un club |
| `GET` | `/api/v1/teams/compare` | Comparador directo cara a cara entre dos clubes |
| `POST` | `/api/v1/predictions/match` | Pronóstico probabilístico 1X2 con modelos de ML |
| `GET` | `/api/v1/analytics/overview` | Indicadores globales de los 1,140 partidos |
| `GET` | `/api/v1/analytics/seasons` | Resumen comparativo de temporadas y campeones |
| `POST` | `/api/v1/analytics/chart` | Generador de gráficos analíticos en Base64 |

---

## 🧪 Ejecución de Pruebas Automatizadas

```bash
# 1. Pruebas de la API REST y Dashboard (Fase 4 - 24 tests)
pytest tests/test_api.py -v

# 2. Pruebas Unitarias del Agente y Herramientas (Fase 3 - 27 tests)
pytest tests/test_agent.py -v

# 3. Pruebas en Vivo contra Gemini API (Fase 3 - 5 tests)
pytest tests/test_live_gemini.py -v

# 4. Batería Completa (56 tests)
pytest tests/ -v
```

---

## 🔒 Políticas de Seguridad y Confiabilidad

1. **Aislamiento de Datos:** DuckDB opera permanentemente en modo `read_only = True` con bloqueo de comandos DDL/DML e inyección de límites (`LIMIT 50`).
2. **Sin Ejecución Arbitraria de Código:** `enable_code_execution = False` por defecto. Cualquier llamada dinámica es rechazada.
3. **Protección de Credenciales:** Ninguna API Key o variable de entorno sensible se expone en la API, logs o navegador web.
4. **Respeto de Cuotas:** Reintentos exponenciales con límite estricto de 2 intentos y backoff ante HTTP 429 para respetar las tasas de Google AI Studio.
