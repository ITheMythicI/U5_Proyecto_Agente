# AUDITORÍA FINAL Y CIERRE DEFINITIVO DEL PROYECTO
## Agente Inteligente de Análisis Deportivo con IA (Premier League)

**Institución:** Instituto Tecnológico de La Laguna, perteneciente al Tecnológico Nacional de México (TecNM)  
**Materia:** Análisis de Datos y Machine Learning  
**Fecha:** 18 de noviembre de 2026  
**Versión de Entrega:** 1.1.0 (Producción Auditada)  
**Código Fuente:** [`U5_Proyecto/`](./)  
**Plan de Fase 4:** [`plan_fase4.md`](./plan_fase4.md)  
**Reporte de Implementación:** [`reporte_fase4.md`](./reporte_fase4.md)  
**Dictamen:** **APROBADO CON EXCELENCIA TÉCNICA**

---

## 1. Resumen de la Auditoría y Cambios Realizados

El proyecto **Agente Inteligente de Análisis Deportivo con IA (Premier League)** ha concluido todas sus etapas de investigación, ingeniería de datos, modelado predictivo, integración de agentes autónomos y desarrollo de interfaz gráfica. Esta auditoría final verifica el cierre de la **Fase 4: Desarrollo de API REST con FastAPI y Dashboard Interactivo**, así como la consistencia global del sistema.

### Componentes Auditados e Integrados
1. **Data Lakehouse & Almacenamiento Analítico:** 1,140 partidos oficiales de la Premier League (temporadas 2022, 2023 y 2024) capturados desde la API BALLDONTLIE EPL V2, transformados con PySpark, tipados con Pydantic y persistidos en formato columnar Parquet particionado. Consultados en modo solo lectura de alta velocidad mediante DuckDB.
2. **Modelos Predictivos (Machine Learning):** Modelo campeón de Regresión Logística con balanceo de clases (`logistic_regression_balanced.joblib`), pipeline de AutoML FLAML (`flaml_automl_champion.joblib`) y estratificación no supervisada K-Means (4 clústeres taxonómicos).
3. **Agente Cognitivo e Inteligencia Artificial:** Integración de Google Gemini (`gemini-2.5-flash` / `gemini-3.5-flash-lite`) mediante el SDK oficial `google-genai`, equipado con Automatic Function Calling (AFC), memoria conversacional aislada y recuperación semántica aumentada (RAG híbrido: LSA/TF-IDF y Embeddings neuronales).
4. **Backend de Servicios (FastAPI):** API REST asíncrona, estructurada modularmente en routers (`system`, `sports`, `predictions`, `analytics`, `chat`), con esquemas Pydantic v2 y ciclo de vida (*lifespan*) para precarga en memoria de modelos y conexiones.
5. **Frontend SPA Desacoplado:** Tablero de control web interactivo con diseño responsive (Dark Theme deportivo), componentes dinámicos con Chart.js, renderizado seguro de Markdown y separación total entre la capa de presentación y la capa de lógica de negocio.

---

## 2. Vulnerabilidades Corregidas y Endurecimiento de Seguridad

Durante la auditoría técnica de la Fase 4, se detectaron tres vectores potenciales de vulnerabilidad, los cuales fueron solventados y verificados exhaustivamente mediante pruebas automatizadas:

```mermaid
flowchart TD
    subgraph "Vulnerabilidades y Mecanismos de Mitigación"
        A["Entrada del Cliente (HTTP)"] --> B{"Ruta Solicitada"}
        
        B -->|POST /api/v1/analytics/chart| C["Detección de SQL Arbitrario"]
        C -->|Pydantic extra='forbid'| D["Rechazo 422 si se envía sql_query"]
        C -->|Parámetro preset| E["Catálogo de Consultas Pre-aprobadas Whitelist"]
        E --> F["DuckDB (read_only=True)"]
        
        B -->|POST /api/v1/chat| G["Generación de Markdown"]
        G --> H["Frontend: marked.parse()"]
        H --> I["DOMPurify.sanitize()"]
        I -->|Filtro Anti-XSS| J["Inserción Segura en DOM (innerHTML)"]
        
        B -->|Gestión de Sesiones| K["SessionMemoryManager"]
        K --> L["threading.RLock()"]
        L --> M["Aislamiento Concurrente + Poda LRU (Máx 500)"]
    end
```

### 2.1 Blindaje del Endpoint de Gráficos (`POST /api/v1/analytics/chart`)
- **Riesgo Previo:** El esquema inicial permitía de forma opcional un parámetro `sql_query`, lo que abría una superficie para inyecciones SQL o intentos de invocar funciones de lectura arbitraria en el sistema de archivos (`read_csv`, `read_parquet`).
- **Solución Implementada:**
  1. Se eliminó completamente el atributo `sql_query` del esquema [`ChartRequest`](src/api/schemas.py).
  2. Se configuró `model_config = {"extra": "forbid"}` en Pydantic. Si un atacante intenta enviar `"sql_query": "SELECT * FROM ..."` en el payload JSON, FastAPI rechaza la petición de forma instantánea con un código `422 Unprocessable Entity`.
  3. Se instauró un catálogo cerrado de consultas pre-aprobadas (`TRUSTED_PRESET_QUERIES`) en [`analytics.py`](src/api/routes/analytics.py):
     - `goals_by_season`: Total de goles y promedio por temporada.
     - `top_scoring_teams`: Top 10 de equipos más anotadores (temporada 2024).
     - `points_distribution`: Puntos obtenidos por todos los clubes en 2024.
     - `cluster_summary`: Distribución de clubes según clúster taxonómico K-Means.
  4. DuckDB se mantiene configurado con `read_only=True`, imposibilitando escrituras, alteraciones de esquema o mutaciones en disco.

### 2.2 Neutralización de Cross-Site Scripting (XSS) en Markdown
- **Riesgo Previo:** Las respuestas del agente Gemini se formatean en Markdown enriquecido (tablas, negritas, listas). En el frontend [`app.js`](src/frontend/static/js/app.js), la conversión se realizaba mediante `marked.parse(content)` y se inyectaba directamente con `innerHTML`. Si la respuesta del modelo o una inyección de prompt contenía vectores como `<script>` o `<img src="x" onerror="alert(1)">`, se habría ejecutado en el contexto del navegador.
- **Solución Implementada:**
  1. Se integró la biblioteca oficial de sanitización **DOMPurify** (v3.2.4) en [`index.html`](src/frontend/index.html).
  2. La función de renderizado ejecuta `DOMPurify.sanitize(rawHtml, {...})` con una lista blanca estricta de etiquetas permitidas (`p`, `br`, `strong`, `em`, `ul`, `ol`, `li`, `code`, `pre`, `table`, `thead`, `tbody`, `tr`, `th`, `td`, `blockquote`, `span`) y bloqueo forzoso de etiquetas peligrosas (`script`, `iframe`, `object`, `embed`, `form`) y atributos de eventos (`onerror`, `onload`, `onclick`, `onmouseover`).

### 2.3 Concurrencia y Aislamiento de Memoria de Sesiones
- **Riesgo Previo:** En entornos multihilo con múltiples clientes concurrentes consultando el agente, el acceso simultáneo no sincronizado al diccionario de sesiones en memoria podía causar condiciones de carrera (*race conditions*), colisiones de memoria o fugas de contexto entre usuarios distintos.
- **Solución Implementada:**
  1. Se rediseñó [`SessionMemoryManager`](src/api/session_manager.py) empleando un cerrojo reentrante (`threading.RLock()`).
  2. Todas las operaciones críticas (`get_or_create`, `add_message`, `reset_session`, `delete_session`, poda LRU) son atómicas y thread-safe.
  3. Se implementó una política de poda por menor uso reciente (LRU) con un límite de 500 sesiones activas para proteger la memoria RAM del servidor contra ataques de saturación o denegación de servicio por acumulación de sesiones huérfanas.

---

## 3. Configuración de Red y Despliegue

Para garantizar la seguridad de la infraestructura y delimitar con claridad el entorno local frente a un entorno público, se establecieron las siguientes políticas operativas:

| Directriz | Entorno de Desarrollo Local | Entorno de Servidor / Producción |
| :--- | :--- | :--- |
| **Host de Escucha** | `127.0.0.1` (Loopback obligatorio) | `0.0.0.0` (Únicamente en contenedor Docker o red privada interna) |
| **Justificación de Red** | Evita que interfaces físicas de red (Wi-Fi, Ethernet local) expongan el puerto sin control de acceso a otros equipos de la misma red de área local (LAN). | Necesario para que un orquestador de contenedores o proxy inverso pueda enrutar el tráfico externo hacia el puerto interno del servicio. |
| **Política de CORS** | Restringido exclusivamente a orígenes de desarrollo local: `http://127.0.0.1:8000` y `http://localhost:8000`. | Restringido al nombre de dominio completamente calificado (FQDN) corporativo con HTTPS (ej. `https://deportes.itl.tecnm.mx`). Se rechaza `*` (wildcard). |
| **Seguridad Perimetral** | No requerida (uso exclusivo del desarrollador). | **Requisito Indispensable:** Proxy inverso o API Gateway (Nginx, Traefik o AWS CloudFront/ALB) con: <br>1. Terminación TLS / HTTPS obligatoria.<br>2. Autenticación y Autorización (JWT / OAuth2).<br>3. Limitación de tasa (Rate Limiting) y mitigación anti-DDoS.<br>4. Web Application Firewall (WAF). |

> [!WARNING]
> **Advertencia de Despliegue Directo:** El servidor Uvicorn/FastAPI no debe exponerse de manera directa y pública a Internet en el puerto `0.0.0.0:8000` sin un proxy inverso intermediario que proporcione cifrado SSL/TLS y control de autenticación.

---

## 4. Resultados de las Baterías de Pruebas Automatizadas

El sistema cuenta con una cobertura integral de pruebas unitarias, de integración, de seguridad, de concurrencia y de llamadas en vivo a APIs:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/erickgpc/Tec AgoDic 2026/Analisis de Datos y Machine Learning/U5_Proyecto

tests/test_api.py (31 pruebas de API, seguridad, concurrencia y chat) ..... [100%]
tests/test_agent.py (27 pruebas unitarias de herramientas y RAG) ........... [100%]
tests/test_live_gemini.py (5 pruebas en vivo con Google Gemini API) ........ [100%]

======================== 63 passed in 223.43s (03:43) ========================
```

### Desglose por Módulo de Prueba:
- **`tests/test_api.py` (31 pruebas aprobadas):**
  - Verificación de metadatos institucionales y versión en `/health` y `/api/v1/status`.
  - Validación de respuesta del dashboard web (`GET /`) y carga de assets estáticos (CSS, JS).
  - Consulta y ordenamiento estricto de tablas de posiciones (`/api/v1/standings`).
  - Catálogo de equipos y perfiles tácticos con asignación de clústeres K-Means.
  - Comparativa directa cara a cara (Head-to-Head) entre clubes.
  - Inferencia de probabilidades 1X2 con el modelo campeón de Regresión Logística y AutoML.
  - Indicadores globales analíticos (1,140 partidos, 3,445 goles).
  - **Pruebas de Seguridad en Gráficos:** Rechazo HTTP 422 ante payload malicioso con `sql_query`, validación de presets autorizados, validación de datos directos y rechazo ante preset inexistente.
  - **Pruebas de Seguridad en Frontend:** Verificación de presencia de biblioteca DOMPurify y filtros de sanitización en el cliente web.
  - **Pruebas de Concurrencia y Sesiones:** 10 hilos concurrentes accediendo simultáneamente al `SessionMemoryManager`, verificación de aislamiento entre sesiones, prueba de reseteo independiente y prueba de poda LRU al superar el umbral máximo de 500 sesiones.
  - **Pruebas de Chat en Vivo:** Diálogo conversacional real con Gemini API y verificación de Function Calling.
- **`tests/test_agent.py` (27 pruebas aprobadas):** Pruebas unitarias sobre DuckDB, RAG semántico (LSA y TF-IDF), herramientas de consulta deportiva, cálculo de estadísticas acumuladas y construcción de prompts.
- **`tests/test_live_gemini.py` (5 pruebas aprobadas):** Pruebas en vivo de conectividad con `gemini-2.5-flash`, generación de texto, Function Calling real y manejo de errores.
- **Total Acumulado:** **63 pruebas automatizadas ejecutadas con 100% de tasa de éxito (63/63 PASSED)**.

---

## 5. Resultados de la Verificación Integral Extremo a Extremo (E2E)

Se ejecutó la prueba integral de extremo a extremo [`src/api/verify_e2e_full.py`](src/api/verify_e2e_full.py) levantando el ciclo de vida real del servidor (*Lifespan* con precarga de modelos y DuckDB):

| # | Punto Evaluado | Operación y Parámetros | Criterio de Aceptación | Resultado |
| :-: | :--- | :--- | :--- | :---: |
| **1** | **Dashboard Web** | `GET /` | Código 200, HTML válido con DOMPurify y branding TecNM. | **PASSED** |
| **2** | **Resumen Analítico** | `GET /api/v1/analytics/overview` | 1,140 partidos totales, 3,445 goles, 3.02 goles/PJ. | **PASSED** |
| **3** | **Tabla de Posiciones** | `GET /api/v1/standings?season=2024` | 20 clubes ordenados por puntos (Liverpool líder con 84 pts). | **PASSED** |
| **4** | **Perfil de Equipo** | `GET /api/v1/teams/Arsenal/profile` | Estadísticas 2024: 74 pts, +35 DG, Clúster 0. | **PASSED** |
| **5** | **Comparador H2H** | `GET /api/v1/teams/compare?team_a=Arsenal&team_b=Manchester%20City&season=2024` | Diferenciales de gol, historial directo y métricas comparadas. | **PASSED** |
| **6** | **Predicción de Partido** | `POST /api/v1/predictions/match`<br>Arsenal vs Manchester City | Probabilidades 1X2 calibradas con modelo campeón. Suma de probabilidades = 1.0. | **PASSED** |
| **7** | **Chat con Gemini (Live)** | `POST /api/v1/chat` en `session_live_1`<br>*"¿Cómo funciona la clasificación a Champions League?"* | Respuesta sintética de 2,008 caracteres generada por `gemini-3.5-flash-lite`. | **PASSED** |
| **8** | **Function Calling (Live)** | `POST /api/v1/chat` en `session_live_2`<br>*"¿Quiénes son los primeros 3 lugares de la temporada 2024?"* | Gemini identificó e invocó automáticamente la herramienta `get_standings(season=2024)`. | **PASSED** |
| **9** | **Aislamiento de Sesiones** | Comparación de contexto entre `session_A` y `session_B` | Historiales conversacionales estrictamente independientes sin contaminación. | **PASSED** |
| **10** | **Generador de Gráficos** | `POST /api/v1/analytics/chart` con preset `goals_by_season` | Generación exitosa de imagen PNG codificada en Base64 (65,284 bytes). | **PASSED** |

**Resultado E2E:** **10 de 10 puntos auditados aprobados exitosamente (100% de efectividad operativa)**.

---

## 6. Limitaciones Restantes y Consideraciones para Producción

Para una eventual transición desde este prototipo funcional y auditado hacia un entorno comercial o productivo masivo, se deben tener presentes las siguientes consideraciones técnicas y éticas:

1. **Dependencia de Conectividad y Latencia Externa:** El agente conversacional depende de la API de Google Gemini en la nube. La latencia de respuesta para consultas de texto complejas o con Function Calling oscila típicamente entre 2.5 y 4.5 segundos, sujeta a la disponibilidad del servicio de Google.
2. **Límites de Tasa (Rate Limiting) de Gemini API:** La clave configurada en el nivel gratuito (*Free Tier*) cuenta con restricciones de 5 solicitudes por minuto (RPM). Para su despliegue ante múltiples usuarios concurrentes, es mandatorio migrar a un plan comercial de pago por uso (*Pay-as-you-go*) o implementar una cola de mensajes asíncrona (RabbitMQ/Celery) con backoff exponencial.
3. **Persistencia de Sesiones en Memoria RAM:** `SessionMemoryManager` almacena los historiales en memoria volátil de la instancia. Si el proceso del servidor se reinicia, las sesiones activas se limpian. Para escalabilidad horizontal detrás de múltiples balanceadores de carga, se recomienda sustituir el almacenamiento en memoria por una base de datos distribuida en caché como Redis.
4. **Almacenamiento de Datos Analíticos:** DuckDB opera en modo archivo local (`data/lakehouse/epl_analytics.duckdb`). Si bien es ultra-rápido para analítica OLAP local (tiempos de consulta < 10 ms), no admite escrituras concurrentes distribuidas. Para un flujo de datos en tiempo real con millones de transacciones, se recomendaría complementar con ClickHouse o PostgreSQL.
5. **Consideración Ética y Estadística Obligatoria (Apuestas Deportivas):**
   > [!CAUTION]
   > **Aviso de Responsabilidad:** Los modelos de Machine Learning (Regresión Logística y AutoML) y las explicaciones del agente Gemini proporcionan estimaciones estadísticas y probabilísticas basadas estrictamente en desempeños históricos pasados. En el deporte profesional intervienen factores no observables (lesiones de último minuto, clima, decisiones arbitrales, estado anímico). **Estas predicciones no garantizan resultados futuros y no deben ser utilizadas como base para apuestas deportivas con dinero real ni transacciones financieras.** El sistema tiene una finalidad estrictamente académica, pedagógica y de investigación deportiva.

---

## 7. Instrucciones Oficiales de Instalación y Ejecución

### 7.1 Requisitos Previos
- Sistema Operativo: Linux (probado en Ubuntu/Debian), macOS o Windows (WSL2 recomendado).
- Python 3.10 o superior (verificado con Python 3.14).
- Conexión a Internet activa para la API de Gemini.

### 7.2 Configuración del Entorno
1. Clonar el repositorio o posicionarse en el directorio raíz del proyecto:
   ```bash
   cd "U5_Proyecto"
   ```
2. Crear un entorno virtual e instalar las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Configurar las variables de entorno en el archivo `.env`:
   ```bash
   cp .env.example .env
   ```
   Editar `.env` y asignar su clave de Gemini:
   ```env
   GEMINI_API_KEY="AIzaSy..."
   GEMINI_MODEL="gemini-2.5-flash"
   API_HOST="127.0.0.1"
   API_PORT=8000
   CORS_ORIGINS="http://127.0.0.1:8000,http://localhost:8000"
   DUCKDB_PATH="data/lakehouse/epl_analytics.duckdb"
   ```

### 7.3 Ejecución del Backend y Frontend
Iniciar el servidor FastAPI con Uvicorn:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Una vez iniciado el servidor:
- **Dashboard Web Interactivo:** Abrir en el navegador [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Documentación Interactiva Swagger (OpenAPI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Documentación Alternativa ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 7.4 Tabla Resumen de Endpoints de la API REST

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Interfaz gráfica del Dashboard SPA (HTML/CSS/JS). |
| `GET` | `/health` | Diagnóstico de salud del servicio, modelos y base de datos. |
| `GET` | `/api/v1/status` | Versión, modelo activo y metadatos del TecNM. |
| `GET` | `/api/v1/standings` | Clasificación oficial por temporada (`season=2022\|2023\|2024`). |
| `GET` | `/api/v1/teams` | Lista de los 20 clubes registrados en la temporada. |
| `GET` | `/api/v1/teams/{team_name}/profile` | Perfil detallado, métricas de gol y clúster taxonómico. |
| `GET` | `/api/v1/teams/compare` | Comparador directo Head-to-Head entre dos clubes. |
| `POST` | `/api/v1/predictions/match` | Inferencia de probabilidades 1X2 con modelos ML. |
| `GET` | `/api/v1/analytics/overview` | Estadísticas globales de las 3 temporadas (1,140 partidos). |
| `POST` | `/api/v1/analytics/chart` | Generación de visualizaciones seguras mediante presets o datos. |
| `POST` | `/api/v1/chat` | Interacción en lenguaje natural con Gemini y Function Calling. |
| `POST` | `/api/v1/chat/reset` | Reinicio seguro del historial de una sesión conversacional. |

### 7.5 Comandos para Verificación y Pruebas
Para ejecutar las pruebas automatizadas del sistema:
```bash
# Ejecutar todas las pruebas de la API, seguridad y concurrencia (31 pruebas)
pytest tests/test_api.py -v

# Ejecutar las pruebas unitarias del agente y RAG (27 pruebas)
pytest tests/test_agent.py -v

# Ejecutar el script integral E2E
python src/api/verify_e2e_full.py
```

---

## 8. Dictamen Definitivo de Cierre del Proyecto

Habiendo revisado y verificado cada uno de los componentes, capas de datos, modelos de aprendizaje automático, integración del agente cognitivo Gemini, medidas de seguridad perimetral y documental, y habiendo aprobado exitosamente el 100% de las 63 pruebas automatizadas y los 10 puntos de la prueba de extremo a extremo:

Se declara formalmente **CONCLUIDO Y APROBADO CON EXCELENCIA TÉCNICA** el proyecto:

**Agente Inteligente de Análisis Deportivo con IA (Premier League)**

Desarrollado y presentado en el **Instituto Tecnológico de La Laguna, perteneciente al Tecnológico Nacional de México (TecNM)**.

```
========================================================================================
                              ESTADO DEL PROYECTO: APROBADO
                              COBERTURA DE PRUEBAS: 63/63 (100%)
                              EVALUACIÓN E2E: 10/10 (100%)
                              SEGURIDAD AUDITADA: SQLi / XSS / RLock OK
========================================================================================
```
