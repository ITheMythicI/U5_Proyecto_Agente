# REPORTE TÉCNICO DE IMPLEMENTACIÓN Y AUDITORÍA DE SEGURIDAD (FASE 4)
## API REST con FastAPI y Dashboard Interactivo de Análisis Deportivo (Premier League)

**Proyecto:** Agente Inteligente de Análisis Deportivo con IA  
**Institución:** Instituto Tecnológico de La Laguna, perteneciente al Tecnológico Nacional de México (TecNM)  
**Fecha:** 18 de noviembre de 2026  
**Versión del Sistema:** 1.1.0 (Producción Auditada)  
**Ubicación del Código:** [`src/api/`](src/api/) y [`src/frontend/`](src/frontend/)  
**Plan de Arquitectura:** [`plan_fase4.md`](plan_fase4.md)  
**Auditoría E2E Ejecutada:** [`src/api/verify_e2e_full.py`](src/api/verify_e2e_full.py) (**10/10 puntos aprobados al 100%**).

---

## 1. Resumen Ejecutivo

La **Fase 4** ha culminado con una auditoría integral de seguridad, concurrencia y consistencia documental, consolidando el Data Lakehouse (1,140 partidos), los modelos supervisados de Machine Learning y el agente inteligente Gemini en una **plataforma web de analítica deportiva desacoplada, segura y lista para operar**:

1. **Blindaje del Endpoint de Gráficos (`/api/v1/analytics/chart`):** Se eliminó cualquier posibilidad de inyección SQL mediante la prohibición estricta de consultas SQL enviadas por el cliente (`extra="forbid"` en Pydantic). El backend utiliza exclusivamente un catálogo de consultas analíticas pre-aprobadas y parametrizadas (`preset`), conservando DuckDB en modo de solo lectura (`read_only=True`).
2. **Sanitización Anti-XSS en el Frontend:** Se integró **DOMPurify** en la interfaz web, sanitizando todas las salidas de Markdown generadas por Gemini antes de su inserción en el DOM, neutralizando la ejecución de scripts, atributos de eventos maliciosos (`onerror`, `onload`) y esquemas peligrosos.
3. **Gestión Concurrente de Sesiones Thread-Safe:** Se reforzó [`session_manager.py`](src/api/session_manager.py) mediante `threading.RLock()`, garantizando aislamiento absoluto entre sesiones, protección ante solicitudes simultáneas y poda controlada bajo política LRU (máximo 500 sesiones).
4. **Configuración Segura de Red:** Se documentó y configuró por defecto el enlace a `127.0.0.1` (loopback) para desarrollo local, diferenciándolo formalmente de entornos contenerizados o con Reverse Proxy (`0.0.0.0`), con CORS restrictivo.
5. **Alineación Institucional Oficial:** Se corrigieron todas las referencias institucionales en código, interfaz, esquemas y documentación, acreditando formalmente al **Instituto Tecnológico de La Laguna (TecNM)**.

---

## 2. Auditoría de Seguridad y Endurecimiento de Componentes

### 2.1 Blindaje de `POST /api/v1/analytics/chart`
- **Vulnerabilidad Previa Identificada:** El esquema original aceptaba una cadena libre `sql_query` opcional, lo cual generaba una superficie de ataque para inyecciones SQL o intentos de lectura de rutas no autorizadas.
- **Correcciones Implementadas:**
  1. Se eliminó el campo `sql_query` de [`ChartRequest`](src/api/schemas.py) y se configuró `model_config = {"extra": "forbid"}`. Cualquier solicitud que intente enviar `sql_query` es rechazada de inmediato por FastAPI con código HTTP 422.
  2. Se implementó una lista blanca de plantillas analíticas pre-autorizadas en [`analytics.py`](src/api/routes/analytics.py):
     - `goals_by_season`: Agregación de goles en partidos oficiales por temporada.
     - `top_scoring_teams`: Top 10 equipos más goleadores de la temporada 2024.
     - `points_distribution`: Puntos de todos los clubes de la temporada 2024.
     - `cluster_summary`: Cantidad de clubes por estrato taxonómico K-Means.
  3. Soporte alternativo seguro para datos estructurados primitivos en el campo `data` (`categories` y `values`), graficados directamente en memoria sin intermediación SQL.

### 2.2 Prevención de Cross-Site Scripting (XSS) en Chat Markdown
- **Vulnerabilidad Previa Identificada:** En [`src/frontend/static/js/app.js`](src/frontend/static/js/app.js), la función `appendBotMessage` insertaba el HTML resultante de `marked.parse()` mediante `div.innerHTML`. Si una respuesta o inyección externa contenía `<script>` o `<img onerror=...>`, existía riesgo de ejecución no deseada en el navegador del usuario.
- **Correcciones Implementadas:**
  1. Se cargó la librería oficial **DOMPurify** (v3.2.4) en [`index.html`](src/frontend/index.html).
  2. Se envolvió el parseo con `DOMPurify.sanitize(rawHtml, {...})` con lista blanca de etiquetas semánticas (`p`, `table`, `tr`, `td`, `th`, `code`, `pre`, `ul`, `ol`, `li`, `strong`, `em`) y bloqueo taxativo de `script`, `iframe`, `object`, `form` y atributos de eventos (`onerror`, `onload`, `onclick`).

### 2.3 Seguridad ante Hilos y Concurrencia en Sesiones
- En [`session_manager.py`](src/api/session_manager.py):
  - Todas las operaciones sobre el diccionario `self._sessions` (`get_or_create`, `reset_session`, `delete_session`, poda LRU) están sincronizadas mediante un cerrojo reentrante `threading.RLock()`.
  - Probado exitosamente con 10 hilos concurrentes simultáneos sin colisiones ni fuga de estado.

---

## 3. Configuración de Red y Despliegue

| Parámetro | Entorno de Desarrollo Local | Entorno de Despliegue en Servidor |
| :--- | :--- | :--- |
| **Host de Enlace** | `127.0.0.1` (Loopback obligatorio) | `0.0.0.0` (exclusivo dentro de contenedores o red interna) |
| **Justificación de Host** | Impide que otros dispositivos en la misma red Wi-Fi/LAN accedan al servicio sin autenticación. | Permite que el proxy inverso (Nginx, Traefik o AWS ALB) enrute las peticiones hacia el contenedor. |
| **CORS** | `http://127.0.0.1:8000,http://localhost:8000` | Dominio corporativo seguro (e.g. `https://epl.itl.tecnm.mx`) |
| **Seguridad Perimetral** | No requerida para pruebas en localhost. | **Obligatorio:** Reverse Proxy con terminación TLS/HTTPS, autenticación (JWT/OAuth2), WAF y limitación de tasa (Rate Limiting). |

> **Declaración de Estado:** Esta aplicación no está diseñada para exponerse de forma pública y directa a Internet sin un Gateway de seguridad perimetral intermedio que gestione autenticación y mitigación de denegación de servicio (DDoS).

---

## 4. Resultados de la Verificación Integral de Extremo a Extremo (E2E)

Se ejecutó el script de auditoría E2E [`verify_e2e_full.py`](src/api/verify_e2e_full.py) con el ciclo de vida real del servidor:

| # | Escenario Evaluado | Verificación Técnica Realizada | Resultado |
| :-: | :--- | :--- | :--- | :---: |
| **1** | **Apertura del Dashboard** | Carga de `GET /`, validación de DOMPurify e identidad TecNM en HTML. | **PASSED** |
| **2** | **Carga de Indicadores Globales** | `GET /api/v1/analytics/overview`: 1,140 partidos, 3,445 goles, 3.02 goles/PJ. | **PASSED** |
| **3** | **Clasificación por Temporada** | `GET /api/v1/standings?season=2024`: 20 clubes, líder Liverpool (84 pts). | **PASSED** |
| **4** | **Perfil Táctico de Equipo** | `GET /api/v1/teams/Arsenal/profile`: 74 pts, +35 DG, Clúster 0. | **PASSED** |
| **5** | **Comparador Directo (H2H)** | `GET /api/v1/teams/compare?team_a=Arsenal&team_b=Manchester%20City&season=2024`. | **PASSED** |
| **6** | **Predicción de Partido (1X2)** | `POST /api/v1/predictions/match`: Inferencia con modelo campeón (`logistic_regression_balanced.joblib`). Resultado más probable: Empate. | **PASSED** |
| **7** | **Conversación Real con Gemini** *(Live)* | `POST /api/v1/chat`: Consulta sobre reglas Champions League. Respuesta de 2,008 caracteres generada por `gemini-3.5-flash-lite`. | **PASSED** |
| **8** | **Function Calling desde Chat** *(Live)* | `POST /api/v1/chat`: Gemini identificó e invocó automáticamente `get_standings` para devolver la tabla 2024. | **PASSED** |
| **9** | **Aislamiento de Sesiones** | Consultas en `session-A` (Chelsea) y `session-B` (saludo). Verificado que no hay fuga de entidades entre sesiones. | **PASSED** |
| **10** | **Generación Segura de Gráficas** | `POST /api/v1/analytics/chart` con preset `goals_by_season`: PNG en Base64 de 65,284 bytes generado. | **PASSED** |

---

## 5. Baterías de Pruebas Automatizadas

```text
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/erickgpc/Tec AgoDic 2026/Analisis de Datos y Machine Learning/U5_Proyecto

tests/test_api.py (31 pruebas de API, seguridad y concurrencia) ...... [100%]
tests/test_agent.py (27 pruebas unitarias de herramientas y RAG) ..... [100%]
tests/test_live_gemini.py (5 pruebas en vivo contra Gemini API) ....... [100%]

======================== 63 passed in 223.43s ========================
```

---

## 6. Dictamen de Entrega

La **Fase 4** y el proyecto en su totalidad cumplen rigurosamente con los estándares analíticos, metodológicos, de seguridad y documentación exigidos.
