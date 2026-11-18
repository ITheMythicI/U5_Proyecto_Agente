# PLAN MAESTRO DE ARQUITECTURA E IMPLEMENTACIÓN (FASE 4)
## API REST con FastAPI y Dashboard Interactivo de Análisis Deportivo (Premier League)

**Proyecto:** Agente Inteligente de Análisis Deportivo con IA (Instituto Tecnológico de La Laguna - TecNM)  
**Fecha:** 18 de noviembre de 2026  
**Versión del Plan:** 1.0.0  
**Estado:** Aprobado para ejecución  

---

## 1. Resumen Ejecutivo y Objetivos

El objetivo de la **Fase 4** es convertir la infraestructura analítica construida y validada en las Fases 1, 2 y 3 (Lakehouse DuckDB con 1,140 partidos, modelos supervisados y no supervisados `.joblib`, RAG dual semántico y agente Gemini con Function Calling real) en una **aplicación web de nivel de producción**, modular, desacoplada, segura e interactiva.

### Metas Fundamentales:
1. **API REST de Alto Rendimiento:** Implementar un backend en **FastAPI** completamente tipado con **Pydantic v2**, ciclo de vida controlado (`lifespan`) para recursos pesados (modelos, base de datos y vector stores en memoria), gestión centralizada de excepciones y documentación interactiva OpenAPI/Swagger.
2. **Gestor de Sesiones para el Agente:** Extender la memoria conversacional de Fase 3 mediante un `SessionMemoryManager` para soportar múltiples usuarios/sesiones simultáneas con contexto independiente.
3. **Consumo de Modelos sin Reentrenamiento:** Reutilizar directamente el modelo campeón (`logistic_regression_balanced.joblib`), el modelo alternativo (`flaml_automl_best.joblib`) y el modelo K-Means (`kmeans_cluster_model.joblib`).
4. **Dashboard Profesional y Responsive:** Diseñar una interfaz interactiva de analítica deportiva profesional estructurada en 6 módulos clave: *Overview*, *Clasificación*, *Equipos*, *Predicciones*, *Asistente IA* y *Storytelling*.
5. **Seguridad Integral:** Mantener DuckDB en modo `read_only = True`, deshabilitar ejecución arbitraria de código por HTTP, no exponer credenciales al cliente y restringir CORS.

---

## 2. Auditoría del Estado Actual y Reutilización de Módulos

| Componente Existente | Ubicación | Estado | Estrategia de Reutilización en Fase 4 |
| :--- | :--- | :---: | :--- |
| **Lakehouse DuckDB** | `data/lakehouse/epl.duckdb` | 100% Validado | Conexión única en modo `read_only=True` cargada en el ciclo de vida (`lifespan`) del backend. |
| **Modelos de ML** | `models/*.joblib` | Entrenados (Fase 2) | Precargados en memoria durante el inicio del servidor para inferencia en tiempo submilisegundo sin reentrenamiento. |
| **Módulo de Herramientas** | `src/agent/tools.py` | Validado (Fase 3) | Reutilizado directamente por los controladores de FastAPI (`standings`, `profile`, `predict_match`, `generate_chart`). |
| **Agente Gemini (AFC)** | `src/agent/agent.py` | Validado (Fase 3) | Orquestador central para el endpoint `/api/v1/chat` con fallback dinámico y reintentos ante 429. |
| **RAG Dual** | `src/agent/rag.py` | Validado (Fase 3) | Conservado con selección transparente de modo (`gemini_neural_3072d` vs `local_lsa_32d`). |
| **Memoria Conversacional** | `src/agent/memory.py` | Validado (Fase 3) | Integrado bajo un gestor concurrente de sesiones (`SessionMemoryManager`). |
| **Datos Parquet Históricos** | `data/processed/*.parquet` | Congelados | Base inmutable de consulta. |

---

## 3. Justificación Tecnológica

### 3.1 Backend: FastAPI + Uvicorn + Pydantic v2
* **Razón de Selección:** FastAPI ofrece rendimiento asíncrono superior (basado en Starlette y AnyIO), validación automática y tipado estricto en tiempo de ejecución con Pydantic v2, y generación nativa de esquemas OpenAPI (`/docs` y `/redoc`).
* **Ciclo de Vida Controlado (`lifespan`):** La inicialización de la base DuckDB, los clasificadores de Machine Learning y el indexador semántico de RAG se ejecuta una sola vez al arrancar Uvicorn, garantizando que ninguna petición HTTP incurra en sobrecostos de lectura de disco o deserialización de modelos.

### 3.2 Frontend: SPA Moderna con Tailwind CSS, Vanilla JS y Chart.js
* **Evaluación Comparativa:**
  * *Streamlit:* Aunque permite prototipado rápido en Python, su arquitectura monolítica con recargas de script por estado y servidor acoplado en puerto 8501 contradice el requisito de una arquitectura cliente-servidor REST desacoplada y dificulta el control fino de la experiencia de usuario móvil.
  * *SPA Compilada Pesada (React/Vue/Angular con Node Build):* Requiere dependencias externas de Node.js, procesos adicionales de compilación en segundo plano y mayor complejidad de despliegue.
  * *SPA Desacoplada (HTML5 + Tailwind CSS + Vanilla ES6 + Chart.js + Marked.js):* **Solución Óptima Elegida**. 
    1. **100% Desacoplada:** Consume exclusivamente los endpoints JSON de la API REST mediante `fetch()`.
    2. **Rendimiento Máximo:** Cero tiempo de compilación, carga instantánea en navegador y compatibilidad universal con escritorio y dispositivos móviles.
    3. **Visualizaciones Interactivas:** Gráficas dinámicas renderizadas en cliente con **Chart.js** y gráficos estadísticos de alta densidad generados por el backend en Matplotlib/Seaborn.
    4. **Seguridad Absoluta:** Ninguna credencial de Gemini o variable de entorno se transmite al cliente.

---

## 4. Especificación de Endpoints de la API REST

### 4.1 Sistema y Diagnóstico
* `GET /health`: Estado de salud, marca de tiempo y verificación del Lakehouse.
* `GET /api/v1/status`: Diagnóstico detallado (modelo LLM activo, tablas disponibles en DuckDB, estado de RAG y modelos de Machine Learning cargados).

### 4.2 Agente Conversacional
* `POST /api/v1/chat`:
  * *Request Body:* `{"message": str, "session_id": Optional[str], "forced_rag_mode": Optional[str]}`
  * *Response:* `{"session_id": str, "response": str, "mode": str, "model_name": str, "tool_calls": List[dict], "memory_entities": dict}`
* `POST /api/v1/chat/reset`:
  * *Request Body:* `{"session_id": str}`
  * *Response:* `{"status": "success", "message": "Memoria conversacional reiniciada.", "session_id": str}`

### 4.3 Datos Deportivos del Lakehouse
* `GET /api/v1/standings?season=2024`: Tabla de posiciones con estadísticas completas y asignación al clúster K-Means actualizado (*Alto Rendimiento / Contendientes Europeos*).
* `GET /api/v1/teams`: Lista canónica de los 24 clubes ingleses rastreados.
* `GET /api/v1/teams/{team_name}/profile?season=2024`: Perfil táctico, clúster y coordenadas PCA de un equipo.
* `GET /api/v1/teams/compare?team_a=Arsenal&team_b=Manchester%20City&season=2024`: Comparación lado a lado de métricas ofensivas y defensivas.

### 4.4 Predicciones y Machine Learning
* `POST /api/v1/predictions/match`:
  * *Request Body:* `{"home_team": str, "away_team": str, "model_type": Optional[str] = "champion"}`
  * *Response:* Probabilidades `Local`, `Empate`, `Visita`, resultado estimado, factores estadísticos y advertencia metodológica obligatoria.

### 4.5 Analítica y Visualizaciones
* `GET /api/v1/analytics/overview`: Resumen global de los 1,140 partidos (goles totales, porcentaje de victorias locales/visitantes/empates, promedios).
* `GET /api/v1/analytics/seasons`: Desglose comparativo entre las temporadas 2022, 2023 y 2024.
* `POST /api/v1/analytics/chart`: Generador de gráficos analíticos pre-renderizados en base64 para visualizaciones de servidor.

---

## 5. Módulos del Dashboard Interactivo

1. **A. Inicio / Overview:** Tarjetas KPI con total de partidos analizados (1,140), temporadas auditadas (3), número de equipos únicos (24), distribución de resultados 1X2 y balance global de goles.
2. **B. Clasificación:** Tabla de posiciones interactiva con ordenamiento, filtrado por temporada (2022, 2023, 2024), insignias visuales de los tres estratos competitivos K-Means y tarjetas de leyenda analítica.
3. **C. Equipos:** Explorador individual de clubes con estadísticas históricas acumuladas, gráficos de radar/barras ofensivas vs defensivas y selector de comparación directa entre dos equipos.
4. **D. Predicciones:** Interfaz de pronósticos 1X2 con selectores de equipo local y visitante, selector de modelo (*Champion* / *AutoML*), visualizador de barras de probabilidad porcentual, resultado estimado, desglose de métricas pre-partido e indicador de advertencia estadística.
5. **E. Asistente IA:** Interfaz de chat moderna con soporte de markdown, selector de sesión, historial en tiempo real, chips de herramientas invocadas por Function Calling y renderizado de gráficos embebidos.
6. **F. Storytelling:** Galería de consultas predefinidas de alto impacto analítico ("¿Cómo evolucionó el Manchester City?", "Comparar capacidad ofensiva de Arsenal y Liverpool", "Patrones en equipos de alto rendimiento") que ejecutan análisis multivariable con gráficos automáticos.

---

## 6. Plan de Implementación Incremental

1. **Paso 1:** Definir esquemas Pydantic v2 en `src/api/schemas.py` y gestor de sesiones en `src/api/session_manager.py`.
2. **Paso 2:** Construir la aplicación FastAPI modular en `src/api/main.py` y `src/api/routes/` con routers especializados y manejo centralizado de excepciones.
3. **Paso 3:** Implementar el Dashboard web en `src/frontend/` (HTML, CSS y JS modular) y montarlo en FastAPI mediante `StaticFiles`.
4. **Paso 4:** Desarrollar batería de pruebas exhaustivas en `tests/test_api.py` (endpoints, esquemas, sesiones, errores y seguridad).
5. **Paso 5:** Generar `reporte_fase4.md`, actualizar dependencias y preparar guía de ejecución.
