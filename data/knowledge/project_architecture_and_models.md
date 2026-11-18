# Arquitectura Técnica del Proyecto y Modelado Predictivo

## 1. Flujo de Datos y Lakehouse
El sistema implementa una arquitectura desacoplada y escalable:
1. **Ingesta Oficial:** Extracción paginada desde la API BALLDONTLIE EPL V2 (1,140 partidos entre 2022 y 2025). Checkpoints atómicos para tolerancia a fallos.
2. **Batch ETL con PySpark Local:** Limpieza, tipado y validación de esquemas con Pydantic. Particionado en disco por temporada en formato Parquet (`data/processed/matches.parquet`).
3. **Capa Analítica con DuckDB:** Registro de tablas analíticas (`matches`, `teams`, `matches_features`, `teams_clustered`) en `data/lakehouse/epl.duckdb`, permitiendo consultas SQL vectorizadas de ultra-baja latencia sin necesidad de mantener servicios de base de datos activos en segundo plano.

## 2. Ingeniería de Características (Zero Leakage)
Se construyeron 26 variables predictoras exclusivamente pre-partido:
- Variables móviles de los últimos 5 partidos (puntos, goles anotados, goles recibidos, diferencia de goles) excluyendo estrictamente el encuentro en juego.
- Métricas acumuladas de la temporada (puntos acumulados, partidos jugados, puntos por partido acumulados).
- Días de descanso entre partidos para medir fatiga.
- Variables de control como `home_games_tracked` y `away_games_tracked` para inicios de temporada.

## 3. Desempeño de los Modelos Supervisados (Test Set 2024-25)
- **División Temporal:** Entrenamiento en temporadas 2022-23 y 2023-24 (760 partidos); Prueba en temporada 2024-25 (380 partidos).
- **Modelo Campeón (Regresión Logística Balanceada):**
  - Macro-F1: 0.4581 (el más alto de todos los modelos).
  - Weighted-F1: 0.4868.
  - Accuracy: 49.47%.
  - Recall de Empates: 22.58% (acertó 21 empates reales).
  - Log Loss: 1.0288.
- **FLAML AutoML (LRL1 - Regresión Logística L1 SAGA):**
  - Log Loss: 1.0271 (el menor error probabilístico global).
  - Accuracy: 50.53%.
  - Recall de Empates: 0.00% (al optimizar log loss sin balanceo forzado de clases, P(Empate) nunca superó el umbral argmax de 0.3388).
- **Random Forest:**
  - Sin balancear: 51.84% accuracy, pero solo 1.08% de recall en empates.
  - Balanceado: 48.95% accuracy, 41.86% Macro-F1, elevando el recall de empate a 9.68%.
