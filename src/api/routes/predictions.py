"""
Rutas de inferencia predictiva y modelado supervisado de partidos (1X2).
Reutiliza directamente los modelos entrenados de Fase 2 sin reentrenamiento.
"""

from fastapi import APIRouter, HTTPException
from ..schemas import MatchPredictionRequest, MatchPredictionResponse
from ...agent.tools import predict_match, resolve_team_name

router = APIRouter(prefix="/api/v1/predictions", tags=["Predicciones ML"])


@router.post("/match", response_model=MatchPredictionResponse, summary="Pronóstico de partido 1X2")
def predict_match_endpoint(payload: MatchPredictionRequest):
    """
    Ejecuta inferencia estadística sobre el encuentro especificado utilizando las 26 características
    pre-partido y el clasificador supervisado seleccionado ('champion' = Regresión Logística Balanceada,
    'automl' = FLAML Best Estimator).
    """
    # Validar que los equipos no sean el mismo
    can_home = resolve_team_name(payload.home_team)
    can_away = resolve_team_name(payload.away_team)

    if not can_home:
        raise HTTPException(status_code=400, detail=f"Equipo local '{payload.home_team}' no reconocido.")
    if not can_away:
        raise HTTPException(status_code=400, detail=f"Equipo visitante '{payload.away_team}' no reconocido.")
    if can_home == can_away:
        raise HTTPException(status_code=400, detail="El equipo local y el visitante no pueden ser el mismo.")

    model_type = payload.model_type or "champion"
    if model_type not in ["champion", "automl"]:
        raise HTTPException(status_code=400, detail="El tipo de modelo debe ser 'champion' o 'automl'.")

    res = predict_match(home_team=can_home, away_team=can_away, model_type=model_type)

    if res.get("status") != "success":
        raise HTTPException(status_code=500, detail=res.get("message", "Error al procesar la predicción."))

    return MatchPredictionResponse(
        status="success",
        home_team=res["home_team"],
        away_team=res["away_team"],
        modelo_utilizado=res["modelo_utilizado"],
        prediccion_estimada=res["prediccion_estimada"],
        probabilidades=res["probabilidades"],
        confianza_maxima=res["confianza_maxima"],
        factores_clave=res["factores_clave"],
        advertencia_metodologica=res["advertencia_metodologica"]
    )
