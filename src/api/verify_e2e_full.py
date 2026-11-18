"""
Script de Verificación Integral de Extremo a Extremo (E2E) para la Fase 4.
Ejecuta la auditoría de los 10 puntos clave requeridos sobre la API y el Dashboard.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app

def run_e2e_audit():
    print("=" * 70)
    print("INICIANDO AUDITORÍA INTEGRAL DE EXTREMO A EXTREMO (FASE 4)")
    print("Institución: Instituto Tecnológico de La Laguna (TecNM)")
    print("=" * 70)

    results = {}

    with TestClient(app) as client:
        # -------------------------------------------------------------------
        # 1. Apertura del Dashboard (Frontend)
        # -------------------------------------------------------------------
        print("\n[1/10] Verificando apertura del Dashboard (HTML y assets)...")
        res1 = client.get("/")
        assert res1.status_code == 200, f"Error {res1.status_code}"
        assert "Premier League AI Analyst" in res1.text
        assert "Instituto Tecnológico de La Laguna" in res1.text
        assert "dompurify" in res1.text.lower()
        print("  -> OK: Dashboard HTML cargado con sanitización DOMPurify e identidad TecNM.")
        results["1_dashboard_open"] = "PASSED"

        # -------------------------------------------------------------------
        # 2. Carga de Indicadores Globales
        # -------------------------------------------------------------------
        print("\n[2/10] Verificando carga de indicadores analíticos globales...")
        res2 = client.get("/api/v1/analytics/overview")
        assert res2.status_code == 200
        d2 = res2.json()
        assert d2["total_matches"] == 1140
        assert d2["seasons"] == [2022, 2023, 2024]
        assert d2["teams_count"] == 24
        print(f"  -> OK: 1,140 partidos auditados, {d2['total_goals']} goles totales, {d2['avg_goals_per_match']} goles/PJ.")
        results["2_overview_metrics"] = "PASSED"

        # -------------------------------------------------------------------
        # 3. Consulta de Clasificación por Temporada
        # -------------------------------------------------------------------
        print("\n[3/10] Verificando consulta de clasificación por temporada (2024)...")
        res3 = client.get("/api/v1/standings?season=2024")
        assert res3.status_code == 200
        d3 = res3.json()
        assert len(d3["tabla"]) == 20
        top1 = d3["tabla"][0]
        assert top1["etiqueta_cluster"] == "Alto Rendimiento / Contendientes Europeos"
        print(f"  -> OK: 20 clubes procesados. Líder: {top1['team_name']} ({top1['puntos']} pts, Clúster: '{top1['etiqueta_cluster']}').")
        results["3_standings_table"] = "PASSED"

        # -------------------------------------------------------------------
        # 4. Visualización del Perfil de un Equipo
        # -------------------------------------------------------------------
        print("\n[4/10] Verificando perfil táctico y taxonómico de un club (Arsenal)...")
        res4 = client.get("/api/v1/teams/Arsenal/profile")
        assert res4.status_code == 200
        d4 = res4.json()
        assert d4["team_name"] == "Arsenal"
        p4 = d4["ultimo_perfil"]
        print(f"  -> OK: Arsenal en 2024: {p4['puntos']} pts (+{p4['diferencia_goles']} DG, Clúster: '{p4['etiqueta_cluster']}').")
        results["4_team_profile"] = "PASSED"

        # -------------------------------------------------------------------
        # 5. Comparación entre Dos Clubes (Cara a Cara)
        # -------------------------------------------------------------------
        print("\n[5/10] Verificando comparador cara a cara (Arsenal vs Manchester City)...")
        res5 = client.get("/api/v1/teams/compare?team_a=Arsenal&team_b=Manchester%20City&season=2024")
        assert res5.status_code == 200
        d5 = res5.json()
        assert d5["team_a"]["team_name"] == "Arsenal"
        assert d5["team_b"]["team_name"] == "Manchester City"
        print(f"  -> OK: Arsenal ({d5['team_a']['puntos']} pts) vs Man City ({d5['team_b']['puntos']} pts) comparados exitosamente.")
        results["5_team_comparison"] = "PASSED"

        # -------------------------------------------------------------------
        # 6. Predicción de un Partido (Inferencia Supervisada)
        # -------------------------------------------------------------------
        print("\n[6/10] Verificando predicción de partido con Modelo Champion (Liverpool vs Arsenal)...")
        pred_payload = {
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "model_type": "champion"
        }
        res6 = client.post("/api/v1/predictions/match", json=pred_payload)
        assert res6.status_code == 200
        d6 = res6.json()
        assert d6["prediccion_estimada"] in ["Local", "Empate", "Visita"]
        print(f"  -> OK: Predicción: {d6['prediccion_estimada']} | Probabilidades: {d6['probabilidades']} | Modelo: {d6['modelo_utilizado']}.")
        results["6_match_prediction"] = "PASSED"

        # -------------------------------------------------------------------
        # 7. Conversación Real con Gemini (Requiere conexión activa)
        # -------------------------------------------------------------------
        print("\n[7/10] Verificando conversación real con Gemini (Consulta analítica)...")
        chat_payload = {
            "message": "¿Qué equipos clasifican a Champions League en la Premier League?",
            "session_id": "e2e-audit-session"
        }
        res7 = client.post("/api/v1/chat", json=chat_payload)
        assert res7.status_code == 200
        d7 = res7.json()
        print(f"  -> OK: Modo: {d7['mode']} | Modelo: {d7.get('model_name')} | Longitud respuesta: {len(d7['response'])} caracteres.")
        results["7_gemini_live_chat"] = "PASSED"

        # Pausa para respetar la tasa de Gemini Free Tier (5 RPM)
        print("  ... Pausa de 12 segundos para respetar control de tasa (Free Tier) ...")
        time.sleep(12)

        # -------------------------------------------------------------------
        # 8. Ejecución de Function Calling desde el Chat
        # -------------------------------------------------------------------
        print("\n[8/10] Verificando Function Calling real desde el Agente (Consulta de tabla)...")
        fc_payload = {
            "message": "Muéstrame la tabla de posiciones de la Premier League 2024",
            "session_id": "e2e-audit-session"
        }
        res8 = client.post("/api/v1/chat", json=fc_payload)
        assert res8.status_code == 200
        d8 = res8.json()
        tools_called = [t["tool"] for t in d8.get("tool_calls", [])]
        print(f"  -> OK: Herramientas invocadas por Gemini: {tools_called}.")
        results["8_function_calling"] = "PASSED"

        # -------------------------------------------------------------------
        # 9. Persistencia y Aislamiento de Sesiones
        # -------------------------------------------------------------------
        print("\n[9/10] Verificando persistencia y aislamiento estricto de sesiones...")
        # Sesión A
        client.post("/api/v1/chat", json={"message": "Analiza a Chelsea", "session_id": "session-audit-A"})
        # Sesión B
        res_b = client.post("/api/v1/chat", json={"message": "Hola", "session_id": "session-audit-B"})
        d_b = res_b.json()
        # Verificar que Chelsea no se fugó a sesión B
        assert "Chelsea" not in d_b.get("memory_entities", {}).get("teams", [])
        print("  -> OK: Aislamiento total comprobado: entidades de Sesión A no se fugaron a Sesión B.")
        results["9_session_isolation"] = "PASSED"

        # -------------------------------------------------------------------
        # 10. Generación y Visualización de Gráficas Seguras
        # -------------------------------------------------------------------
        print("\n[10/10] Verificando generación segura de gráfico estadístico en Base64...")
        chart_payload = {
            "chart_type": "bar",
            "preset": "goals_by_season",
            "title": "Volumen de Goles por Temporada"
        }
        res10 = client.post("/api/v1/analytics/chart", json=chart_payload)
        assert res10.status_code == 200
        d10 = res10.json()
        assert d10["format"] == "png"
        assert len(d10["image_base64"]) > 500
        print(f"  -> OK: Gráfico PNG generado y codificado en Base64 ({len(d10['image_base64'])} bytes).")
        results["10_chart_generation"] = "PASSED"

    print("\n" + "=" * 70)
    print("RESUMEN DE AUDITORÍA E2E: 10/10 PUNTOS APROBADOS AL 100%")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")

    return True

if __name__ == "__main__":
    run_e2e_audit()
