"""
Batería de Pruebas Automatizadas para la API REST de FastAPI y Dashboard (Fase 4).
Valida endpoints del sistema, datos deportivos, predicciones ML, analítica, chat, sesiones y seguridad.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app

# Cliente de pruebas síncrono para FastAPI
client = TestClient(app)


# ===========================================================================
# 1. PRUEBAS DE SISTEMA Y SALUD
# ===========================================================================

class TestSystemEndpoints:
    """Verificación de conectividad, salud y diagnóstico de subsistemas."""

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "connected"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"

    def test_status_endpoint(self):
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert isinstance(data["duckdb_tables"], list)
        assert "matches" in data["duckdb_tables"]
        assert "teams_clustered" in data["duckdb_tables"]
        assert isinstance(data["cached_models"], list)
        assert any("logistic_regression" in m for m in data["cached_models"])
        assert "rag_status" in data


# ===========================================================================
# 2. PRUEBAS DE DATOS DEPORTIVOS
# ===========================================================================

class TestSportsEndpoints:
    """Verificación de tablas de clasificación, perfiles de clubes y comparador."""

    def test_get_standings_default(self):
        response = client.get("/api/v1/standings?season=2024")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["season"] == 2024
        assert data["total_teams"] == 20
        assert len(data["tabla"]) == 20
        # Verificar ordenamiento por puntos
        assert data["tabla"][0]["puntos"] >= data["tabla"][1]["puntos"]
        # Verificar presencia de etiqueta de clúster actualizada
        assert data["tabla"][0]["etiqueta_cluster"] in [
            "Alto Rendimiento / Contendientes Europeos",
            "Mitad de Tabla Consolidada",
            "Zona de Descenso / Lucha por Permanencia"
        ]

    def test_get_standings_validation_error(self):
        # Temporada fuera de rango permitido (2022 a 2024)
        response = client.get("/api/v1/standings?season=1999")
        assert response.status_code == 422

    def test_list_teams(self):
        response = client.get("/api/v1/teams")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_teams"] == 24
        names = [t["team_name"] for t in data["teams"]]
        assert "Arsenal" in names
        assert "Manchester City" in names
        assert "Liverpool" in names

    def test_get_team_profile_success(self):
        response = client.get("/api/v1/teams/Arsenal/profile")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["team_name"] == "Arsenal"
        assert "ultimo_perfil" in data
        assert data["ultimo_perfil"]["puntos"] > 0
        assert "etiqueta_cluster" in data["ultimo_perfil"]

    def test_get_team_profile_not_found(self):
        response = client.get("/api/v1/teams/EquipoInexistenteFC/profile")
        assert response.status_code == 404

    def test_compare_teams_success(self):
        response = client.get("/api/v1/teams/compare?team_a=Arsenal&team_b=Manchester%20City&season=2024")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["team_a"]["team_name"] == "Arsenal"
        assert data["team_b"]["team_name"] == "Manchester City"
        assert data["season"] == 2024

    def test_compare_teams_invalid_team(self):
        response = client.get("/api/v1/teams/compare?team_a=InvalidClub&team_b=Arsenal&season=2024")
        assert response.status_code == 404


# ===========================================================================
# 3. PRUEBAS DE PREDICCIONES (MACHINE LEARNING)
# ===========================================================================

class TestPredictionsEndpoints:
    """Verificación de inferencia supervisada sobre modelos entrenados de Fase 2."""

    def test_prediction_champion_model(self):
        payload = {
            "home_team": "Liverpool",
            "away_team": "Arsenal",
            "model_type": "champion"
        }
        response = client.post("/api/v1/predictions/match", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["home_team"] == "Liverpool"
        assert data["away_team"] == "Arsenal"
        assert data["prediccion_estimada"] in ["Local", "Empate", "Visita"]
        probs = data["probabilidades"]
        # Probabilidades deben sumar 1.0 (con margen numérico)
        total_prob = probs["Local"] + probs["Empate"] + probs["Visita"]
        assert pytest.approx(total_prob, 0.01) == 1.0
        assert len(data["factores_clave"]) > 0
        assert "advertencia_metodologica" in data

    def test_prediction_automl_model(self):
        payload = {
            "home_team": "Manchester City",
            "away_team": "Chelsea",
            "model_type": "automl"
        }
        response = client.post("/api/v1/predictions/match", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "LRL1" in data["modelo_utilizado"] or "AutoML" in data["modelo_utilizado"]

    def test_prediction_same_teams_rejected(self):
        payload = {
            "home_team": "Arsenal",
            "away_team": "Arsenal",
            "model_type": "champion"
        }
        response = client.post("/api/v1/predictions/match", json=payload)
        assert response.status_code == 400
        assert "no pueden ser el mismo" in response.json()["detail"]

    def test_prediction_invalid_team(self):
        payload = {
            "home_team": "Fake Club",
            "away_team": "Liverpool",
            "model_type": "champion"
        }
        response = client.post("/api/v1/predictions/match", json=payload)
        assert response.status_code == 400

    def test_prediction_invalid_model_type(self):
        payload = {
            "home_team": "Arsenal",
            "away_team": "Liverpool",
            "model_type": "deep_neural_nonexistent"
        }
        response = client.post("/api/v1/predictions/match", json=payload)
        assert response.status_code == 400


# ===========================================================================
# 4. PRUEBAS DE ANALÍTICA Y VISUALIZACIONES
# ===========================================================================

class TestAnalyticsEndpoints:
    """Verificación de agregaciones del Lakehouse y generación de gráficos."""

    def test_analytics_overview(self):
        response = client.get("/api/v1/analytics/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_matches"] == 1140
        assert data["seasons"] == [2022, 2023, 2024]
        assert data["total_goals"] > 3000
        assert data["avg_goals_per_match"] > 2.5
        assert data["teams_count"] == 24
        pct_sum = data["home_wins_pct"] + data["draws_pct"] + data["away_wins_pct"]
        assert pytest.approx(pct_sum, 0.5) == 100.0

    def test_analytics_seasons(self):
        response = client.get("/api/v1/analytics/seasons")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["seasons"]) == 3
        for s in data["seasons"]:
            assert s["matches_count"] == 380
            assert s["champion_team"] != ""
            assert s["champion_points"] >= 80

    def test_analytics_chart_generation(self):
        payload = {
            "chart_type": "bar",
            "data": {"categories": ["2022", "2023", "2024"], "values": [1084, 1246, 1100]},
            "title": "Goles por Temporada",
            "x_label": "Temporada",
            "y_label": "Goles Totales"
        }
        response = client.post("/api/v1/analytics/chart", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["format"] == "png"
        assert len(data["image_base64"]) > 100

    def test_chart_with_authorized_preset(self):
        payload = {
            "chart_type": "bar",
            "preset": "goals_by_season",
            "title": "Volumen Histórico de Goles"
        }
        response = client.post("/api/v1/analytics/chart", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["image_base64"]) > 100

    def test_chart_rejects_arbitrary_sql_injection(self):
        # El endpoint prohíbe explícitamente recibir SQL arbitrario (extra="forbid")
        payload = {
            "chart_type": "bar",
            "sql_query": "SELECT * FROM matches; DROP TABLE matches;",
            "title": "Intento de Inyección"
        }
        response = client.post("/api/v1/analytics/chart", json=payload)
        assert response.status_code == 422

    def test_chart_rejects_unauthorized_preset(self):
        payload = {
            "chart_type": "bar",
            "preset": "malicious_preset_or_read_parquet",
            "title": "Preset No Autorizado"
        }
        response = client.post("/api/v1/analytics/chart", json=payload)
        assert response.status_code == 400
        assert "no autorizado" in response.json()["detail"].lower()


# ===========================================================================
# 5. PRUEBAS DE CHAT Y GESTIÓN DE SESIONES
# ===========================================================================

class TestChatAndSessionEndpoints:
    """Verificación del asistente conversacional y aislamiento de sesiones."""

    def test_chat_auto_session_generation(self):
        payload = {"message": "Muéstrame la tabla de posiciones de 2024"}
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "session_id" in data
        assert len(data["session_id"]) > 5
        assert "response" in data
        assert len(data["response"]) > 0

    def test_chat_session_memory_persistence(self):
        session_id = "test-session-12345"
        # Turno 1: Hablar de Manchester City
        res1 = client.post("/api/v1/chat", json={"message": "Háblame de Manchester City", "session_id": session_id})
        assert res1.status_code == 200

        # Turno 2: Pregunta contextual ("¿y cuál es su clúster?")
        res2 = client.post("/api/v1/chat", json={"message": "¿en qué clúster está?", "session_id": session_id})
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["session_id"] == session_id
        # La memoria debió registrar Manchester City
        assert "Manchester City" in data2["memory_entities"].get("teams", [])

    def test_chat_reset_session(self):
        session_id = "test-session-to-reset"
        # 1. Crear contexto
        client.post("/api/v1/chat", json={"message": "Analiza al Liverpool", "session_id": session_id})
        # 2. Reiniciar
        reset_res = client.post("/api/v1/chat/reset", json={"session_id": session_id})
        assert reset_res.status_code == 200
        assert reset_res.json()["status"] == "success"

        # 3. Verificar que la sesión quedó limpia
        res_after = client.post("/api/v1/chat", json={"message": "Hola", "session_id": session_id})
        assert res_after.status_code == 200
        assert "Liverpool" not in res_after.json()["memory_entities"].get("teams", [])


# ===========================================================================
# 6. PRUEBAS DE FRONTEND Y SEGURIDAD
# ===========================================================================

class TestFrontendAndSecurity:
    """Verificación de servicio de archivos estáticos y validación de entrada."""

    def test_serve_dashboard_index(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Premier League AI Analyst" in response.text

    def test_serve_static_css(self):
        response = client.get("/static/css/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")

    def test_serve_static_js(self):
        response = client.get("/static/js/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "")

    def test_chat_empty_message_validation(self):
        response = client.post("/api/v1/chat", json={"message": ""})
        assert response.status_code == 422

    def test_frontend_has_dompurify_and_tecnm(self):
        index_res = client.get("/")
        assert index_res.status_code == 200
        # Verificación de inclusión de DOMPurify
        assert "dompurify" in index_res.text.lower()
        # Verificación institucional
        assert "Instituto Tecnológico de La Laguna" in index_res.text
        assert "TecNM" in index_res.text


# ===========================================================================
# 7. PRUEBAS DE CONCURRENCIA Y GESTIÓN DE SESIONES
# ===========================================================================

class TestSessionConcurrency:
    """Verificación de aislamiento de sesiones, thread-safety y poda LRU."""

    def test_concurrent_sessions_thread_safety(self):
        from concurrent.futures import ThreadPoolExecutor
        from src.api.session_manager import SessionMemoryManager

        mgr = SessionMemoryManager(max_turns_per_session=5, max_active_sessions=100)
        
        def worker(session_num: int):
            sid = f"concurrent-session-{session_num}"
            s_id, mem = mgr.get_or_create(sid)
            mem.add_user_message(f"Mensaje del hilo {session_num}")
            return s_id, len(mem.history)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(25)]
            results = [f.result() for f in futures]

        assert len(results) == 25
        assert mgr.list_active_sessions() == 25
        for sid, hist_len in results:
            assert hist_len == 1

    def test_session_isolation(self):
        from src.api.session_manager import SessionMemoryManager

        mgr = SessionMemoryManager()
        _, mem_a = mgr.get_or_create("session-A")
        _, mem_b = mgr.get_or_create("session-B")

        mem_a.add_user_message("Consulta sobre Arsenal")
        mem_b.add_user_message("Consulta sobre Chelsea")

        assert "Arsenal" in mem_a.entities["mentioned_teams"]
        assert "Chelsea" not in mem_a.entities["mentioned_teams"]

        assert "Chelsea" in mem_b.entities["mentioned_teams"]
        assert "Arsenal" not in mem_b.entities["mentioned_teams"]

    def test_lru_pruning_under_capacity_limit(self):
        from src.api.session_manager import SessionMemoryManager

        # Instancia con límite bajo de 5 sesiones
        mgr = SessionMemoryManager(max_active_sessions=5)
        for i in range(10):
            mgr.get_or_create(f"session-{i}")

        # La capacidad debe mantenerse acotada
        assert mgr.list_active_sessions() <= 5
