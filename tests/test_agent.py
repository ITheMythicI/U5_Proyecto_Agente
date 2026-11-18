"""
Suite Completa de Pruebas Unitarias e Integración para el Agente Deportivo (Fase 3).
Verifica:
1. Configuración y detección de Gemini.
2. Memoria conversacional y extracción de entidades.
3. Herramientas analíticas (DuckDB, Modelos supervisados, K-Means, generación de gráficos).
4. Seguridad de consultas SQL (bloqueo de DDL/DML).
5. Sistema RAG (fragmentación, indexación semántica y recuperación por similitud).
6. Entorno de ejecución controlada (AST, librerías permitidas y bloqueo de código malicioso).
7. Interacción end-to-end del agente.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Configurar path del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.config import config, AgentConfig
from src.agent.memory import AgentMemory
from src.agent.tools import (
    query_duckdb,
    predict_match,
    get_team_profile,
    get_standings,
    generate_chart,
    resolve_team_name
)
from src.agent.rag import RAGPipeline, DocumentProcessor, DenseVectorStore
from src.agent.code_executor import ControlledCodeExecutor
from src.agent.agent import SportsAnalystAgent


class TestAgentConfiguration(unittest.TestCase):
    """Pruebas para el módulo de configuración."""

    def test_paths_exist(self):
        self.assertTrue(config.lakehouse_db_path.exists(), "La base de datos DuckDB debe existir")
        self.assertTrue(config.models_dir.exists(), "El directorio models/ debe existir")
        self.assertTrue(config.knowledge_dir.exists(), "El directorio knowledge/ debe existir")

    def test_model_name_configurable(self):
        custom_cfg = AgentConfig(gemini_model_name="gemini-1.5-pro")
        self.assertEqual(custom_cfg.gemini_model_name, "gemini-1.5-pro")


class TestAgentMemory(unittest.TestCase):
    """Pruebas para la memoria conversacional y extracción de entidades."""

    def setUp(self):
        self.mem = AgentMemory(max_turns=3)

    def test_entity_extraction(self):
        self.mem.add_user_message("¿Cuál fue el rendimiento de Arsenal y Manchester City en la temporada 2023?")
        entities = self.mem.entities
        self.assertIn("Arsenal", entities["mentioned_teams"])
        self.assertIn("Manchester City", entities["mentioned_teams"])
        self.assertIn(2023, entities["mentioned_seasons"])

    def test_history_trimming(self):
        for i in range(10):
            self.mem.add_user_message(f"Consulta {i}")
            self.mem.add_model_message(f"Respuesta {i}")
        # Máximo de turnos retenidos
        self.assertLessEqual(len(self.mem.history), 6)

    def test_context_summary(self):
        self.mem.add_user_message("Pronostica Chelsea vs Liverpool en 2024")
        summary = self.mem.get_context_summary()
        self.assertIn("Chelsea", summary)
        self.assertIn("Liverpool", summary)


class TestAgentTools(unittest.TestCase):
    """Pruebas para las herramientas analíticas."""

    def test_team_name_resolution(self):
        self.assertEqual(resolve_team_name("Man City"), "Manchester City")
        self.assertEqual(resolve_team_name("spurs"), "Tottenham Hotspur")
        self.assertEqual(resolve_team_name("wolves"), "Wolverhampton Wanderers")
        self.assertIsNone(resolve_team_name("Real Madrid"))

    def test_duckdb_read_success(self):
        res = query_duckdb("SELECT count(*) as total FROM matches")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["rows"][0]["total"], 1140)

    def test_duckdb_security_blocking(self):
        # Intentos de modificación destructiva
        bad_queries = [
            "DROP TABLE matches",
            "DELETE FROM matches WHERE season = 2022",
            "INSERT INTO teams VALUES (99, 'Fake FC')",
            "UPDATE matches SET home_score = 99",
            "ALTER TABLE matches ADD COLUMN hacked INT"
        ]
        for q in bad_queries:
            res = query_duckdb(q)
            self.assertEqual(res["status"], "error", f"La consulta '{q}' debió ser rechazada")
            self.assertIn("restringid", res["message"].lower())

    def test_predict_match_champion(self):
        res = predict_match("Arsenal", "Chelsea", model_type="champion")
        self.assertEqual(res["status"], "success")
        self.assertIn("probabilidades", res)
        probs = res["probabilidades"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        self.assertIn(res["prediccion_estimada"], ["Local", "Empate", "Visita"])
        self.assertIn("advertencia_metodologica", res)
        self.assertTrue(len(res["factores_clave"]) > 0)

    def test_predict_match_automl(self):
        res = predict_match("Manchester City", "Liverpool", model_type="automl")
        self.assertEqual(res["status"], "success")
        self.assertIn("LRL1", res["modelo_utilizado"])

    def test_predict_match_validation_errors(self):
        # Mismo equipo
        res1 = predict_match("Arsenal", "Arsenal")
        self.assertEqual(res1["status"], "error")
        # Equipo desconocido
        res2 = predict_match("Barcelona FC", "Arsenal")
        self.assertEqual(res2["status"], "error")

    def test_team_profile(self):
        res = get_team_profile("Manchester City")
        self.assertEqual(res["status"], "success")
        ultimo = res["ultimo_perfil"]
        self.assertEqual(ultimo["team_name"], "Manchester City")
        self.assertIn("cluster", ultimo)
        self.assertIn("etiqueta_cluster", ultimo)

    def test_standings(self):
        res = get_standings(season=2024)
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["tabla"]), 20)

    def test_generate_chart(self):
        res = generate_chart(
            chart_type="bar",
            sql_query="SELECT team_name, puntos FROM teams_clustered WHERE season = 2024 ORDER BY puntos DESC LIMIT 5",
            title="Top 5 Premier League 2024",
            output_filename="test_top5_chart.png"
        )
        self.assertEqual(res["status"], "success")
        out_file = Path(res["output_path"])
        self.assertTrue(out_file.exists())


class TestRAGSystem(unittest.TestCase):
    """Pruebas para la fragmentación, indexación y recuperación RAG."""

    def setUp(self):
        self.rag = RAGPipeline()
        self.rag.initialize()

    def test_chunks_generated(self):
        chunks = self.rag.processor.load_and_chunk_all()
        self.assertGreater(len(chunks), 4, "Debe haber múltiples fragmentos indexados")

    def test_dense_vector_store(self):
        self.assertIsNotNone(self.rag.vector_store.vectors)
        self.assertEqual(len(self.rag.vector_store.vectors), len(self.rag.vector_store.chunks))

    def test_semantic_retrieval_rules(self):
        results = self.rag.retrieve("¿Cuántos puntos se otorgan por victoria y cómo funciona el descenso?", top_k=3)
        self.assertTrue(len(results) > 0)
        retrieved_sources = [r["source_file"] for r in results]
        self.assertIn("epl_rules_and_structure.md", retrieved_sources)
        self.assertGreater(results[0]["similarity_score"], 0.2)

    def test_semantic_retrieval_architecture(self):
        results = self.rag.retrieve("¿Cuál es el modelo campeón supervisado y qué Macro F1 obtuvo?", top_k=2)
        self.assertTrue(len(results) > 0)
        combined_text = " ".join([r["text"] for r in results])
        self.assertTrue("Regresión Logística" in combined_text or "0.4581" in combined_text)


class TestControlledCodeExecutor(unittest.TestCase):
    """Pruebas para el entorno de ejecución controlada del Agente Programador."""

    def setUp(self):
        self.executor = ControlledCodeExecutor(timeout_seconds=5)

    def test_default_disabled(self):
        """Verifica que la ejecución de código esté deshabilitada por defecto."""
        res = self.executor.execute_python("print('hola')")
        self.assertEqual(res["status"], "disabled")
        self.assertIn("deshabilitada por defecto", res["message"])

    def test_blocked_os_import_ast(self):
        errors = self.executor.validate_code("import os; os.system('ls')")
        self.assertTrue(any("bloqueado" in err for err in errors))

    def test_blocked_subprocess_ast(self):
        errors = self.executor.validate_code("import subprocess; subprocess.run(['echo', 'hacked'])")
        self.assertTrue(any("bloqueado" in err for err in errors))

    def test_blocked_open_builtin_ast(self):
        errors = self.executor.validate_code("with open('/tmp/test.txt', 'w') as f: f.write('hello')")
        self.assertTrue(any("open" in err for err in errors))

    def test_blocked_dunder_globals_ast(self):
        errors = self.executor.validate_code("x = ().__class__.__bases__[0].__subclasses__()")
        self.assertTrue(any("restringido" in err for err in errors))


class TestSportsAnalystAgent(unittest.TestCase):
    """Pruebas unitarias de interacción con el agente en modo determinista local."""

    def setUp(self):
        # Modo local para pruebas unitarias deterministas sin consumo de cuota de API
        local_cfg = AgentConfig(gemini_api_key="")
        self.agent = SportsAnalystAgent(agent_config=local_cfg)

    def test_prediction_query(self):
        res = self.agent.ask("¿Quién tiene mayor probabilidad de ganar entre Arsenal y Chelsea?")
        self.assertEqual(res["status"], "success")
        self.assertIn("Pronóstico Estadístico", res["response"])
        self.assertIn("Arsenal", res["response"])
        self.assertIn("Chelsea", res["response"])
        self.assertIn("prediccion_estimada", str(res["tool_calls"]))

    def test_team_cluster_query(self):
        res = self.agent.ask("¿A qué clúster pertenece Manchester City y cuál es su perfil?")
        self.assertEqual(res["status"], "success")
        self.assertIn("Perfil Taxonómico", res["response"])
        self.assertIn("Manchester City", res["response"])

    def test_standings_query(self):
        res = self.agent.ask("Muéstrame la tabla de posiciones de la Premier League 2024")
        self.assertEqual(res["status"], "success")
        self.assertIn("Tabla General de Posiciones", res["response"])

    def test_rag_knowledge_query(self):
        res = self.agent.ask("Explícame las reglas de clasificación a Champions League y descenso en la Premier League")
        self.assertEqual(res["status"], "success")
        self.assertIn("Información Recuperada mediante RAG", res["response"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
