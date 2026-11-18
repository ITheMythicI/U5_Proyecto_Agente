"""
Suite de Pruebas de Integración en Vivo con Google Gemini API (Fase 3).
Requiere conexión activa a internet y una GEMINI_API_KEY válida en .env.

Verifica:
1. Autenticación y generación de contenido en vivo con Gemini.
2. Function Calling real en chat con llamadas a DuckDB (get_standings y predict_match).
3. Generación de embeddings neuronales con gemini-embedding-001 (dimensión 3072).
4. Manejo de excepciones ante API Key inválida o modelo inexistente.
"""

import sys
import unittest
import os
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from src.agent.config import config
from src.agent.tools import get_standings, predict_match, get_team_profile
from src.agent.rag import DenseVectorStore, DocumentChunk


class TestLiveGeminiIntegration(unittest.TestCase):
    """Pruebas en vivo contra Google Gemini API."""

    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or len(api_key) < 20 or "tu_" in api_key.lower():
            raise unittest.SkipTest("GEMINI_API_KEY no está configurada con una credencial válida en .env.")
        cls.client = genai.Client(api_key=api_key)
        cls.test_model = "gemini-3.5-flash-lite"

    def test_01_live_authentication_and_generation(self):
        """Verifica la conectividad y respuesta real del modelo."""
        response = self.client.models.generate_content(
            model=self.test_model,
            contents="Responde exactamente con la palabra: CONEXION_EXITOSA"
        )
        self.assertIsNotNone(response.text)
        self.assertIn("CONEXION_EXITOSA", response.text.strip().upper())

    def test_02_live_function_calling_standings(self):
        """Verifica que Gemini identifique y ejecute la herramienta get_standings."""
        cfg = types.GenerateContentConfig(tools=[get_standings], temperature=0.1)
        chat = self.client.chats.create(model=self.test_model, config=cfg)
        response = chat.send_message("Muéstrame la tabla de posiciones de la Premier League 2024")

        # Verificar en el historial que se ejecutó una FunctionCall a get_standings
        history = chat.get_history()
        called_tools = []
        for msg in history:
            for part in msg.parts:
                if hasattr(part, "function_call") and part.function_call:
                    called_tools.append(part.function_call.name)

        self.assertIn("get_standings", called_tools, "Gemini debió invocar get_standings")
        self.assertIn("Liverpool", response.text)
        self.assertIn("Arsenal", response.text)

    def test_03_live_function_calling_prediction(self):
        """Verifica que Gemini invoque predict_match para un partido entre Liverpool y Arsenal."""
        cfg = types.GenerateContentConfig(tools=[predict_match], temperature=0.1)
        chat = self.client.chats.create(model=self.test_model, config=cfg)
        response = chat.send_message("¿Qué probabilidades tiene Liverpool contra Arsenal?")

        history = chat.get_history()
        called_tools = []
        for msg in history:
            for part in msg.parts:
                if hasattr(part, "function_call") and part.function_call:
                    called_tools.append(part.function_call.name)

        self.assertIn("predict_match", called_tools, "Gemini debió invocar predict_match")
        self.assertTrue(any(w in response.text.lower() for w in ["probabilidad", "%", "local", "empate", "visita"]))

    def test_04_live_gemini_neural_embeddings(self):
        """Verifica la generación de embeddings neuronales con gemini-embedding-001 (dimensión 3072)."""
        store = DenseVectorStore()
        chunks = [
            DocumentChunk(chunk_id="c1", source_file="doc1.md", section_title="Reglas", text="Premier League tiene 20 clubes."),
            DocumentChunk(chunk_id="c2", source_file="doc2.md", section_title="Campeón", text="Manchester City fue campeón.")
        ]
        store.build_index(chunks, forced_mode="gemini")
        self.assertEqual(store.active_mode, "gemini_neural_3072d")
        self.assertEqual(store.vectors.shape, (2, 3072))

        # Consulta semántica
        res = store.query("¿Quién se coronó en la liga?", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["chunk_id"], "c2")
        self.assertGreater(res[0]["similarity_score"], 0.6)

    def test_05_api_error_handling_invalid_key(self):
        """Verifica que una clave inválida dispare ClientError adecuadamente."""
        bad_client = genai.Client(api_key="AIzaSy_CLAVE_FALSA_PARA_PRUEBAS_12345678")
        with self.assertRaises(ClientError):
            bad_client.models.generate_content(
                model=self.test_model,
                contents="test"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
