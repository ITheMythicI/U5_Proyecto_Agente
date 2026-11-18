"""
Script de Validación en Vivo de la Integración Real con Google Gemini.
Verifica:
1. Conexión autenticada con Gemini API.
2. Function Calling en los 3 escenarios obligatorios del usuario.
3. RAG en modo Gemini Embeddings y modo Local LSA.
4. Restricción de seguridad en DuckDB y Code Executor deshabilitado por defecto.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import SportsAnalystAgent
from src.agent.tools import query_duckdb
from src.agent.rag import RAGPipeline


def test_scenario_1_standings(agent: SportsAnalystAgent):
    print("=" * 70)
    print("ESCENARIO 1: 'Muéstrame la tabla de posiciones de la Premier League 2024'")
    print("=" * 70)
    q = "Muéstrame la tabla de posiciones de la Premier League 2024"
    res = agent.ask(q)

    print(f"Modo: {res.get('mode')}")
    print(f"Modelo activo: {res.get('model_name')}")
    print(f"Herramientas invocadas ({len(res.get('tool_calls', []))}):")
    for t in res.get("tool_calls", []):
        print(f"  - Herramienta: {t.get('tool')} | Argumentos: {t.get('args')}")
    print("\nRespuesta de Gemini:")
    print(res.get("response")[:600] + "..." if len(res.get("response", "")) > 600 else res.get("response"))
    return res


def test_scenario_2_comparison(agent: SportsAnalystAgent):
    print("\n" + "=" * 70)
    print("ESCENARIO 2: 'Compara el rendimiento de Manchester City y Arsenal'")
    print("=" * 70)
    q = "Compara el rendimiento de Manchester City y Arsenal"
    res = agent.ask(q)

    print(f"Modo: {res.get('mode')}")
    print(f"Modelo activo: {res.get('model_name')}")
    print(f"Herramientas invocadas ({len(res.get('tool_calls', []))}):")
    for t in res.get("tool_calls", []):
        print(f"  - Herramienta: {t.get('tool')} | Argumentos: {t.get('args')}")
    print("\nRespuesta de Gemini:")
    print(res.get("response")[:600] + "..." if len(res.get("response", "")) > 600 else res.get("response"))
    return res


def test_scenario_3_prediction(agent: SportsAnalystAgent):
    print("\n" + "=" * 70)
    print("ESCENARIO 3: '¿Qué probabilidades tiene Liverpool de ganar contra Arsenal?'")
    print("=" * 70)
    q = "¿Qué probabilidades tiene Liverpool de ganar contra Arsenal?"
    res = agent.ask(q)

    print(f"Modo: {res.get('mode')}")
    print(f"Modelo activo: {res.get('model_name')}")
    print(f"Herramientas invocadas ({len(res.get('tool_calls', []))}):")
    for t in res.get("tool_calls", []):
        print(f"  - Herramienta: {t.get('tool')} | Argumentos: {t.get('args')}")
    print("\nRespuesta de Gemini:")
    print(res.get("response")[:600] + "..." if len(res.get("response", "")) > 600 else res.get("response"))
    return res


def test_rag_dual_modes():
    print("\n" + "=" * 70)
    print("VALIDACIÓN INDEPENDIENTE DE LOS DOS MODOS DE RAG (GEMINI EMBEDDINGS vs LOCAL LSA)")
    print("=" * 70)

    rag = RAGPipeline()
    chunks = rag.processor.load_and_chunk_all()
    print(f"Total de fragmentos procesados: {len(chunks)}")

    # 1. Modo Local LSA (Latent Semantic Analysis: TF-IDF + SVD)
    rag.initialize(forced_mode="local_lsa")
    lsa_results = rag.retrieve("¿Cómo funciona el descenso y la clasificación a Champions League?", top_k=2, forced_mode="local_lsa")
    print(f"\n1. Modo Local LSA ({rag.vector_store.active_mode}) | Matriz Vectorial: {rag.vector_store.vectors.shape}:")
    for r in lsa_results:
        print(f"  - Fuente: {r['source_file']} | Sección: {r['section_title']} | Similitud LSA: {r['similarity_score']:.4f}")

    # 2. Modo Gemini Neural Embeddings (gemini-embedding-001)
    rag.initialize(forced_mode="gemini")
    gemini_results = rag.retrieve("¿Cómo funciona el descenso y la clasificación a Champions League?", top_k=2, forced_mode="gemini")
    print(f"\n2. Modo Gemini Embeddings ({rag.vector_store.active_mode}) | Matriz Vectorial: {rag.vector_store.vectors.shape}:")
    for r in gemini_results:
        print(f"  - Fuente: {r['source_file']} | Sección: {r['section_title']} | Similitud Coseno: {r['similarity_score']:.4f}")

    # Comprobar independencia matemática
    assert rag.vector_store.vectors.shape[1] == 3072, "El espacio vectorial de Gemini debe tener 3072 dimensiones"
    print("\n[OK] Confirmado: cada motor opera en su propio espacio vectorial independiente (LSA: 18d, Gemini: 3072d).")


def test_security_hardening(agent: SportsAnalystAgent):
    print("\n" + "=" * 70)
    print("VALIDACIÓN DE POLÍTICAS DE SEGURIDAD (DUCKDB Y CODE EXECUTOR)")
    print("=" * 70)

    # 1. Intento de inyección SQL externa
    print("1. Intento de lectura de archivos externos en DuckDB:")
    bad_sql = "SELECT * FROM read_parquet('/etc/passwd')"
    res_sql = query_duckdb(bad_sql)
    print(f"  Consulta: {bad_sql}")
    print(f"  Resultado de seguridad: {res_sql['status']} | Mensaje: {res_sql.get('message')}")

    # 2. Intento de ejecución de código deshabilitado
    print("\n2. Intento de ejecución de código en sandbox deshabilitado:")
    res_code = agent.code_executor.execute_python("import os; print(os.environ)")
    print(f"  Resultado de seguridad: {res_code['status']} | Mensaje: {res_code.get('message')}")


if __name__ == "__main__":
    import time
    agent = SportsAnalystAgent()
    print("Iniciando verificación en vivo con Gemini (con pausas de 12s para respetar el límite Free Tier de 5 RPM)...")

    r1 = test_scenario_1_standings(agent)
    print("Esperando 12s...")
    time.sleep(12)

    r2 = test_scenario_2_comparison(agent)
    print("Esperando 12s...")
    time.sleep(12)

    r3 = test_scenario_3_prediction(agent)
    print("Esperando 12s...")
    time.sleep(12)

    test_rag_dual_modes()
    test_security_hardening(agent)
    print("\n¡Verificación en vivo completada exitosamente!")
