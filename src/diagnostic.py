"""
Script de Diagnóstico Oficial de BALLDONTLIE EPL API V2.
Verifica:
1. Lectura de clave de API desde .env.
2. Autenticación y respuesta en endpoint gratuito (/epl/v2/teams).
3. Estructura JSON y validación Pydantic.
4. Sondeo controlado de permisos para /epl/v2/matches (detección de 403 vs 200).
5. Enmascaramiento estricto de credenciales en logs y consola.
"""

import os
import sys
import time
import requests
from pathlib import Path

# Ajustar PYTHONPATH al directorio del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BALLDONTLIE_API_KEY,
    BALLDONTLIE_BASE_URL,
    ALLOW_PAID_ENDPOINTS
)
from src.schemas.balldontlie_schemas import SoccerTeam, SoccerMatch


def mask_key(key: str) -> str:
    """Enmascara la clave para protegerla en reportes y salidas."""
    if not key or len(key) < 8:
        return "[NO CONFIGURADA O INVÁLIDA]"
    return f"{key[:4]}...{key[-4:]} (longitud: {len(key)})"


def run_diagnostic():
    print("=" * 75)
    print("DIAGNÓSTICO TÉCNICO DE CONEXIÓN: BALLDONTLIE EPL API V2")
    print("=" * 75)

    # 1. Comprobación de configuración local
    print("\n[PASO 1] Verificación de Variables de Entorno (.env):")
    if not BALLDONTLIE_API_KEY or BALLDONTLIE_API_KEY == "tu_api_key_aqui":
        print("  ❌ ERROR: Credencial no configurada en .env.")
        print("     Edita U5_Proyecto/.env y coloca tu API Key.")
        return False

    print("  ✔ Credencial configurada correctamente en .env.")
    print(f"  ✔ Base URL configurada: {BALLDONTLIE_BASE_URL}")
    print(f"  ✔ Salvaguarda ALLOW_PAID_ENDPOINTS: {ALLOW_PAID_ENDPOINTS}")

    session = requests.Session()
    session.headers.update({
        "Authorization": BALLDONTLIE_API_KEY,
        "Accept": "application/json",
        "User-Agent": "EPL-Diagnostic-Runner/1.0"
    })

    # 2. Prueba de Autenticación en Endpoint Gratuito (/epl/v2/teams)
    print("\n[PASO 2] Consulta de Bajo Costo en Endpoint Gratuito (/epl/v2/teams):")
    teams_url = f"{BALLDONTLIE_BASE_URL}/teams"
    start_t = time.time()
    try:
        resp = session.get(teams_url, timeout=15)
        if resp.status_code == 429:
            print("  ⏳ Límite de tasa (429) alcanzado por peticiones previas. Esperando 30s para reinicio de cuota...")
            time.sleep(30)
            resp = session.get(teams_url, timeout=15)

        elapsed = time.time() - start_t
        print(f"  ✔ Código de respuesta HTTP: {resp.status_code} ({resp.reason})")
        print(f"  ✔ Latencia de red: {elapsed:.2f} segundos")

        # Inspección de cabeceras relevantes
        rate_limit_rem = resp.headers.get("x-ratelimit-remaining", "N/A")
        server_header = resp.headers.get("server", "N/A")
        print(f"  ℹ Servidor / Proxy: {server_header} | Solicitudes restantes: {rate_limit_rem}")

        if resp.status_code == 401:
            print("  ❌ ERROR 401: La API Key configurada fue rechazada por BALLDONTLIE.")
            print("     Verifica que la clave copiada en .env sea exacta y esté activa.")
            return False
        elif resp.status_code != 200:
            print(f"  ❌ ERROR inesperado en /teams: {resp.status_code} - {resp.text}")
            return False

        # Validación de estructura JSON
        data_json = resp.json()
        if "data" not in data_json:
            print("  ❌ Estructura JSON inválida: falta el campo raíz 'data'.")
            return False

        items = data_json["data"]
        print(f"  ✔ Estructura JSON válida: Recibidos {len(items)} equipos en 'data'.")

        # Validación con Pydantic
        if items:
            sample_team = SoccerTeam(**items[0])
            print(f"  ✔ Validación Pydantic exitosa para equipo muestra: {sample_team.name} ({sample_team.abbreviation or 'N/A'}) - ID: {sample_team.id}")

    except Exception as e:
        print(f"  ❌ Error de conexión al consultar /teams: {e}")
        return False

    # Pausa de cortesía para respetar la tasa de 5 req/min
    print("\n  Pausa de cortesía de 3 segundos...")
    time.sleep(3)

    # 3. Sondeo controlado de permisos sobre /epl/v2/matches
    print("\n[PASO 3] Sondeo Controlado de Permisos para Partidos (/epl/v2/matches):")
    print("  * Nota: Se enviará exactamente 1 petición de sondeo (per_page=1) sin activar suscripciones ni pruebas.")
    matches_url = f"{BALLDONTLIE_BASE_URL}/matches"
    try:
        resp_m = session.get(matches_url, params={"per_page": 1}, timeout=15)
        print(f"  Código HTTP obtenido en /matches: {resp_m.status_code} ({resp_m.reason})")

        if resp_m.status_code == 200:
            print("  ✔ ACCESO CONFIRMADO: La cuenta cuenta con permisos activos para consultar /epl/v2/matches.")
            m_data = resp_m.json().get("data", [])
            if m_data:
                sample_m = SoccerMatch(**m_data[0])
                print(f"    Partido de muestra validado: {sample_m.home_team_id} vs {sample_m.away_team_id} (Temporada {sample_m.season})")
            return "200_AUTHORIZED"

        elif resp_m.status_code in (401, 403):
            print(f"  ⛔ ACCESO RESTRINGIDO (HTTP {resp_m.status_code} {resp_m.reason}):")
            print("     El endpoint de partidos (/epl/v2/matches) está restringido para el plan actual de esta cuenta.")
            print("     La API de BALLDONTLIE retorna 401/403 cuando la API Key pertenece al plan Free y no a ALL-STAR/GOAT.")
            print("     [POLÍTICA APLICADA]: La descarga de partidos se detiene formalmente sin inventar datos.")
            return "RESTRICTED_TIER"

        elif resp_m.status_code == 429:
            print("  ⚠ LÍMITE DE TASA (HTTP 429 Too Many Requests): Se ha alcanzado el límite de 5 req/min del plan Free.")
            return "429_RATE_LIMIT"

        else:
            print(f"  ❌ Respuesta no esperada: {resp_m.status_code} - {resp_m.text}")
            return False

    except Exception as e:
        print(f"  ❌ Error de conexión al verificar /matches: {e}")
        return False


if __name__ == "__main__":
    resultado = run_diagnostic()
    print("\n" + "=" * 75)
    print(f"RESULTADO FINAL DEL DIAGNÓSTICO: {resultado}")
    print("=" * 75)
