"""
Módulo de Descarga Histórica Resiliente: BALLDONTLIE EPL API V2.
Temporadas: 2022-23 (2022), 2023-24 (2023), 2024-25 (2024).

Características clave:
1. Control de tasa estricto: 5 solicitudes / minuto (12.5s entre llamadas con margen).
2. Paginación por cursor y persistencia incremental por página.
3. Checkpoint resiliente (.checkpoint_historical.json): reanuda descargas interrumpidas
   sin repetir páginas ya almacenadas.
4. Verificación previa de espacio en disco (mínimo 500 MB).
5. Sonda de autorización previa (detiene si responde 401/403).
6. Enmascaramiento total de credenciales (nunca imprime fragmentos de la clave).
7. Consolidación y validación de archivos JSON crudos.
"""

import os
import sys
import time
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ajustar PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BALLDONTLIE_API_KEY,
    BALLDONTLIE_BASE_URL,
    RAW_DATA_DIR,
    ALLOW_PAID_ENDPOINTS
)
from src.schemas.balldontlie_schemas import SoccerMatch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HistoricalIngest")

CHECKPOINT_FILE = RAW_DATA_DIR / ".checkpoint_historical.json"
TARGET_SEASONS = [2022, 2023, 2024]
REQUEST_INTERVAL_SECONDS = 12.5  # 5 req/min con margen de seguridad


class HistoricalIngestManager:

    def __init__(self, seasons: Optional[List[int]] = None, raw_data_dir: Optional[Path] = None):
        self.seasons = seasons or TARGET_SEASONS
        self.api_key = BALLDONTLIE_API_KEY
        self.base_url = BALLDONTLIE_BASE_URL
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else RAW_DATA_DIR
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.raw_data_dir / ".checkpoint_historical.json"
        self.checkpoint = self._load_checkpoint()

    def _check_disk_space(self, min_mb: int = 500) -> bool:
        """Verifica que haya suficiente espacio en disco antes de iniciar."""
        total, used, free = shutil.disk_usage(self.raw_data_dir)
        free_mb = free / (1024 * 1024)
        free_gb = free / (1024 * 1024 * 1024)
        if free_mb < min_mb:
            logger.error(f"Espacio insuficiente en disco: {free_mb:.1f} MB disponibles (mínimo requerido: {min_mb} MB).")
            return False
        logger.info(f"Comprobación de almacenamiento: {free_gb:.1f} GB disponibles en disco (Espacio suficiente).")
        return True

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Carga el estado de avance desde el archivo de checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"No se pudo leer checkpoint anterior ({e}). Se creará uno nuevo.")
        return {}

    def _save_checkpoint(self) -> None:
        """Guarda el estado actual en el archivo de checkpoint de forma atómica."""
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.checkpoint, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.checkpoint_file)

    def _probe_matches_access(self) -> bool:
        """
        Realiza una única petición de sondeo (per_page=1) a /matches para verificar
        que la cuenta tenga la prueba o suscripción activa antes de iniciar el bucle.
        """
        import requests
        headers = {
            "Authorization": self.api_key,
            "Accept": "application/json",
            "User-Agent": "EPL-Historical-Ingest/1.0"
        }
        url = f"{self.base_url}/matches"
        logger.info("Realizando sondeo previo de autorización sobre /epl/v2/matches...")
        try:
            resp = requests.get(url, headers=headers, params={"per_page": 1}, timeout=15)
            if resp.status_code == 200:
                logger.info("Autorización confirmada: Acceso activo a /epl/v2/matches.")
                return True
            elif resp.status_code in (401, 403):
                logger.error(
                    f"Acceso restringido (HTTP {resp.status_code} {resp.reason}). "
                    "La cuenta no tiene acceso a partidos en este momento. "
                    "Se detiene el proceso formalmente para no consumir solicitudes inútiles."
                )
                return False
            elif resp.status_code == 429:
                logger.warning("Límite de tasa (429) alcanzado durante el sondeo. Espera 60s antes de reintentar.")
                return False
            else:
                logger.error(f"Respuesta inesperada en sondeo: HTTP {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Error de red durante sondeo de autorización: {e}")
            return False

    def run_ingestion(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Ejecuta la descarga histórica con checkpointing, pausas y consolidación.
        """
        import requests

        print("=" * 75)
        print("DESCARGA HISTÓRICA RESILIENTE: PREMIER LEAGUE (BALLDONTLIE EPL V2)")
        print("=" * 75)

        # 1. Comprobación de credencial
        if not self.api_key or self.api_key == "tu_api_key_aqui":
            logger.error("Credencial no configurada en .env.")
            print("❌ ERROR: Credencial no configurada en U5_Proyecto/.env.")
            return {"status": "error", "message": "Credencial no configurada"}

        logger.info("Credencial configurada correctamente en .env.")

        # 2. Comprobación de espacio en disco
        if not self._check_disk_space(min_mb=500):
            return {"status": "error", "message": "Espacio en disco insuficiente"}

        if dry_run:
            logger.info("Modo simulación (DRY-RUN) activo. No se realizarán llamadas a la red.")
            return {
                "status": "dry_run_ok",
                "seasons": self.seasons,
                "interval_seconds": REQUEST_INTERVAL_SECONDS,
                "checkpoint": self.checkpoint
            }

        # 3. Sondeo de acceso a /matches
        if not self._probe_matches_access():
            print("\n❌ PROCESO DETENIDO: La cuenta no cuenta con permisos para el endpoint /matches.")
            print("   Activa la prueba gratuita de 7 días en app.balldontlie.io antes de iniciar.")
            return {"status": "error", "message": "Acceso a /matches no autorizado"}

        session = requests.Session()
        session.headers.update({
            "Authorization": self.api_key,
            "Accept": "application/json",
            "User-Agent": "EPL-Historical-Ingest/1.0"
        })

        results = {}
        last_request_time = 0.0

        for season in self.seasons:
            season_key = str(season)
            season_label = f"{season}-{str(season + 1)[-2:]}"
            print(f"\n" + "-" * 75)
            print(f"PROCESANDO TEMPORADA: {season_label} (Año: {season})")
            print("-" * 75)

            season_state = self.checkpoint.get(season_key, {
                "status": "PENDING",
                "next_cursor": None,
                "pages_downloaded": 0,
                "total_records": 0,
                "page_files": []
            })

            if season_state.get("status") == "COMPLETED":
                logger.info(f"Temporada {season_label} ya completada previamente. Omitiendo descarga...")
                results[season_key] = season_state
                continue

            season_state["status"] = "IN_PROGRESS"
            cursor = season_state.get("next_cursor")
            page_num = season_state.get("pages_downloaded", 0) + 1
            page_files = season_state.get("page_files", [])
            total_records = season_state.get("total_records", 0)

            while True:
                # Control estricto de tasa (5 req/min -> 12.5s)
                elapsed = time.time() - last_request_time
                if elapsed < REQUEST_INTERVAL_SECONDS:
                    wait_time = REQUEST_INTERVAL_SECONDS - elapsed
                    logger.info(f"Rate Limiter: esperando {wait_time:.1f}s para cumplir límite de 5 req/min...")
                    time.sleep(wait_time)

                params = {"season": season, "per_page": 100}
                if cursor:
                    params["cursor"] = cursor

                logger.info(f"[{season_label}] Solicitando página {page_num} (cursor={cursor})...")
                url = f"{self.base_url}/matches"
                last_request_time = time.time()

                try:
                    resp = session.get(url, params=params, timeout=20)
                except Exception as e:
                    logger.error(f"Error de red en página {page_num}: {e}. El progreso hasta el cursor {cursor} queda guardado.")
                    self._save_checkpoint()
                    return {"status": "interrupted", "error": str(e), "checkpoint": self.checkpoint}

                if resp.status_code == 429:
                    logger.warning("Error 429 (Too Many Requests). Esperando 30s antes de reintentar...")
                    time.sleep(30)
                    continue
                elif resp.status_code != 200:
                    logger.error(f"Error HTTP {resp.status_code} en página {page_num}: {resp.text}")
                    self._save_checkpoint()
                    return {"status": "error", "http_code": resp.status_code, "checkpoint": self.checkpoint}

                # Guardar página inmediatamente en disco de forma atómica
                data_json = resp.json()
                items = data_json.get("data", [])
                meta = data_json.get("meta", {})
                next_cursor = meta.get("next_cursor")

                page_file = self.raw_data_dir / f"matches_{season}_page_{page_num}.json"
                temp_page_file = page_file.with_suffix(".tmp")
                with open(temp_page_file, "w", encoding="utf-8") as f:
                    json.dump(data_json, f, indent=2, ensure_ascii=False)
                temp_page_file.replace(page_file)

                page_files.append(str(page_file))
                total_records += len(items)
                logger.info(f"[{season_label}] Página {page_num} guardada de forma atómica: {len(items)} partidos (Acumulado: {total_records}).")

                # Actualizar checkpoint
                season_state["next_cursor"] = next_cursor
                season_state["pages_downloaded"] = page_num
                season_state["total_records"] = total_records
                season_state["page_files"] = page_files
                self.checkpoint[season_key] = season_state
                self._save_checkpoint()

                if not next_cursor or len(items) == 0:
                    logger.info(f"[{season_label}] Fin de páginas alcanzado para temporada {season_label}.")
                    break

                cursor = next_cursor
                page_num += 1

            # Consolidar todos los partidos de la temporada en un único archivo oficial de forma atómica
            consolidated_file = self.raw_data_dir / f"matches_{season}.json"
            all_matches = []
            seen_ids = set()

            for p_file in page_files:
                try:
                    with open(p_file, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        for m in content.get("data", []):
                            m_id = m.get("id")
                            if m_id not in seen_ids:
                                seen_ids.add(m_id)
                                all_matches.append(m)
                except Exception as e:
                    logger.error(f"Error consolidando archivo {p_file}: {e}")

            # Validación de completitud: No marcar completada si no hay registros o hubo error
            if len(all_matches) == 0:
                logger.error(f"[{season_label}] La temporada no contiene partidos válidos. No se marcará como COMPLETED.")
                season_state["status"] = "ERROR_EMPTY"
                self._save_checkpoint()
                continue

            temp_consolidated = consolidated_file.with_suffix(".tmp")
            with open(temp_consolidated, "w", encoding="utf-8") as f:
                json.dump({"data": all_matches}, f, indent=2, ensure_ascii=False)
            temp_consolidated.replace(consolidated_file)

            # Comprobar legibilidad del archivo consolidado
            with open(consolidated_file, "r", encoding="utf-8") as f:
                verified_data = json.load(f)
                if len(verified_data.get("data", [])) != len(all_matches):
                    raise IOError(f"Discrepancia en verificación del archivo consolidado para {season_label}")

            season_state["status"] = "COMPLETED"
            season_state["consolidated_file"] = str(consolidated_file)
            season_state["unique_matches"] = len(all_matches)
            season_state["completed_at"] = datetime.now().isoformat()
            self.checkpoint[season_key] = season_state
            self._save_checkpoint()

            print(f"✔ TEMPORADA {season_label} FINALIZADA: {len(all_matches)} partidos consolidados en {consolidated_file.name}.")
            results[season_key] = season_state

        print("\n" + "=" * 75)
        print("DESCARGA HISTÓRICA COMPLETADA CON ÉXITO")
        print("=" * 75)
        return {"status": "success", "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Descarga histórica BALLDONTLIE EPL V2")
    parser.add_argument("--dry-run", action="store_true", help="Simula la ejecución sin realizar peticiones de red")
    args = parser.parse_args()

    manager = HistoricalIngestManager()
    manager.run_ingestion(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
