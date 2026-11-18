"""
Cliente oficial y resiliente para BALLDONTLIE EPL API V2.
Implementa control de tasa (rate limiting), paginación por cursor, salvaguarda de endpoints
de pago, reintentos con backoff, y persistencia de respuestas JSON crudas.
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import requests

from src.config import (
    BALLDONTLIE_API_KEY,
    BALLDONTLIE_BASE_URL,
    REQUEST_INTERVAL_SECONDS,
    ALLOW_PAID_ENDPOINTS,
    RAW_DATA_DIR
)
from src.schemas.balldontlie_schemas import (
    SoccerTeam,
    SoccerStanding,
    SoccerPlayer,
    SoccerMatch,
    IngestionReport
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BALLDONTLIEClient")

PAID_ENDPOINTS = {
    "/matches",
    "/match_events",
    "/match_lineups",
    "/team_match_stats",
    "/player_match_stats",
    "/odds",
    "/match_shots"
}


class BallDontLieAPIError(Exception):
    """Excepción para errores de la API de BallDontLie."""
    pass


class PaidEndpointRestrictedError(BallDontLieAPIError):
    """Excepción cuando se intenta consultar un endpoint de pago sin autorización explícita."""
    pass


class BALLDONTLIEClient:
    """Cliente para la API V2 de Premier League de BALLDONTLIE."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_interval: Optional[float] = None,
        allow_paid_endpoints: Optional[bool] = None
    ):
        self.api_key = api_key if api_key is not None else BALLDONTLIE_API_KEY
        self.base_url = (base_url or BALLDONTLIE_BASE_URL).rstrip("/")
        self.request_interval = request_interval if request_interval is not None else REQUEST_INTERVAL_SECONDS
        self.allow_paid_endpoints = allow_paid_endpoints if allow_paid_endpoints is not None else ALLOW_PAID_ENDPOINTS
        self.last_request_time: float = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EPL-Analytics-Agent/1.0",
            "Accept": "application/json"
        })
        if self.api_key:
            self.session.headers.update({"Authorization": self.api_key})

    def _throttle(self) -> None:
        """Aplica control de tasa estricto para evitar bloqueos por cuota (Rate Limit)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            logger.debug(f"Rate Limiter: esperando {sleep_time:.2f}s antes de la siguiente petición...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Dict[str, Any]:
        """Realiza una petición HTTP GET con manejo de errores, reintentos y control de pago."""
        clean_endpoint = "/" + endpoint.lstrip("/")
        
        # Salvaguarda de endpoints de pago
        if clean_endpoint in PAID_ENDPOINTS and not self.allow_paid_endpoints:
            raise PaidEndpointRestrictedError(
                f"Acceso bloqueado por salvaguarda: El endpoint '{clean_endpoint}' requiere confirmación "
                "de acceso a plan ALL-STAR / GOAT o prueba activa. No se realizarán llamadas a endpoints de pago "
                "hasta confirmación explícita."
            )

        url = f"{self.base_url}{clean_endpoint}"
        retries = 0
        backoff = 2.0

        while retries <= max_retries:
            self._throttle()
            try:
                logger.info(f"GET {url} (params={params})")
                response = self.session.get(url, params=params, timeout=20)
                
                # Manejo de estados HTTP
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    if clean_endpoint in PAID_ENDPOINTS:
                        raise BallDontLieAPIError(
                            f"Error 401 Unauthorized en endpoint de pago '{clean_endpoint}': "
                            "La API Key pertenece al plan Free y no tiene suscripción activa para acceder a este recurso "
                            "(requiere ALL-STAR / GOAT o prueba activa). Se detiene la ingesta conforme a la política de integridad."
                        )
                    else:
                        raise BallDontLieAPIError(
                            "Error 401 Unauthorized: La API Key de BALLDONTLIE no es válida o está ausente. "
                            "Verifica BALLDONTLIE_API_KEY en tu archivo .env."
                        )
                elif response.status_code == 403:
                    raise BallDontLieAPIError(
                        f"Error 403 Forbidden en '{clean_endpoint}': El plan actual de la API Key no tiene "
                        "permisos para este endpoint. Se detiene la ingesta para reportar la limitación."
                    )
                elif response.status_code == 429:
                    retries += 1
                    wait_time = backoff * (2 ** retries)
                    logger.warning(f"Error 429 Too Many Requests. Reintentando en {wait_time}s (Intento {retries}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    response.raise_for_status()
            except requests.RequestException as e:
                retries += 1
                if retries > max_retries:
                    raise BallDontLieAPIError(f"Fallo de conexión tras {max_retries} reintentos: {e}")
                wait_time = backoff * retries
                logger.warning(f"Error de red: {e}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)

        raise BallDontLieAPIError(f"No fue posible completar la petición a {url} tras {max_retries} intentos.")

    def _save_raw_json(self, name: str, data: Any) -> str:
        """Persiste la respuesta cruda en JSON dentro de data/raw/ con marca de tiempo."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = RAW_DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Datos crudos guardados en: {filepath}")
        return str(filepath)

    def fetch_teams(self) -> Tuple[List[SoccerTeam], IngestionReport]:
        """Obtiene el directorio de equipos de Premier League desde /epl/v2/teams."""
        logger.info("Iniciando descarga de equipos (endpoint Free: /epl/v2/teams)...")
        raw_response = self._request("/teams")
        self._save_raw_json("teams", raw_response)

        raw_items = raw_response.get("data", [])
        valid_teams: List[SoccerTeam] = []
        seen_ids = set()
        duplicates = 0
        invalid = 0

        for item in raw_items:
            try:
                team = SoccerTeam(**item)
                if team.id in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(team.id)
                valid_teams.append(team)
            except Exception as e:
                logger.error(f"Error validando equipo {item.get('id')}: {e}")
                invalid += 1

        report = IngestionReport(
            endpoint="/epl/v2/teams",
            total_registros_recibidos=len(raw_items),
            registros_validos=len(valid_teams),
            registros_invalidos=invalid,
            duplicados_descartados=duplicates,
            ids_unicos=list(seen_ids)
        )
        return valid_teams, report

    def fetch_standings(self, season: Optional[int] = None) -> Tuple[List[SoccerStanding], IngestionReport]:
        """
        Obtiene la tabla de clasificación histórica o vigente desde /epl/v2/standings.
        Soporta paginación por cursores.
        """
        logger.info(f"Iniciando descarga de clasificaciones (endpoint Free: /epl/v2/standings, season={season})...")
        params: Dict[str, Any] = {}
        if season is not None:
            params["season"] = season

        all_raw_items: List[Dict[str, Any]] = []
        cursor: Optional[int] = None

        while True:
            if cursor:
                params["cursor"] = cursor
            
            resp = self._request("/standings", params=params)
            items = resp.get("data", [])
            all_raw_items.extend(items)

            meta = resp.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        self._save_raw_json(f"standings_{season or 'all'}", {"data": all_raw_items})

        valid_standings: List[SoccerStanding] = []
        seen_keys = set()
        duplicates = 0
        invalid = 0
        error_details = []

        for item in all_raw_items:
            try:
                standing = SoccerStanding(**item)
                key = (standing.season, standing.team.id)
                if key in seen_keys:
                    duplicates += 1
                    continue
                seen_keys.add(key)
                valid_standings.append(standing)
            except Exception as e:
                msg = f"Error validando clasificación (season={item.get('season')}, team={item.get('team', {}).get('id')}): {e}"
                logger.error(msg)
                error_details.append(msg)
                invalid += 1

        report = IngestionReport(
            endpoint="/epl/v2/standings",
            total_registros_recibidos=len(all_raw_items),
            registros_validos=len(valid_standings),
            registros_invalidos=invalid,
            duplicados_descartados=duplicates,
            ids_unicos=[s.team.id for s in valid_standings],
            detalles_error=error_details
        )
        return valid_standings, report

    def fetch_matches(self, season: Optional[int] = None) -> Tuple[List[SoccerMatch], IngestionReport]:
        """
        Descarga de partidos oficiales desde /epl/v2/matches.
        NOTA: Requiere que ALLOW_PAID_ENDPOINTS=True y que la cuenta cuente con acceso activo.
        """
        logger.info(f"Intentando descarga de partidos (/epl/v2/matches, season={season})...")
        params: Dict[str, Any] = {}
        if season is not None:
            params["season"] = season

        all_raw_items: List[Dict[str, Any]] = []
        cursor: Optional[int] = None

        while True:
            if cursor:
                params["cursor"] = cursor

            resp = self._request("/matches", params=params)
            items = resp.get("data", [])
            all_raw_items.extend(items)

            meta = resp.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break

        self._save_raw_json(f"matches_{season or 'all'}", {"data": all_raw_items})

        valid_matches: List[SoccerMatch] = []
        seen_ids = set()
        duplicates = 0
        invalid = 0
        error_details = []

        for item in all_raw_items:
            try:
                match = SoccerMatch(**item)
                if match.id in seen_ids:
                    duplicates += 1
                    continue
                seen_ids.add(match.id)
                valid_matches.append(match)
            except Exception as e:
                msg = f"Error validando partido {item.get('id')}: {e}"
                logger.error(msg)
                error_details.append(msg)
                invalid += 1

        report = IngestionReport(
            endpoint="/epl/v2/matches",
            total_registros_recibidos=len(all_raw_items),
            registros_validos=len(valid_matches),
            registros_invalidos=invalid,
            duplicados_descartados=duplicates,
            ids_unicos=list(seen_ids),
            detalles_error=error_details
        )
        return valid_matches, report
