"""
Gestor de sesiones concurrentes de memoria conversacional para la API REST.
Permite mantener contextos aislados por usuario y reiniciar sesiones a petición.
Implementación segura ante hilos concurrentes mediante threading.RLock.
"""

from typing import Dict, Tuple, Optional
import uuid
import time
import threading
from ..agent.memory import AgentMemory


class SessionMemoryManager:
    """Administra las instancias de AgentMemory indexadas por session_id de forma thread-safe."""

    def __init__(self, max_turns_per_session: int = 10, max_active_sessions: int = 500):
        self.max_turns = max_turns_per_session
        self.max_sessions = max_active_sessions
        self._sessions: Dict[str, Tuple[AgentMemory, float]] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: Optional[str] = None) -> Tuple[str, AgentMemory]:
        """Recupera la memoria asociada a un session_id o genera una nueva sesión."""
        now = time.time()
        with self._lock:
            # Si se excede el límite de sesiones activas, purgar las más antiguas (LRU)
            if len(self._sessions) >= self.max_sessions:
                sorted_sessions = sorted(self._sessions.items(), key=lambda item: item[1][1])
                for old_id, _ in sorted_sessions[:50]:
                    self._sessions.pop(old_id, None)

            if not session_id or session_id not in self._sessions:
                new_id = session_id or str(uuid.uuid4())
                new_mem = AgentMemory(max_turns=self.max_turns)
                self._sessions[new_id] = (new_mem, now)
                return new_id, new_mem

            mem, _ = self._sessions[session_id]
            self._sessions[session_id] = (mem, now)
            return session_id, mem

    def reset_session(self, session_id: str) -> bool:
        """Reinicia la memoria de una sesión existente sin afectar a otras."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id][0].clear()
                self._sessions[session_id] = (self._sessions[session_id][0], time.time())
                return True
            return False

    def list_active_sessions(self) -> int:
        """Retorna la cantidad de sesiones activas en memoria."""
        with self._lock:
            return len(self._sessions)

    def delete_session(self, session_id: str) -> bool:
        """Elimina por completo una sesión de la memoria."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


# Instancia singleton del gestor de sesiones
session_manager = SessionMemoryManager()
