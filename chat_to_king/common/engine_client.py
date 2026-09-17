from __future__ import annotations

import time
from enum import Enum

import httpx


class EngineState(str, Enum):
    LOADING = "loading"
    SERVING = "serving"


class EngineClient:
    def __init__(self, base_url: str, cache_s: float = 2.0, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._cache_s = cache_s
        self._timeout_s = timeout_s
        self._state = EngineState.LOADING
        self._checked_at = 0.0

    async def state(self) -> EngineState:
        now = time.monotonic()
        if now - self._checked_at < self._cache_s:
            return self._state
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                ok = (await client.get(f"{self.base_url}/health")).status_code == 200
        except Exception:
            ok = False
        self._state = EngineState.SERVING if ok else EngineState.LOADING
        self._checked_at = now
        return self._state
