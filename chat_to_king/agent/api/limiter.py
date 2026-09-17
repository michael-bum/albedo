from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from agent.db import ApiKey

STATUS_HEADER = "x-albedo-status"


def _error(status: int, message: str, code: str, headers: dict[str, str] | None = None):
    body = {"error": {"message": message, "type": code, "code": code}}
    return JSONResponse(body, status_code=status, headers=headers or {})


class Limiter:
    def __init__(self, global_parallel: int) -> None:
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._parallel: Counter[str] = Counter()
        self._global = 0
        self._global_max = global_parallel

    def acquire(self, key: ApiKey, now: float) -> tuple[str, int] | None:
        window = self._windows[key.key_hash]
        while window and window[0] <= now - 60.0:
            window.popleft()
        if len(window) >= key.rpm:
            return "rpm", max(1, int(window[0] + 60.0 - now) + 1)
        if self._parallel[key.key_hash] >= key.parallel:
            return "parallel", 0
        if self._global_max and self._global >= self._global_max:
            return "busy", 0
        window.append(now)
        self._parallel[key.key_hash] += 1
        self._global += 1
        return None

    def release(self, key_hash: str) -> None:
        if self._parallel[key_hash] > 0:
            self._parallel[key_hash] -= 1
        if self._global > 0:
            self._global -= 1
