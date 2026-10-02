"""HTTP client for the repo-context service.

Lives beside judge_llm_client.py rather than in shared/ because it takes JudgeSettings and shared/
is deliberately config-free. Both the eval judge API and the pre-eval dispatcher construct one, so
grounding is resolved the same way on both paths.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import NamedTuple

import httpx
from loguru import logger

from albedo_config import JudgeSettings


class Grounding(NamedTuple):
    context: str | None
    exact_output: str | None
    exact_returncode: int | None
    state: str
    leading_output: str | None = None
    # the command split into exact parts and gaps to simulate (see stitch_parts)
    parts: list[dict] | None = None


async def stitch_parts(
    parts: list[dict], answer: Callable[[dict], Awaitable[tuple[str, int | None]]]
) -> tuple[str, int | None] | None:
    """A command's output from its parts, run as the shell runs them: each part after the first
    runs or not by the `&&` or `||` joining it and the exit status of the last part that ran (a
    `;` always runs it). An exact part prints its text; a gap that runs is answered by `answer`,
    which gives its observation's body and exit status, and prints that body with a final line
    break. The status is the last run part's. None when a gap's unknown status would decide
    whether a part runs."""
    pieces: list[str] = []
    last: int | None = None
    for index, part in enumerate(parts):
        separator = part["separator"] if index else ""
        if separator in ("&&", "||"):
            if last is None:
                return None
            if (separator == "&&") != (last == 0):
                continue
        if part["kind"] == "gap":
            body, last = await answer(part)
            pieces.append(body + "\n" if body else "")
        else:
            pieces.append(part["output"])
            last = part["returncode"]
    return "".join(pieces).removesuffix("\n"), last


def _valid_part(part) -> bool:
    if not isinstance(part, dict):
        return False
    if part.get("kind") == "exact":
        return isinstance(part.get("output"), str) and isinstance(part.get("returncode"), int)
    return part.get("kind") == "gap" and all(
        isinstance(part.get(key), str) for key in ("command", "context")
    )


def _valid_parts(value) -> list[dict] | None:
    """The parts of a response when each is well formed, else None: the whole command is then
    simulated as one."""
    return value if isinstance(value, list) and value and all(map(_valid_part, value)) else None


class RepoContextClient:
    def __init__(self, settings: JudgeSettings):
        self._client = httpx.AsyncClient(
            base_url=settings.repo_context_url.rstrip("/"),
            timeout=httpx.Timeout(settings.repo_context_timeout_seconds, pool=None),
            limits=httpx.Limits(max_connections=8),
        )
        self._last_warning = 0.0

    async def context_for(
        self, sample_id: str, assistant_output: str, messages: list[dict[str, str]] | None = None
    ) -> Grounding:
        try:
            response = await self._client.post(
                "/repo-context",
                json={
                    "sample_id": sample_id,
                    "assistant_output": assistant_output,
                    "messages": messages or [],
                },
            )
            response.raise_for_status()
            body = response.json()
            context = body.get("context")
            exact = body.get("exact_output")
            returncode = body.get("exact_returncode")
            state = body.get("state")
            leading = body.get("leading_output")
            return Grounding(
                context if isinstance(context, str) and context else None,
                exact if isinstance(exact, str) else None,
                returncode if isinstance(returncode, int) else None,
                state if isinstance(state, str) else "",
                leading if isinstance(leading, str) else None,
                _valid_parts(body.get("parts")),
            )
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_warning > 60.0:
                self._last_warning = now
                logger.warning(
                    "repo_context_unavailable sample_id={} error={}",
                    sample_id,
                    f"{type(exc).__name__}: {exc}",
                )
            return Grounding(None, None, None, "")

    async def aclose(self) -> None:
        await self._client.aclose()
