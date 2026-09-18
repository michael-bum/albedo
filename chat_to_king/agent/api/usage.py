from __future__ import annotations

from common.usage import TAIL_BYTES as _TAIL_BYTES
from common.usage import usage_from_bytes as _usage_from_bytes

__all__ = ["_TAIL_BYTES", "_prompt_chars", "_usage_from_bytes"]


_PROMPT_FIELDS = ("messages", "system", "input", "prompt", "instructions", "tools")
_CHARS_PER_TOKEN_ID = 4


def _text_len(obj) -> int:
    if isinstance(obj, str):
        return len(obj)
    if isinstance(obj, bool):
        return 0
    if isinstance(obj, int):
        return _CHARS_PER_TOKEN_ID
    if isinstance(obj, dict):
        return sum(_text_len(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_text_len(v) for v in obj)
    return 0


def _prompt_chars(payload: dict) -> int:
    return sum(_text_len(payload.get(k)) for k in _PROMPT_FIELDS)
