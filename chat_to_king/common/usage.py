from __future__ import annotations

import json
import re

_USAGE_RE = re.compile(rb'"usage"\s*:\s*(\{[^{}]*\})')
TAIL_BYTES = 16_384


def usage_from_bytes(raw: bytes) -> tuple[int, int]:
    matches = _USAGE_RE.findall(raw[-TAIL_BYTES:])
    for blob in reversed(matches):
        try:
            usage = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(usage, dict):
            continue
        if usage.get("prompt_tokens") is not None:
            return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        if usage.get("input_tokens") is not None or usage.get("output_tokens") is not None:
            return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    return 0, 0
