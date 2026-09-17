from __future__ import annotations

import re

from agent.protocols.parse import _responses_text, _text

PROBE_TRIGGER = "albedo"
PROBE_REPLY = "galbedo"


def last_user_text(payload: dict, kind: str) -> str:
    if kind == "responses":
        items = payload.get("input")
        if isinstance(items, str):
            return items
        for it in reversed(items if isinstance(items, list) else []):
            if not isinstance(it, dict):
                continue
            if it.get("type") not in (None, "message"):
                return ""
            if it.get("role") == "user":
                return _responses_text(it.get("content"))
            return ""
        return ""
    for m in reversed(payload.get("messages") or []):
        if not isinstance(m, dict) or m.get("role") in ("system", "developer"):
            continue
        if m.get("role") != "user":
            return ""
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            return ""
        return _text(content)
    return ""


_WRAPPER_TAG_RE = re.compile(r"<([A-Za-z_][\w.-]*)(?:\s[^>]*)?>.*?</\1\s*>", re.DOTALL)
_LONE_TAG_RE = re.compile(r"<[A-Za-z_][\w.-]*(?:\s[^>]*)?/?>")


def strip_client_wrappers(text: str) -> str:
    body = _WRAPPER_TAG_RE.sub("", text or "")
    return _LONE_TAG_RE.sub("", body).strip()


def is_probe(payload: dict, kind: str) -> bool:
    return strip_client_wrappers(last_user_text(payload, kind)).lower() == PROBE_TRIGGER
