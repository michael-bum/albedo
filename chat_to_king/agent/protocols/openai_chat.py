from __future__ import annotations

import json
import secrets
import time
from collections.abc import Iterator

from agent.protocols.tools import _decide


def _openai_usage(usage: dict | None) -> dict:
    u = usage or {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens") or 0),
        "completion_tokens": int(u.get("completion_tokens") or 0),
        "total_tokens": int(u.get("total_tokens") or 0),
    }


def _openai_reply(rid, model, text, tool, usage, rc_marker) -> tuple[dict, Iterator[str]]:
    content, args, outcome = _decide(text, tool, rc_marker)
    created = int(time.time())
    if outcome == "call":
        call = {
            "id": f"call_{secrets.token_hex(8)}",
            "type": "function",
            "function": {"name": tool.name, "arguments": json.dumps(args)},
        }
        message = {"role": "assistant", "content": content or None, "tool_calls": [call]}
        finish = "tool_calls"
    else:
        message, finish = {"role": "assistant", "content": content}, "stop"
    body = {
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": _openai_usage(usage),
    }

    def events() -> Iterator[str]:
        def chunk(delta: dict, finish_reason: str | None = None) -> str:
            c = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(c)}\n\n"

        yield chunk({"role": "assistant", "content": ""})
        if message.get("content"):
            yield chunk({"content": message["content"]})
        if message.get("tool_calls"):
            deltas = [
                {"index": i, "id": c["id"], "type": "function", "function": c["function"]}
                for i, c in enumerate(message["tool_calls"])
            ]
            yield chunk({"tool_calls": deltas})
        yield chunk({}, finish)
        tail = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": body["usage"],
        }
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    return body, events()
