from __future__ import annotations

import json
import secrets
from collections.abc import Iterator

from agent.protocols.tools import _decide


def _anthropic_reply(rid, model, text, tool, usage, rc_marker) -> tuple[dict, Iterator[str]]:
    content, args, outcome = _decide(text, tool, rc_marker)
    blocks: list[dict] = []
    if content:
        blocks.append({"type": "text", "text": content})
    if outcome == "call":
        blocks.append(
            {
                "type": "tool_use",
                "id": f"toolu_{secrets.token_hex(12)}",
                "name": tool.name,
                "input": args,
            }
        )
        stop = "tool_use"
    else:
        stop = "end_turn"
    u = usage or {}
    in_tok = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    out_tok = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    body = {
        "id": rid,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }

    def events() -> Iterator[str]:
        def ev(kind: str, data: dict) -> str:
            return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"

        start = {**body, "content": [], "stop_reason": None}
        start["usage"] = {"input_tokens": in_tok, "output_tokens": 0}
        yield ev("message_start", {"message": start})
        for i, b in enumerate(blocks):
            if b["type"] == "text":
                yield ev(
                    "content_block_start",
                    {"index": i, "content_block": {"type": "text", "text": ""}},
                )
                yield ev(
                    "content_block_delta",
                    {"index": i, "delta": {"type": "text_delta", "text": b["text"]}},
                )
            else:
                head = {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}
                yield ev("content_block_start", {"index": i, "content_block": head})
                yield ev(
                    "content_block_delta",
                    {
                        "index": i,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(b["input"]),
                        },
                    },
                )
            yield ev("content_block_stop", {"index": i})
        yield ev(
            "message_delta",
            {
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": out_tok},
            },
        )
        yield ev("message_stop", {})

    return body, events()
