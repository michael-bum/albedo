from __future__ import annotations

import json
import secrets
import time
from collections.abc import Iterator

from agent.protocols.tools import _decide


def _responses_reply(rid, model, text, tool, usage, rc_marker) -> tuple[dict, Iterator[str]]:
    content, args, outcome = _decide(text, tool, rc_marker)
    output: list[dict] = []
    if content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{secrets.token_hex(12)}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )
    if outcome == "call":
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{secrets.token_hex(12)}",
                "call_id": f"call_{secrets.token_hex(8)}",
                "name": tool.name,
                "arguments": json.dumps(args),
                "status": "completed",
            }
        )
    u = usage or {}
    in_tok = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
    out_tok = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    rid = rid if rid.startswith("resp_") else f"resp_{rid}"
    body = {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }

    def events() -> Iterator[str]:
        seq = 0

        def ev(kind: str, data: dict) -> str:
            nonlocal seq
            seq += 1
            payload = {"type": kind, "sequence_number": seq, **data}
            return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"

        pending = {**body, "status": "in_progress", "output": [], "usage": None}
        yield ev("response.created", {"response": pending})
        yield ev("response.in_progress", {"response": pending})
        for i, item in enumerate(output):
            if item["type"] == "message":
                head = {**item, "status": "in_progress", "content": []}
                yield ev("response.output_item.added", {"output_index": i, "item": head})
                part = {"type": "output_text", "text": "", "annotations": []}
                yield ev(
                    "response.content_part.added",
                    {"item_id": item["id"], "output_index": i, "content_index": 0, "part": part},
                )
                text_out = item["content"][0]["text"]
                yield ev(
                    "response.output_text.delta",
                    {
                        "item_id": item["id"],
                        "output_index": i,
                        "content_index": 0,
                        "delta": text_out,
                    },
                )
                yield ev(
                    "response.output_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": i,
                        "content_index": 0,
                        "text": text_out,
                    },
                )
                yield ev(
                    "response.content_part.done",
                    {
                        "item_id": item["id"],
                        "output_index": i,
                        "content_index": 0,
                        "part": item["content"][0],
                    },
                )
            else:
                head = {**item, "status": "in_progress", "arguments": ""}
                yield ev("response.output_item.added", {"output_index": i, "item": head})
                yield ev(
                    "response.function_call_arguments.delta",
                    {"item_id": item["id"], "output_index": i, "delta": item["arguments"]},
                )
                yield ev(
                    "response.function_call_arguments.done",
                    {"item_id": item["id"], "output_index": i, "arguments": item["arguments"]},
                )
            yield ev("response.output_item.done", {"output_index": i, "item": item})
        yield ev("response.completed", {"response": body})

    return body, events()
