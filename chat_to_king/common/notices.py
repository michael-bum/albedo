from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse

from common.king_meta import KingInfo

NOTICE_ID = "chatcmpl-king-loading"


def loading_text(info: KingInfo | None) -> str:
    if info is None:
        return "The model is loading — please resend your message in a moment."
    return (
        f"👑 King {info.roman} is loading on the GPUs (~1–2 min). "
        "Please resend your message shortly."
    )


def openai_notice(text: str, model: str, stream: bool):
    created = int(time.time())
    if not stream:
        body = {
            "id": NOTICE_ID,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(body)

    def _chunk(delta: dict, finish=None) -> str:
        payload = {
            "id": NOTICE_ID,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def _gen() -> AsyncIterator[str]:
        yield _chunk({"role": "assistant", "content": text})
        yield _chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
