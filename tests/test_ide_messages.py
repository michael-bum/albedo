from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.api import STATUS_HEADER, create_ide_app  # noqa: E402
from agent.config import KingAgentSettings  # noqa: E402
from agent.db import KeyStore  # noqa: E402
from agent_key_helpers import FakeEngine, issue_key  # noqa: E402
from common.king_meta import KingInfo, KingInfoSource  # noqa: E402

KING = KingInfo(roman="CXXV", repo="dendriteholdings/king", sha="a" * 40, hotkey="5Fabc")


def _fake_vllm(seen: list[tuple[str, dict]]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request):
        payload = json.loads(await request.body())
        seen.append(("/v1/messages", payload))
        if payload.get("stream"):

            async def gen():
                start = {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 30, "output_tokens": 0}},
                }
                yield f"event: message_start\ndata: {json.dumps(start)}\n\n"
                delta = {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hi"},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                end = {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7},
                }
                yield f"event: message_delta\ndata: {json.dumps(end)}\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {"input_tokens": 30, "output_tokens": 7},
            }
        )

    @app.post("/v1/messages/count_tokens")
    async def count(request: Request):
        seen.append(("/v1/messages/count_tokens", json.loads(await request.body())))
        return JSONResponse({"input_tokens": 30})

    return app


@pytest.fixture
def env(pg_url):
    settings = KingAgentSettings(database_url=pg_url, max_tokens_cap=100)
    engine = FakeEngine()
    store = KeyStore(pg_url)
    key = issue_key(store, "cc", rpm=10, parallel=2, daily_tokens=10_000)
    app = create_ide_app(engine, settings, store, king=KingInfoSource(fixed=KING))
    seen: list[tuple[str, dict]] = []
    app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ide")
    return SimpleNamespace(store=store, key=key, client=client, seen=seen)


def _body(**extra) -> dict:
    return {
        "model": "albedo-king",
        "max_tokens": 5000,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


@pytest.mark.anyio
async def test_messages_nonstream_proxied_and_usage_recorded(env):
    r = await env.client.post(
        "/v1/messages", json=_body(), headers={"authorization": f"Bearer {env.key}"}
    )
    assert r.status_code == 200 and r.headers[STATUS_HEADER] == "serving"
    assert r.json()["content"][0]["text"] == "Hi"
    path, payload = env.seen[0]
    assert path == "/v1/messages" and payload["max_tokens"] == 100
    assert "stream_options" not in payload
    assert env.store.report(1)[0]["prompt_tokens"] == 30
    assert env.store.report(1)[0]["completion_tokens"] == 7


@pytest.mark.anyio
async def test_messages_stream_passthrough_no_stream_options(env):
    r = await env.client.post(
        "/v1/messages", json=_body(stream=True), headers={"authorization": f"Bearer {env.key}"}
    )
    assert r.status_code == 200
    assert "event: message_start" in r.text and "text_delta" in r.text
    assert "stream_options" not in env.seen[0][1]
    assert env.store.report(1)[0]["completion_tokens"] == 7


@pytest.mark.anyio
async def test_count_tokens_needs_key_and_is_proxied(env):
    r = await env.client.post("/v1/messages/count_tokens", json=_body())
    assert r.status_code == 401
    r = await env.client.post(
        "/v1/messages/count_tokens", json=_body(), headers={"authorization": f"Bearer {env.key}"}
    )
    assert r.status_code == 200 and r.json() == {"input_tokens": 30}
