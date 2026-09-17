from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent_key_helpers import FakeEngine  # noqa: E402
from common.king_meta import KingInfo, KingInfoSource  # noqa: E402
from web.config import KingChatSettings  # noqa: E402
from web.gateway import STATUS_HEADER, create_app  # noqa: E402

KING = KingInfo(roman="CXXV", repo="dendriteholdings/king", sha="a" * 40, hotkey="5Fabc")


def _fake_vllm(seen: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        seen.append(payload)
        return JSONResponse(
            {
                "id": "c1",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
            }
        )

    return app


@pytest.fixture
async def env(tmp_path):
    llms = tmp_path / "llms.txt"
    llms.write_text("Albedo is subnet 97. Kings are crowned by duel.")
    settings = KingChatSettings(llms_path=str(llms), retry_after_s=30)
    engine = FakeEngine()
    app = create_app(settings, engine, KingInfoSource(fixed=KING))
    seen: list[dict] = []
    async with app.router.lifespan_context(app):
        app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen)))
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://chat")
        yield type("Env", (), {"client": client, "engine": engine, "seen": seen, "app": app})()


@pytest.mark.anyio
async def test_models_and_health_show_the_roman(env):
    r = await env.client.get("/v1/models")
    assert r.json()["data"][0]["id"] == "albedo-king-cxxv"
    assert r.json()["data"][0]["king"]["roman"] == "CXXV"
    h = (await env.client.get("/health")).json()
    assert h["state"] == "serving" and h["king"]["model_id"] == "albedo-king-cxxv"
    for path in ("/docs", "/openapi.json"):
        assert (await env.client.get(path)).status_code == 404


@pytest.mark.anyio
async def test_roman_id_is_rewritten_and_knowledge_injected(env):
    body = {
        "model": "albedo-king-cxxv",
        "messages": [{"role": "user", "content": "what is albedo?"}],
    }
    r = await env.client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200 and r.headers[STATUS_HEADER] == "serving"
    sent = env.seen[0]
    assert sent["model"] == "albedo-king"
    assert sent["messages"][0]["role"] == "system" and "King CXXV" in sent["messages"][0]["content"]
    assert "subnet 97" in sent["messages"][0]["content"]

    r = await env.client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king", "messages": [{"role": "user", "content": "hi there"}]},
    )
    assert r.status_code == 200 and env.seen[1]["messages"][0]["role"] == "user"


@pytest.mark.anyio
async def test_loading_notice_when_engine_is_down(env):
    env.engine.serving = False
    r = await env.client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king-cxxv", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200 and r.headers[STATUS_HEADER] == "loading"
    assert r.headers["retry-after"] == "30"
    text = r.json()["choices"][0]["message"]["content"]
    assert "King CXXV" in text and "loading" in text and r.json()["model"] == "albedo-king-cxxv"
    assert env.seen == []
