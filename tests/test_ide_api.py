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

from agent.api import STATUS_HEADER, Limiter, create_ide_app  # noqa: E402
from agent.config import KingAgentSettings  # noqa: E402
from agent.db import ApiKey, KeyStore  # noqa: E402
from agent_key_helpers import FakeEngine, issue_key  # noqa: E402
from common.king_meta import KingInfo, KingInfoSource  # noqa: E402

KING = KingInfo(roman="CXXIV", repo="dendriteholdings/king", sha="a" * 40, hotkey="5Fabc")


def _fake_vllm(seen: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        seen.append(payload)
        if payload.get("stream"):

            async def gen():
                for tok in ("Hel", "lo"):
                    chunk = {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": tok}}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                tail = {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
                }
                yield f"data: {json.dumps(tail)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "c1",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            }
        )

    return app


@pytest.fixture
def env(pg_url):
    settings = KingAgentSettings(
        database_url=pg_url,
        max_tokens_cap=100,
        max_body_bytes=2000,
        max_prompt_chars=500,
        retry_after_s=45,
    )
    engine = FakeEngine()
    store = KeyStore(pg_url)
    key = issue_key(store, "alice", rpm=2, parallel=2, daily_tokens=1000)
    app = create_ide_app(engine, settings, store, king=KingInfoSource(fixed=KING))
    seen: list[dict] = []
    app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ide")
    return SimpleNamespace(
        settings=settings, engine=engine, store=store, key=key, app=app, client=client, seen=seen
    )


def _auth(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


def _chat(content: str = "hi", **extra) -> dict:
    return {"model": "albedo-king", "messages": [{"role": "user", "content": content}], **extra}


async def _post(env, body, key=None, headers=None):
    return await env.client.post(
        "/v1/chat/completions", json=body, headers=headers or _auth(key or env.key)
    )


def _total_tokens(env) -> int:
    row = env.store.report(1)[0]
    return row["prompt_tokens"] + row["completion_tokens"]


@pytest.mark.anyio
async def test_auth_required_and_revocation(env):
    r = await env.client.post("/v1/chat/completions", json=_chat())
    assert r.status_code == 401
    r = await _post(env, _chat(), key="ak-nope")
    assert r.status_code == 401
    assert (await env.client.get("/v1/models")).status_code == 401
    r = await env.client.get("/v1/models", headers=_auth(env.key))
    ids = [m["id"] for m in r.json()["data"]]
    assert r.status_code == 200 and ids == ["albedo-king", "albedo-king-cxxiv", "albedo-king-agent"]
    assert r.json()["data"][1]["king"]["roman"] == "CXXIV"
    assert env.store.revoke(hint=env.key[-4:], actor="test") == 1
    r = await _post(env, _chat())
    assert r.status_code == 401


@pytest.mark.anyio
async def test_passthrough_records_usage_and_clamps(env):
    r = await _post(env, _chat(max_tokens=5000))
    assert r.status_code == 200
    assert r.headers[STATUS_HEADER] == "serving"
    assert r.json()["choices"][0]["message"]["content"] == "Hello"
    assert env.seen[0]["max_tokens"] == 100
    assert env.seen[0]["model"] == "albedo-king"
    assert "stream_options" not in env.seen[0]
    assert _total_tokens(env) == 17

    r = await _post(env, {"messages": [{"role": "user", "content": "x"}], "max_tokens": 7})
    assert r.status_code == 200 and env.seen[1]["max_tokens"] == 7
    assert env.seen[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.anyio
async def test_client_thinking_choice_is_kept(env):
    r = await _post(env, _chat(chat_template_kwargs={"enable_thinking": True, "x": 1}))
    assert r.status_code == 200
    assert env.seen[0]["chat_template_kwargs"] == {"enable_thinking": True, "x": 1}


@pytest.mark.anyio
async def test_streaming_relays_and_injects_usage(env):
    async with env.client.stream(
        "POST", "/v1/chat/completions", json=_chat(stream=True), headers=_auth(env.key)
    ) as r:
        assert r.status_code == 200
        text = "".join([chunk async for chunk in r.aiter_text()])
    assert text.count("data: ") == 4 and text.endswith("data: [DONE]\n\n")
    assert env.seen[0]["stream_options"] == {"include_usage": True}
    assert _total_tokens(env) == 17


@pytest.mark.anyio
async def test_request_validation(env):
    r = await _post(env, {"model": "gpt-4", "messages": []})
    assert r.status_code == 200 and env.seen[-1]["model"] == "albedo-king"
    r = await env.client.post("/v1/chat/completions", content=b"{nope", headers=_auth(env.key))
    assert r.status_code == 400
    r = await _post(env, _chat("x" * 600))
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    r = await env.client.post("/v1/chat/completions", content=b" " * 2001, headers=_auth(env.key))
    assert r.status_code == 413
    assert len(env.seen) == 1


@pytest.mark.anyio
async def test_rpm_limit(env):
    assert (await _post(env, _chat())).status_code == 200
    assert (await _post(env, _chat())).status_code == 200
    r = await _post(env, _chat())
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(r.headers["retry-after"]) >= 1
    assert len(env.seen) == 2


@pytest.mark.anyio
async def test_daily_quota(env):
    env.store.record(
        env.store.lookup(env.key),
        path="/v1/chat/completions",
        status=200,
        stream=False,
        prompt_tokens=900,
        completion_tokens=1000,
        latency_ms=1,
    )
    r = await _post(env, _chat())
    assert r.status_code == 429 and r.json()["error"]["code"] == "quota_exceeded"
    assert "Albedo channel on the Bittensor Discord" in r.json()["error"]["message"]


@pytest.mark.anyio
async def test_reload_notice_non_stream_and_stream(env):
    env.engine.serving = False
    r = await _post(env, _chat())
    assert r.status_code == 200
    assert r.headers[STATUS_HEADER] == "loading"
    assert r.headers["retry-after"] == "45"
    body = r.json()
    assert body["id"] == "chatcmpl-king-loading"
    text = body["choices"][0]["message"]["content"]
    assert "King CXXIV" in text and "loading" in text

    async with env.client.stream(
        "POST", "/v1/chat/completions", json=_chat(stream=True), headers=_auth(env.key)
    ) as r:
        assert r.headers[STATUS_HEADER] == "loading"
        stream_text = "".join([c async for c in r.aiter_text()])
    assert "King CXXIV" in stream_text and stream_text.endswith("data: [DONE]\n\n")
    assert env.seen == []

    s = (await env.client.get("/status")).json()
    assert s["state"] == "loading" and s["king"]["roman"] == "CXXIV"
    assert s["retry_after_s"] == 45 and "loading" in s["notice"]

    env.engine.serving = True
    s = (await env.client.get("/status")).json()
    assert s["state"] == "serving" and s["notice"] is None and s["retry_after_s"] == 0


@pytest.mark.anyio
async def test_upstream_down_returns_notice(env):
    def _boom(request):
        raise httpx.ConnectError("refused", request=request)

    env.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(_boom))
    r = await _post(env, _chat())
    assert r.status_code == 200 and r.headers[STATUS_HEADER] == "loading"
    assert "resend" in r.json()["choices"][0]["message"]["content"].lower()


def _limiter_key(account_id: int, parallel: int) -> ApiKey:
    return ApiKey(
        id=1,
        account_id=account_id,
        key_hash=f"h{account_id}",
        hint="abcd",
        label=None,
        owner="o",
        tier="standard",
        rpm=100,
        parallel=parallel,
        daily_completion_tokens=1,
        max_prompt_tokens=1000,
        created_at=0,
        expires_at=None,
        revoked_at=None,
        account_disabled_at=None,
    )


def test_limiter_parallel_and_global():
    key = _limiter_key(1, 1)
    other = _limiter_key(2, 5)
    lim = Limiter(global_parallel=2)
    assert lim.acquire(key, 0.0) is None
    assert lim.acquire(key, 0.0) == ("parallel", 0)
    assert lim.acquire(other, 0.0) is None
    assert lim.acquire(other, 0.0) == ("busy", 0)
    lim.release(1)
    assert lim.acquire(other, 0.0) is None


def test_limiter_is_per_account_not_per_key():
    lim = Limiter(global_parallel=0)
    first = _limiter_key(7, 1)
    rotated = ApiKey(**{**first.__dict__, "id": 2, "key_hash": "fresh"})
    assert lim.acquire(first, 0.0) is None
    assert lim.acquire(rotated, 0.0) == ("parallel", 0)


@pytest.mark.anyio
async def test_rpm_window_shared_across_keys_of_one_account(env):
    account_id = env.store.lookup(env.key).account_id
    _key, shown = env.store.issue(account_id, rpm=2, parallel=2, daily_completion_tokens=1000)
    assert (await _post(env, _chat(), key=env.key)).status_code == 200
    assert (await _post(env, _chat(), key=shown["albedo"])).status_code == 200
    r = await _post(env, _chat(), key=shown["albedo"])
    assert r.status_code == 429 and "rpm" in r.json()["error"]["message"]


@pytest.mark.anyio
async def test_no_schema_or_docs_exposed(env):
    for path in ("/docs", "/redoc", "/openapi.json", "/admin", "/v1/keys"):
        r = await env.client.get(path)
        assert r.status_code in (404, 405), path
        assert "openapi" not in r.text.lower()


@pytest.mark.anyio
@pytest.mark.parametrize("debug", [False, True])
async def test_unhandled_error_detail_only_in_debug(pg_url, debug):
    settings = KingAgentSettings(database_url=pg_url, debug=debug)
    engine = FakeEngine()
    store = KeyStore(pg_url)
    key = issue_key(store, "bob", rpm=5, parallel=2, daily_tokens=1000)
    app = create_ide_app(engine, settings, store, king=KingInfoSource(fixed=KING))

    def _boom(request):
        raise ValueError("secret detail")

    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(_boom))
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://ide"
    )
    r = await client.post("/v1/chat/completions", json=_chat(), headers=_auth(key))
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert ("secret detail" in r.json()["error"]["message"]) is debug
