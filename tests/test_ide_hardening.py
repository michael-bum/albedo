from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.api import create_ide_app  # noqa: E402
from agent.api.usage import _prompt_chars  # noqa: E402
from agent.config import KingAgentSettings  # noqa: E402
from agent.db import KeyStore  # noqa: E402
from agent_key_helpers import FakeEngine  # noqa: E402
from common.king_meta import KingInfo, KingInfoSource  # noqa: E402

KING = KingInfo(roman="CXXV", repo="dendriteholdings/king", sha="b" * 40, hotkey="5Fabc")
VALIDATION_400 = {
    "error": {
        "message": "1 validation error:\n  {'type': 'list_type', 'loc': ('body', 'messages')}",
        "type": "BadRequestError",
        "code": 400,
    }
}
CONTEXT_400 = {
    "error": {
        "message": "This model's maximum context length is 262144 tokens.",
        "type": "BadRequestError",
        "code": 400,
    }
}


def _fake_vllm(seen: list[dict], reply: dict[str, object]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        seen.append(payload)
        if reply.get("status"):
            return JSONResponse(reply["body"], status_code=reply["status"])
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
    settings = KingAgentSettings(database_url=pg_url, max_body_bytes=2000)
    store = KeyStore(pg_url)
    account = store.add_account("alice", tier="standard")
    _key, shown = store.issue(account, max_prompt_tokens=10)
    app = create_ide_app(FakeEngine(), settings, store, king=KingInfoSource(fixed=KING))
    seen: list[dict] = []
    reply: dict[str, object] = {}
    app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen, reply)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ide")
    return SimpleNamespace(store=store, key=shown["albedo"], client=client, seen=seen, reply=reply)


def _chat(content: str = "hi", **extra) -> dict:
    return {"model": "albedo-king", "messages": [{"role": "user", "content": content}], **extra}


async def _post(env, body, **headers):
    return await env.client.post(
        "/v1/chat/completions", json=body, headers={"authorization": f"Bearer {env.key}", **headers}
    )


@pytest.mark.anyio
async def test_declared_content_length_rejected_before_reading(env):
    r = await env.client.post(
        "/v1/chat/completions",
        content=b"{}",
        headers={"authorization": f"Bearer {env.key}", "content-length": "999999"},
    )
    assert r.status_code == 413 and env.seen == []


@pytest.mark.anyio
async def test_multi_sample_and_logprob_fields_are_stripped(env):
    r = await _post(env, _chat(n=50, best_of=50, prompt_logprobs=5, logprobs=True, top_logprobs=20))
    assert r.status_code == 200
    sent = env.seen[0]
    assert sent["n"] == 1 and sent["logprobs"] is True
    assert not {"best_of", "prompt_logprobs", "top_logprobs"} & sent.keys()
    await _post(env, _chat())
    assert "n" not in env.seen[1]


@pytest.mark.anyio
async def test_key_prompt_token_limit_enforced(env):
    r = await _post(env, _chat("x" * 200))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"
    assert "10 tokens" in r.json()["error"]["message"]
    assert env.seen == []
    r = await _post(env, _chat("short"))
    assert r.status_code == 200


def test_prompt_chars_covers_every_protocol_field():
    assert _prompt_chars({"messages": [{"role": "user", "content": "abcd"}]}) >= 4
    assert _prompt_chars({"system": "s" * 50, "messages": []}) >= 50
    assert (
        _prompt_chars(
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": "t" * 30}]}]}
        )
        >= 30
    )
    assert _prompt_chars({"prompt": "p" * 7}) == 7
    assert _prompt_chars({"messages": [{"role": "user", "content": 5}]}) < 10
    assert _prompt_chars({"prompt": list(range(100))}) >= 400
    assert _prompt_chars({"prompt": [list(range(50)), list(range(50))]}) >= 400
    assert (
        _prompt_chars({"messages": [], "tools": [{"function": {"description": "d" * 300}}]}) >= 300
    )
    assert _prompt_chars({"prompt": True}) == 0


@pytest.mark.anyio
async def test_every_reply_length_field_is_capped(env):
    r = await _post(env, _chat(max_tokens=99999, max_completion_tokens=99999))
    assert r.status_code == 200
    sent = env.seen[0]
    assert sent["max_tokens"] == 8192 and sent["max_completion_tokens"] == 8192
    r = await env.client.post(
        "/v1/responses",
        json={"model": "albedo-king", "input": "hi", "max_output_tokens": 99999},
        headers={"authorization": f"Bearer {env.key}"},
    )
    assert r.status_code in (200, 404, 500)
    r = await _post(env, _chat(max_tokens=True))
    assert env.seen[-1]["max_tokens"] == 8192


@pytest.mark.anyio
async def test_expensive_engine_extras_are_stripped(env):
    r = await _post(
        env,
        _chat(
            min_tokens=8000,
            ignore_eos=True,
            guided_regex="(a|b)*",
            guided_json={"type": "object"},
            structured_outputs={"x": 1},
            priority=-100,
            temperature=0.2,
        ),
    )
    assert r.status_code == 200
    sent = env.seen[0]
    assert (
        not {
            "min_tokens",
            "ignore_eos",
            "guided_regex",
            "guided_json",
            "structured_outputs",
            "priority",
        }
        & sent.keys()
    )
    assert sent["temperature"] == 0.2


@pytest.mark.anyio
async def test_token_id_prompt_hits_the_prompt_cap(env):
    r = await env.client.post(
        "/v1/completions",
        json={"model": "albedo-king", "prompt": list(range(500))},
        headers={"authorization": f"Bearer {env.key}"},
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    assert env.seen == []


@pytest.mark.anyio
async def test_upstream_validation_text_is_replaced(env):
    env.reply.update(status=400, body=VALIDATION_400)
    r = await _post(env, _chat())
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "Invalid request body."
    assert "loc" not in r.text


@pytest.mark.anyio
async def test_upstream_context_error_passes_through(env):
    env.reply.update(status=400, body=CONTEXT_400)
    r = await _post(env, _chat())
    assert r.status_code == 400
    assert "maximum context length" in r.json()["error"]["message"]


@pytest.mark.anyio
async def test_streaming_request_with_upstream_error_returns_json(env):
    env.reply.update(status=400, body=VALIDATION_400)
    r = await _post(env, _chat(stream=True))
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["message"] == "Invalid request body."
