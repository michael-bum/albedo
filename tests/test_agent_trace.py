from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.api import create_ide_app  # noqa: E402
from agent.config import KingAgentSettings  # noqa: E402
from agent.db import KeyStore  # noqa: E402
from agent_key_helpers import FakeEngine, issue_key  # noqa: E402
from common.king_meta import KingInfoSource  # noqa: E402
from test_ide_agent_route import KING, TERMINAL, _fake_vllm  # noqa: E402


def _app(pg_url, trace_dir, debug=True):
    settings = KingAgentSettings(database_url=pg_url, trace_dir=trace_dir, debug=debug)
    engine = FakeEngine()
    store = KeyStore(pg_url)
    key = issue_key(store, "tracer", rpm=10, parallel=2, daily_tokens=10_000)
    app = create_ide_app(engine, settings, store, king=KingInfoSource(fixed=KING))
    seen: list[dict] = []
    app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ide")
    return client, key


@pytest.mark.anyio
async def test_trace_records_agent_turn_and_passthrough(tmp_path, pg_url):
    trace_dir = tmp_path / "traces"
    client, key = _app(pg_url, str(trace_dir))
    headers = {"Authorization": f"Bearer {key}", "user-agent": "claude-cli/2.0"}
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "albedo-king-agent",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [TERMINAL],
        },
        headers=headers,
    )
    assert r.status_code == 200
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )
    assert r.status_code == 200
    files = list(trace_dir.rglob("*.jsonl"))
    assert len(files) == 1 and files[0].name == "claude-code.jsonl"
    sessions = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(sessions) == 2 and all(len(s["turns"]) == 1 for s in sessions)
    agent, plain = sessions[0]["turns"][0], sessions[1]["turns"][0]
    assert sessions[0]["client"] == "claude-code" and sessions[0]["account"] == "tracer"
    assert agent["path"] == "/v1/chat/completions" and agent["account"] == "tracer"
    assert agent["model_requested"] == "albedo-king-agent"
    assert agent["client_payload"]["tools"][0]["function"]["name"] == TERMINAL["function"]["name"]
    assert agent["king_requests"][0]["messages"][-1]["content"] == "list files"
    assert agent["king_responses"][0]["choices"][0]["message"]["content"]
    assert agent["final_text"] and agent["reply"]["choices"][0]["message"]["tool_calls"]
    assert "duration_ms" in agent
    assert plain["notes"] == ["passthrough 200"]
    assert plain["king_requests"][0]["messages"][0]["content"] == "hi"
    assert plain["king_responses"][0]["choices"][0]["message"]["content"]


def test_trace_root_follows_debug_flag(tmp_path):
    off = KingAgentSettings(trace_dir=str(tmp_path / "t"))
    assert off.trace_root == ""
    on = KingAgentSettings(trace_dir=str(tmp_path / "traces"), debug=True)
    assert on.trace_root == str(tmp_path / "traces")


@pytest.mark.anyio
async def test_trace_disabled_writes_nothing(tmp_path, pg_url):
    client, key = _app(pg_url, str(tmp_path / "traces"), debug=False)
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 200
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.anyio
async def test_trace_groups_turns_of_one_session(tmp_path, pg_url):
    trace_dir = tmp_path / "traces"
    client, key = _app(pg_url, str(trace_dir))
    headers = {"Authorization": f"Bearer {key}", "user-agent": "codex_cli_rs/0.154"}
    first = {"role": "user", "content": "count the files"}
    r1 = await client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king-agent", "messages": [first], "tools": [TERMINAL]},
        headers=headers,
    )
    assert r1.status_code == 200
    follow_up = [
        first,
        r1.json()["choices"][0]["message"],
        {"role": "tool", "tool_call_id": "x", "content": "3"},
    ]
    r2 = await client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king-agent", "messages": follow_up, "tools": [TERMINAL]},
        headers=headers,
    )
    assert r2.status_code == 200
    r3 = await client.post(
        "/v1/chat/completions",
        json={
            "model": "albedo-king-agent",
            "messages": [{"role": "user", "content": "something else"}],
            "tools": [TERMINAL],
        },
        headers=headers,
    )
    assert r3.status_code == 200
    lines = (trace_dir.rglob("codex.jsonl").__next__()).read_text().splitlines()
    sessions = [json.loads(line) for line in lines]
    assert [len(s["turns"]) for s in sessions] == [2, 1]
    assert sessions[0]["turns"][1]["client_payload"]["messages"][-1]["content"] == "3"
    assert sessions[0]["last_ts"] >= sessions[0]["started_ts"]
    assert "session" not in sessions[0]["turns"][0]


def test_session_key_prefers_client_ids():
    from agent.devtools.trace import session_key

    cc = {
        "metadata": {"user_id": json.dumps({"device_id": "d", "session_id": "abc"})},
        "messages": [],
    }
    assert session_key(cc, "anthropic", "claude-code", "a") == session_key(
        {**cc, "messages": [{"role": "user", "content": "different"}]},
        "anthropic",
        "claude-code",
        "a",
    )
    codex = {"prompt_cache_key": "k1", "input": [{"role": "user", "content": "x"}]}
    assert session_key(codex, "responses", "codex", "a") != session_key(
        {**codex, "prompt_cache_key": "k2"}, "responses", "codex", "a"
    )
    plain = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]}
    assert session_key(plain, "openai", "cursor", "a") != session_key(
        plain, "openai", "cursor", "b"
    )
