from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import psycopg
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent_key_helpers import FakeEngine  # noqa: E402
from common.king_meta import KingInfo, KingInfoSource  # noqa: E402
from common.token_meter import TokenMeter, report, report_table  # noqa: E402
from web.config import KingChatSettings  # noqa: E402
from web.gateway import create_app  # noqa: E402

KING = KingInfo(roman="CXXV", repo="r", sha="s")


def test_meter_records_and_report_buckets(pg_url):
    chat = TokenMeter(pg_url, "chat")
    agent = TokenMeter(pg_url, "agent")
    assert chat.enabled and agent.enabled
    chat.record(100, 20, True)
    chat.record(50, 5, False)
    agent.record(1000, 300, False)
    TokenMeter("", "off").record(1, 1, False)
    rows = report(pg_url, window_hours=4, days=1)
    by = {(r["service"]): r for r in rows}
    assert by["chat"]["prompt_tokens"] == 150 and by["chat"]["completion_tokens"] == 25
    assert by["chat"]["requests"] == 2 and by["agent"]["prompt_tokens"] == 1000
    start = float(by["chat"]["window_start"])
    assert start % (4 * 3600) == 0 and time.time() - start < 4 * 3600
    table = report_table(rows)
    assert "chat in" in table and "1,000" in table and table.splitlines()[-1].startswith("total")
    with psycopg.connect(pg_url) as conn:
        assert conn.execute("SELECT count(*) FROM token_usage").fetchone()[0] == 3


def _fake_vllm(seen: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        seen.append(payload)
        usage = {"prompt_tokens": 40, "completion_tokens": 7, "total_tokens": 47}
        if payload.get("stream"):

            async def gen():
                yield 'data: {"id":"c","choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
                yield f"data: {json.dumps({'id': 'c', 'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {"id": "c", "choices": [{"index": 0, "message": {"content": "Hi"}}], "usage": usage}
        )

    return app


class FakeMeter:
    enabled = True

    def __init__(self):
        self.rows = []

    def record(self, p, c, stream):
        self.rows.append((p, c, stream))


@pytest.mark.anyio
async def test_chat_gateway_meters_stream_and_non_stream(tmp_path):
    settings = KingChatSettings(llms_path=str(tmp_path / "none.txt"))
    meter = FakeMeter()
    app = create_app(settings, FakeEngine(), KingInfoSource(fixed=KING), meter=meter)
    seen: list[dict] = []
    async with app.router.lifespan_context(app):
        app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen)))
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://chat")
        body = {"model": "albedo-king-cxxv", "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("/v1/chat/completions", json=body)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Hi"
        async with client.stream(
            "POST", "/v1/chat/completions", json={**body, "stream": True}
        ) as r:
            text = "".join([c async for c in r.aiter_text()])
        assert text.endswith("data: [DONE]\n\n")
    assert seen[0].get("stream_options") is None
    assert seen[1]["stream_options"] == {"include_usage": True}
    assert meter.rows == [(40, 7, False), (40, 7, True)]


def test_rollup_writes_closed_windows_to_sqlite(tmp_path, pg_url):
    import sqlite3

    from common.token_meter import rollup_to_sqlite

    chat = TokenMeter(pg_url, "chat")
    chat.record(10, 2, False)
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO token_usage (ts, service, prompt_tokens, completion_tokens, stream) "
            "VALUES (%s, 'agent', 500, 50, false), (%s, 'chat', 7, 1, true)",
            (time.time() - 5 * 3600, time.time() - 5 * 3600),
        )
    out = tmp_path / "w.sqlite"
    n = rollup_to_sqlite(pg_url, str(out), window_hours=4, days=1)
    assert n == 2
    assert rollup_to_sqlite(pg_url, str(out), window_hours=4, days=1) == 2
    with sqlite3.connect(out) as conn:
        rows = conn.execute(
            "SELECT service, prompt_tokens, completion_tokens, requests FROM token_windows ORDER BY service"
        ).fetchall()
    assert rows == [("agent", 500, 50, 1), ("chat", 7, 1, 1)]
