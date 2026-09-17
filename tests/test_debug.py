from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.devtools.probe import is_probe, strip_client_wrappers  # noqa: E402
from test_ide_agent_route import TERMINAL, env  # noqa: E402,F401


def test_probe_matches_through_client_wrappers():
    copilot = (
        "<current_datetime>2026-09-16T11:22:24+02:00</current_datetime>\n\nalbedo\n\n"
        "<system_reminder>\n<sql_tables>Available tables: todos</sql_tables>\n</system_reminder>\n\n"
        "<tagged_files>\n* c:\\x\\Meetings.md (8 lines)\n</tagged_files>"
        '<attachment mimeType="text/x-vscode-simple-attachment" name="Browser Pages">\nNo pages\n</attachment>'
    )
    assert strip_client_wrappers(copilot) == "albedo"
    assert is_probe({"messages": [{"role": "user", "content": copilot}]}, "openai")
    claude = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>\nAs you answer...\n</system-reminder>",
                    },
                    {"type": "text", "text": "Albedo"},
                ],
            }
        ]
    }
    assert is_probe(claude, "anthropic")
    assert not is_probe(
        {"messages": [{"role": "user", "content": "albedo please explain"}]}, "openai"
    )


@pytest.mark.anyio
async def test_probe_is_off_unless_debug(env):
    r = await env.client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king", "messages": [{"role": "user", "content": "albedo"}]},
        headers=env.auth,
    )
    assert r.status_code == 200 and len(env.seen) == 1
    assert r.json()["choices"][0]["message"]["content"] != "galbedo"


@pytest.mark.debug
@pytest.mark.anyio
async def test_albedo_probe_answers_galbedo_without_calling_the_king(env):
    r = await env.client.post(
        "/v1/chat/completions",
        json={"model": "albedo-king", "messages": [{"role": "user", "content": " Albedo "}]},
        headers=env.auth,
    )
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "galbedo"
    r = await env.client.post(
        "/v1/chat/completions",
        json={
            "model": "albedo-king-agent",
            "stream": True,
            "tools": [TERMINAL],
            "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "albedo"}],
        },
        headers=env.auth,
    )
    assert (
        r.status_code == 200 and '"galbedo"' in r.text and r.text.rstrip().endswith("data: [DONE]")
    )
    r = await env.client.post(
        "/v1/messages",
        json={
            "model": "albedo-king-agent",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "albedo"}]}],
        },
        headers=env.auth,
    )
    assert r.json()["content"] == [{"type": "text", "text": "galbedo"}]
    r = await env.client.post(
        "/v1/responses", json={"model": "x", "input": "albedo"}, headers=env.auth
    )
    assert r.json()["output"][0]["content"][0]["text"] == "galbedo"
    assert env.seen == []

    r = await env.client.post(
        "/v1/chat/completions",
        json={
            "model": "albedo-king",
            "messages": [{"role": "user", "content": "tell me about albedo"}],
        },
        headers=env.auth,
    )
    assert r.status_code == 200 and len(env.seen) == 1
