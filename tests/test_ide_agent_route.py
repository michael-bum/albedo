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
from agent.config import KingAgentSettings  # noqa: E402
from agent.db import KeyStore  # noqa: E402
from agent_key_helpers import FakeEngine, issue_key  # noqa: E402
from common.king_meta import KingInfo, KingInfoSource  # noqa: E402

KING = KingInfo(roman="CXXV", repo="dendriteholdings/king", sha="a" * 40, hotkey="5Fabc")
TERMINAL = {
    "type": "function",
    "function": {
        "name": "run_in_terminal",
        "description": "Run a shell command in the integrated terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "explanation": {"type": "string"},
                "isBackground": {"type": "boolean"},
            },
            "required": ["command", "explanation", "isBackground"],
        },
    },
}
BASH = {
    "name": "Bash",
    "description": "Executes a bash command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
        "required": ["command"],
    },
}
KING_REPLY = "THOUGHT: look around first\n\n```bash\nls -la\n```"


def _fake_vllm(seen: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        seen.append(payload)
        return JSONResponse(
            {
                "id": "chatcmpl-abc",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": KING_REPLY}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
            }
        )

    return app


@pytest.fixture
def env(tmp_path, pg_url, request):
    debug = request.node.get_closest_marker("debug") is not None
    settings = KingAgentSettings(
        database_url=pg_url, trace_dir=str(tmp_path / "traces"), debug=debug
    )
    engine = FakeEngine()
    store = KeyStore(pg_url)
    key = issue_key(store, "copilot", rpm=10, parallel=2, daily_tokens=10_000)
    app = create_ide_app(engine, settings, store, king=KingInfoSource(fixed=KING))
    seen: list[dict] = []
    app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ide")
    return SimpleNamespace(
        store=store, key=key, client=client, seen=seen, auth={"authorization": f"Bearer {key}"}
    )


def _openai_body(**extra) -> dict:
    return {
        "model": "albedo-king-agent",
        "messages": [
            {"role": "system", "content": "You are Copilot."},
            {"role": "user", "content": "Create hello.py"},
        ],
        "tools": [TERMINAL],
        "temperature": 0.1,
        **extra,
    }


@pytest.mark.anyio
async def test_openai_agent_returns_tool_call_and_keeps_client_prompt(env):
    r = await env.client.post("/v1/chat/completions", json=_openai_body(), headers=env.auth)
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "run_in_terminal"
    assert json.loads(call["function"]["arguments"])["command"] == "ls -la"
    assert choice["message"]["content"] == "look around first"
    assert r.json()["model"] == "albedo-king-agent"

    sent = env.seen[0]
    assert sent["model"] == "albedo-king" and not sent.get("stream") and "tools" not in sent
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["messages"][0] == {"role": "system", "content": "You are Copilot."}
    assert (
        sent["messages"][1]["role"] == "system"
        and "run_in_terminal" in sent["messages"][1]["content"]
    )
    assert sent["messages"][2] == {"role": "user", "content": "Create hello.py"}
    row = env.store.report(1)[0]
    assert row["prompt_tokens"] == 40 and row["completion_tokens"] == 12


@pytest.mark.anyio
async def test_openai_agent_streams(env):
    r = await env.client.post(
        "/v1/chat/completions", json=_openai_body(stream=True), headers=env.auth
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert '"tool_calls"' in r.text and r.text.rstrip().endswith("data: [DONE]")


@pytest.mark.anyio
async def test_anthropic_agent_returns_tool_use(env):
    body = {
        "model": "albedo-king-agent",
        "max_tokens": 1000,
        "system": "You are Claude Code.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Create hello.py"}]}],
        "tools": [BASH],
    }
    r = await env.client.post("/v1/messages", json=body, headers=env.auth)
    assert r.status_code == 200
    data = r.json()
    assert data["stop_reason"] == "tool_use" and data["content"][1]["name"] == "Bash"
    assert data["content"][1]["input"]["command"] == "ls -la"
    assert env.seen[0]["messages"][0] == {"role": "system", "content": "You are Claude Code."}

    body["stream"] = True
    r = await env.client.post("/v1/messages", json=body, headers=env.auth)
    assert (
        r.status_code == 200
        and "event: message_start" in r.text
        and "event: message_stop" in r.text
    )


@pytest.mark.anyio
async def test_agent_model_without_shell_tool_answers_as_text(env):
    body = _openai_body()
    body["tools"] = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop" and "tool_calls" not in choice["message"]
    assert "```bash" in choice["message"]["content"]
    assert all("Command execution protocol" not in m["content"] for m in env.seen[0]["messages"])


@pytest.mark.anyio
async def test_unknown_model_name_is_served_by_the_king(env):
    body = _openai_body(model="gpt-5.6-luna")
    body.pop("tools")
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.status_code == 200 and env.seen[0]["model"] == "albedo-king"


@pytest.mark.anyio
async def test_responses_without_tools_uses_adapter_not_passthrough(env):
    body = {"model": "gpt-5.6-luna", "input": "Give this thread a title.", "store": False}
    r = await env.client.post("/v1/responses", json=body, headers=env.auth)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "response" and data["output"][0]["type"] == "message"
    assert env.seen[0]["messages"][-1] == {"role": "user", "content": "Give this thread a title."}


@pytest.mark.anyio
async def test_models_lists_both_ids(env):
    r = await env.client.get("/v1/models", headers=env.auth)
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == ["albedo-king", "albedo-king-cxxv", "albedo-king-agent"]


@pytest.mark.anyio
async def test_any_other_model_name_runs_the_agent_adapter(env):
    body = _openai_body(model="claude-opus-5")
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert env.seen[0]["messages"][1]["role"] == "system"
    assert "run_in_terminal" in env.seen[0]["messages"][1]["content"]


@pytest.mark.anyio
async def test_reply_without_command_is_continued_inside_a_bash_block(env):
    replies = iter(["<Explore> <search>docs</search> </Explore>", "ls -la\n```"])
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        env.seen.append(payload)
        return JSONResponse(
            {
                "id": "chatcmpl-retry",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": next(replies)}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    gateway_app = env.client._transport.app
    gateway_app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    r = await env.client.post("/v1/chat/completions", json=_openai_body(), headers=env.auth)
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert (
        json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        == "ls -la"
    )
    assert len(env.seen) == 2
    second = env.seen[1]
    assert second["continue_final_message"] is True and second["add_generation_prompt"] is False
    assert second["messages"][-1] == {"role": "assistant", "content": "```bash\n"}
    assert env.store.report(1)[0]["prompt_tokens"] == 20


@pytest.mark.anyio
async def test_preamble_is_kept_as_thought_when_continued(env):
    replies = iter(["I'll list the files first.", "ls\n```"])
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        env.seen.append(payload)
        return JSONResponse(
            {
                "id": "x",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": next(replies)}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    env.client._transport.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app)
    )
    r = await env.client.post("/v1/chat/completions", json=_openai_body(), headers=env.auth)
    choice = r.json()["choices"][0]
    assert choice["message"]["content"] == "I'll list the files first."
    assert (
        json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])["command"] == "ls"
    )
    assert env.seen[1]["messages"][-1]["content"] == "I'll list the files first.\n\n```bash\n"


def _loop_history(n: int, cmd: str = "cat f.md") -> list[dict]:
    msgs = [
        {"role": "system", "content": "You are Copilot."},
        {"role": "user", "content": "fix it"},
    ]
    for i in range(n):
        args = json.dumps({"command": cmd, "explanation": "x", "isBackground": False})
        msgs.append(
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "run_in_terminal", "arguments": args},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "same output"})
    return msgs


@pytest.mark.anyio
async def test_repeated_command_gets_loop_note(env):
    body = _openai_body(messages=_loop_history(2))
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.status_code == 200
    last = env.seen[0]["messages"][-1]
    assert last["role"] == "user" and "NOTE FROM THE HARNESS" in last["content"]
    assert "2 times in a row" in last["content"]


@pytest.mark.anyio
async def test_persistent_loop_stops_the_turn(env):
    same = "THOUGHT: again\n\n```bash\ncat f.md\n```"
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        env.seen.append(json.loads(await request.body()))
        return JSONResponse(
            {
                "id": "x",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": same}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    env.client._transport.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app)
    )
    body = _openai_body(messages=_loop_history(4))
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop" and "tool_calls" not in choice["message"]
    assert "same command for the 5th time" in choice["message"]["content"]

    body = _openai_body(messages=_loop_history(1))
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.anyio
async def test_agent_sampling_defaults_and_client_override(pg_url):
    settings = KingAgentSettings(
        database_url=pg_url,
        agent_sampling='{"temperature": 1.0, "top_k": 20, "top_p": 0.95}',
        agent_loop_repetition_penalty=1.1,
    )
    engine = FakeEngine()
    store = KeyStore(pg_url)
    key = issue_key(store, "x", rpm=10, parallel=2, daily_tokens=10_000)
    app = create_ide_app(engine, settings, store, king=KingInfoSource(fixed=KING))
    seen: list[dict] = []
    app.state.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=_fake_vllm(seen)))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ide")
    auth = {"authorization": f"Bearer {key}"}

    body = _openai_body()
    body.pop("temperature")
    await client.post("/v1/chat/completions", json=body, headers=auth)
    assert seen[0]["temperature"] == 1.0 and seen[0]["top_k"] == 20 and seen[0]["top_p"] == 0.95
    assert "repetition_penalty" not in seen[0]

    await client.post("/v1/chat/completions", json=_openai_body(temperature=0.1), headers=auth)
    assert seen[1]["temperature"] == 0.1 and seen[1]["top_k"] == 20

    await client.post(
        "/v1/chat/completions", json=_openai_body(messages=_loop_history(2)), headers=auth
    )
    assert seen[2]["repetition_penalty"] == 1.1


@pytest.mark.anyio
async def test_repeated_reasoning_gets_note(env):
    same = "I see the pattern now. The subnet files all share one template and are mostly empty."
    msgs = [
        {"role": "system", "content": "You are Copilot."},
        {"role": "user", "content": "fill md files"},
    ]
    for i, f in enumerate(("a", "b", "c")):
        args = json.dumps({"command": f"cat {f}.md", "explanation": "x", "isBackground": False})
        msgs.append(
            {
                "role": "assistant",
                "content": same,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "run_in_terminal", "arguments": args},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "## Team"})
    r = await env.client.post(
        "/v1/chat/completions", json=_openai_body(messages=msgs), headers=env.auth
    )
    assert r.status_code == 200
    last = env.seen[0]["messages"][-1]
    assert last["role"] == "user" and "repeated the same reasoning" in last["content"]


@pytest.mark.anyio
async def test_chat_model_with_shell_tool_is_routed_to_agent(env):
    cursor_shell = {
        "type": "function",
        "function": {
            "name": "Shell",
            "description": "Executes a given command in a shell session",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_directory": {"type": "string"},
                    "block_until_ms": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    }
    body = {
        "model": "albedo-king",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "list files"}]}],
        "tools": [cursor_shell],
    }
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.status_code == 200
    call = r.json()["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "Shell"
    assert json.loads(call["arguments"]) == {"command": "ls -la"}
    assert env.seen[-1]["messages"][0]["role"] == "system"


@pytest.mark.anyio
async def test_python_block_triggers_harness_note_retry(env):
    replies = iter(
        [
            "Let me create the files.\n\n```python\nopen('a.py','w').write('x')\n```\n",
            "Writing the file with the shell.\n\n```bash\ncat > a.py <<'EOF'\nx\nEOF\n```",
        ]
    )
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        env.seen.append(payload)
        return JSONResponse(
            {
                "id": "chatcmpl-py",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": next(replies)}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    env.client._transport.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app)
    )
    r = await env.client.post("/v1/chat/completions", json=_openai_body(), headers=env.auth)
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    cmd = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])["command"]
    assert cmd.startswith("cat > a.py")
    assert len(env.seen) == 2
    tail = env.seen[1]["messages"][-2:]
    assert tail[0]["role"] == "assistant" and "```python" in tail[0]["content"]
    assert tail[1]["role"] == "user" and "```python code block" in tail[1]["content"]
    assert "continue_final_message" not in env.seen[1]


@pytest.mark.anyio
async def test_repeated_preamble_is_collapsed_and_continued(env):
    loop = "\n\n".join(["I'll read the existing website files to understand the structure."] * 40)
    replies = iter([loop, "ls website\n```"])
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        env.seen.append(payload)
        return JSONResponse(
            {
                "id": "chatcmpl-loop",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": next(replies)}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    env.client._transport.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app)
    )
    r = await env.client.post("/v1/chat/completions", json=_openai_body(), headers=env.auth)
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    args = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "ls website"
    assert choice["message"]["content"].count("I'll read") == 1
    assert env.seen[1]["continue_final_message"] is True


@pytest.mark.anyio
async def test_powershell_hint_from_cursor_user_info(env):
    body = _openai_body()
    body["messages"][1]["content"] = (
        "<user_info>\nOS Version: win32\nShell: powershell\n</user_info>\n<user_query>hi</user_query>"
    )
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.status_code == 200
    system = " ".join(m["content"] for m in env.seen[-1]["messages"] if m["role"] == "system")
    assert "Windows PowerShell" in system
    r = await env.client.post("/v1/chat/completions", json=_openai_body(), headers=env.auth)
    system = " ".join(m["content"] for m in env.seen[-1]["messages"] if m["role"] == "system")
    assert "Windows PowerShell" not in system


@pytest.mark.anyio
async def test_codex_side_task_json_schema_is_honoured(env):
    body = {
        "model": "gpt-5.6-luna",
        "instructions": "",
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "You are Codex."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Provide a short title for a task. User prompt: Add better styling",
                    }
                ],
            },
        ],
        "tools": [],
        "text": {
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "codex_output_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 36},
                        "description": {"type": "string", "minLength": 1},
                    },
                    "required": ["title", "description"],
                    "additionalProperties": False,
                },
            }
        },
    }
    r = await env.client.post("/v1/responses", json=body, headers=env.auth)
    assert r.status_code == 200
    out = r.json()["output"]
    assert out[0]["type"] == "message"
    data = json.loads(out[0]["content"][0]["text"])
    assert set(data) == {"title", "description"}
    assert 1 <= len(data["title"]) <= 36 and data["description"]


@pytest.mark.anyio
async def test_powershell_session_prefills_powershell_and_renders_history(env):
    replies = iter(
        [
            "I'll check the weather using the browser tools.\n\nLet me open a weather site now.",
            "Invoke-RestMethod -Uri 'https://wttr.in/SF?format=3'\n```",
        ]
    )
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = json.loads(await request.body())
        env.seen.append(payload)
        return JSONResponse(
            {
                "id": "chatcmpl-ps",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": next(replies)}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    env.client._transport.app.state.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app)
    )
    body = _openai_body()
    body["messages"] = [
        {
            "role": "user",
            "content": "<user_info>\nShell: powershell\n</user_info>\n<user_query>weather</user_query>",
        },
        {
            "role": "assistant",
            "content": "Listing.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "run_in_terminal",
                        "arguments": '{"command": "Get-ChildItem"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
        {"role": "user", "content": "<user_query>now the weather</user_query>"},
    ]
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    cmd = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])["command"]
    assert cmd.startswith("Invoke-RestMethod")
    second = env.seen[1]
    assert second["messages"][-1] == {
        "role": "assistant",
        "content": "I'll check the weather using the browser tools.\n\nLet me open a weather site now.\n\n```powershell\n",
    }
    history_call = [m for m in env.seen[0]["messages"] if m["role"] == "assistant"][0]
    assert "```powershell\nGet-ChildItem\n```" in history_call["content"]


@pytest.mark.anyio
async def test_default_agent_sampling_is_temperature_one(env):
    r = await env.client.post("/v1/chat/completions", json=_openai_body(), headers=env.auth)
    assert r.status_code == 200
    assert env.seen[0]["temperature"] == 0.1
    body = _openai_body()
    body.pop("temperature")
    r = await env.client.post("/v1/chat/completions", json=body, headers=env.auth)
    assert r.status_code == 200
    assert env.seen[1]["temperature"] == 1.0
    assert env.seen[1]["chat_template_kwargs"] == {"enable_thinking": False}
