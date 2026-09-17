from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.protocols import agent_reply, find_shell_tool, king_messages  # noqa: E402

CODEX_SHELL = {
    "type": "function",
    "name": "shell",
    "description": "Runs a shell command and returns its output.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"}},
            "workdir": {"type": "string"},
            "timeout_ms": {"type": "number"},
        },
        "required": ["command"],
    },
}
APPLY_PATCH = {"type": "function", "name": "apply_patch", "parameters": {"type": "object"}}


def test_find_shell_tool_responses_shape():
    tool = find_shell_tool([APPLY_PATCH, CODEX_SHELL], "responses")
    assert tool.name == "shell" and tool.properties["command"]["type"] == "array"


def test_king_messages_responses_items():
    tool = find_shell_tool([CODEX_SHELL], "responses")
    payload = {
        "instructions": "You are Codex. Keep answers short.",
        "input": [
            {"type": "message", "role": "developer", "content": "Project rule: use pytest."},
            {"role": "user", "content": [{"type": "input_text", "text": "Create hello.py"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shell",
                "arguments": json.dumps({"command": ["bash", "-lc", "ls"], "workdir": "/p"}),
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "a.py\nb.py\n"},
            {"type": "reasoning", "summary": []},
        ],
    }
    msgs = king_messages(payload, tool, "responses")
    assert msgs[0] == {"role": "system", "content": "You are Codex. Keep answers short."}
    assert msgs[1] == {"role": "system", "content": "Project rule: use pytest."}
    assert msgs[2]["role"] == "system" and "`shell`" in msgs[2]["content"]
    assert msgs[3] == {"role": "user", "content": "Create hello.py"}
    assert msgs[4] == {"role": "assistant", "content": "```bash\nls\n```"}
    assert msgs[5] == {
        "role": "user",
        "content": "<returncode>0</returncode>\n<output>\na.py\nb.py\n</output>",
    }
    assert len(msgs) == 6

    plain = king_messages({"input": "just text"}, tool, "responses")
    assert plain[-1] == {"role": "user", "content": "just text"}


def test_agent_reply_responses_function_call_and_stream():
    tool = find_shell_tool([CODEX_SHELL], "responses")
    body, events = agent_reply(
        "responses",
        "resp_1",
        "albedo-king-agent",
        "Listing.\n\n```bash\nls -la\n```",
        tool,
        {"prompt_tokens": 7, "completion_tokens": 3},
    )
    assert body["object"] == "response" and body["status"] == "completed"
    msg, call = body["output"]
    assert msg["type"] == "message" and msg["content"][0] == {
        "type": "output_text",
        "text": "Listing.",
        "annotations": [],
    }
    assert (
        call["type"] == "function_call"
        and call["name"] == "shell"
        and call["call_id"].startswith("call_")
    )
    assert json.loads(call["arguments"]) == {"command": ["bash", "-lc", "ls -la"]}
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}

    text = "".join(events)
    kinds = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert "response.output_item.added" in kinds and "response.output_item.done" in kinds
    assert (
        "response.function_call_arguments.done" in kinds and "response.output_text.delta" in kinds
    )
    done = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")][-1]
    assert done["type"] == "response.completed" and done["response"]["output"] == body["output"]

    body, _ = agent_reply("responses", "resp_2", "m", "No command needed.", tool, None)
    assert body["output"] == [
        {
            "type": "message",
            "id": body["output"][0]["id"],
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "No command needed.", "annotations": []}],
        }
    ]
