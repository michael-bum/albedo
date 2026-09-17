from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.protocols import (  # noqa: E402
    COMPLETE_MARKER,
    DONE_TEXT,
    RC_SUFFIX,
    agent_reply,
    build_args,
    find_shell_tool,
    king_messages,
    split_rc,
    tool_hint,
)

COPILOT_POWERSHELL = {
    "type": "function",
    "function": {
        "name": "powershell",
        "description": "Runs a PowerShell command. Avoid PowerShell 7-only syntax such as &&.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "mode": {"type": "string", "enum": ["sync", "async"]},
            },
            "required": ["command", "description"],
        },
    },
}
COPILOT_READ = {"type": "function", "function": {"name": "read_powershell", "parameters": {}}}
CODEX_SHELL = {
    "type": "function",
    "function": {
        "name": "shell",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}},
                "workdir": {"type": "string"},
                "timeout_ms": {"type": "number"},
            },
            "required": ["command"],
        },
    },
}
CLAUDE_BASH = {
    "name": "Bash",
    "description": "Executes a bash command and returns its output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout": {"type": "number"},
            "run_in_background": {"type": "boolean"},
        },
        "required": ["command"],
    },
}
CLAUDE_READ = {"name": "Read", "input_schema": {"type": "object", "properties": {}}}


def test_find_shell_tool_openai_and_anthropic():
    assert find_shell_tool([COPILOT_READ, COPILOT_POWERSHELL]).name == "powershell"
    assert find_shell_tool([CODEX_SHELL]).name == "shell"
    assert find_shell_tool([CLAUDE_READ, CLAUDE_BASH], "anthropic").name == "Bash"
    odd = {"type": "function", "function": {"name": "my_terminal_exec", "parameters": {}}}
    assert find_shell_tool([COPILOT_READ, odd]).name == "my_terminal_exec"
    assert find_shell_tool([COPILOT_READ]) is None
    assert find_shell_tool(None) is None


def test_build_args_schema_driven():
    ps = find_shell_tool([COPILOT_POWERSHELL])
    assert build_args(ps, "Get-ChildItem", "list files", False) == {
        "command": "Get-ChildItem",
        "description": "list files",
    }
    assert build_args(ps, "ls", "x", True)["command"] == "ls" + RC_SUFFIX
    codex = find_shell_tool([CODEX_SHELL])
    assert build_args(codex, "ls -la", "look", False) == {"command": ["bash", "-lc", "ls -la"]}
    bash = find_shell_tool([CLAUDE_BASH], "anthropic")
    assert build_args(bash, "pytest -q", "run tests", False) == {
        "command": "pytest -q",
        "run_in_background": False,
    }


def test_split_rc_marker_and_exit_code_text():
    assert split_rc("out\n__albedo_rc=0") == (0, "out")
    assert split_rc("err\n__albedo_rc=False") == (1, "err")
    assert split_rc("boom\n<shellId: 0 completed with exit code 1>") == (
        1,
        "boom\n<shellId: 0 completed with exit code 1>",
    )
    assert split_rc("Command exited with code 3") == (3, "Command exited with code 3")
    assert split_rc("plain") == (0, "plain")


def test_king_messages_openai_passthrough_and_hint():
    tool = find_shell_tool([COPILOT_POWERSHELL])
    call_args = json.dumps({"command": "Get-ChildItem", "description": "list"})
    payload = {
        "messages": [
            {"role": "system", "content": "You are Copilot. User rule: answer in Polish."},
            {"role": "user", "content": "<tagged_files>a.md</tagged_files>\nhi"},
            {
                "role": "assistant",
                "content": "Listing files.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "powershell", "arguments": call_args},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "a.md\n<shellId: 0 completed with exit code 0>",
            },
            {"role": "user", "content": [{"type": "text", "text": "now count them"}]},
        ]
    }
    msgs = king_messages(payload, tool, "openai")
    assert msgs[0] == {"role": "system", "content": "You are Copilot. User rule: answer in Polish."}
    assert msgs[1]["role"] == "system" and msgs[1]["content"] == tool_hint(tool, powershell=True)
    assert "`powershell`" in msgs[1]["content"] and "PowerShell 7-only" in msgs[1]["content"]
    assert "Windows PowerShell" in msgs[1]["content"]
    assert msgs[2] == {"role": "user", "content": "<tagged_files>a.md</tagged_files>\nhi"}
    assert msgs[3] == {
        "role": "assistant",
        "content": "Listing files.\n\n```powershell\nGet-ChildItem\n```",
    }
    assert msgs[4]["role"] == "user" and msgs[4]["content"].startswith("<returncode>0</returncode>")
    assert msgs[5] == {"role": "user", "content": "now count them"}


def test_king_messages_anthropic_blocks():
    tool = find_shell_tool([CLAUDE_BASH], "anthropic")
    payload = {
        "system": [
            {"type": "text", "text": "You are Claude Code."},
            {"type": "text", "text": " Be brief."},
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "create hello.txt"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Creating it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "echo hi > hello.txt", "description": "create"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": ""}],
                    },
                    {"type": "text", "text": "<system-reminder>ok</system-reminder>"},
                ],
            },
        ],
    }
    msgs = king_messages(payload, tool, "anthropic")
    assert msgs[0] == {"role": "system", "content": "You are Claude Code. Be brief."}
    assert msgs[1]["role"] == "system" and "`Bash`" in msgs[1]["content"]
    assert msgs[2] == {"role": "user", "content": "create hello.txt"}
    assert msgs[3] == {
        "role": "assistant",
        "content": "Creating it.\n\n```bash\necho hi > hello.txt\n```",
    }
    assert msgs[4] == {
        "role": "user",
        "content": "<returncode>0</returncode>\n<output>\n\n</output>",
    }
    assert msgs[5] == {"role": "user", "content": "<system-reminder>ok</system-reminder>"}


def test_agent_reply_openai_call_done_text():
    tool = find_shell_tool([COPILOT_POWERSHELL])
    body, events = agent_reply(
        "openai",
        "r1",
        "albedo-king-agent",
        "Checking.\n\n```powershell\nGet-Date\n```",
        tool,
        {"prompt_tokens": 5, "completion_tokens": 2},
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] == "Checking."
    args = json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"command": "Get-Date", "description": "Checking."}
    chunks = list(events)
    assert chunks[-1] == "data: [DONE]\n\n"
    assert '"finish_reason": "tool_calls"' in "".join(chunks)

    body, _ = agent_reply(
        "openai", "r2", "m", f"THOUGHT: all good\n```bash\necho {COMPLETE_MARKER}\n```", tool, None
    )
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == f"all good\n\n{DONE_TEXT}"

    body, _ = agent_reply(
        "openai", "r3", "m", "Two blocks\n```bash\nls\n```\n```bash\npwd\n```", tool, None
    )
    assert (
        body["choices"][0]["finish_reason"] == "stop"
        and "tool_calls" not in body["choices"][0]["message"]
    )


def test_agent_reply_anthropic_tool_use_and_stream():
    tool = find_shell_tool([CLAUDE_BASH], "anthropic")
    body, events = agent_reply(
        "anthropic",
        "m1",
        "albedo-king-agent",
        "I'll list files.\n\n```bash\nls -la\n```",
        tool,
        {"prompt_tokens": 9, "completion_tokens": 4},
    )
    assert body["type"] == "message" and body["stop_reason"] == "tool_use"
    assert body["content"][0] == {"type": "text", "text": "I'll list files."}
    use = body["content"][1]
    assert use["type"] == "tool_use" and use["name"] == "Bash" and use["id"].startswith("toolu_")
    assert use["input"] == {"command": "ls -la", "run_in_background": False}
    assert body["usage"] == {"input_tokens": 9, "output_tokens": 4}
    text = "".join(events)
    kinds = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert kinds == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert '"input_json_delta"' in text and '"stop_reason": "tool_use"' in text

    body, _ = agent_reply("anthropic", "m2", "m", "Nothing to run.", tool, None)
    assert body["stop_reason"] == "end_turn" and body["content"] == [
        {"type": "text", "text": "Nothing to run."}
    ]


def test_codex_exec_command_tool_uses_cmd_key():
    exec_tool = {
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "workdir": {"type": "string"},
                    "yield_time_ms": {"type": "number"},
                },
                "required": ["cmd"],
            },
        },
    }
    tool = find_shell_tool([exec_tool])
    assert tool.name == "exec_command"
    assert build_args(tool, "ls -la", "look", False) == {"cmd": "ls -la"}


def test_parse_reply_heredoc_with_nested_fences_is_one_command():
    from agent.protocols import parse_reply

    text = (
        "I'll create the note.\n\n```bash\ncat > PostgreSQL.md << 'EOF'\n# PostgreSQL\n\n"
        "```sql\nSELECT 1;\n```\n\n```bash\npsql -U postgres\n```\nEOF\n```\n"
    )
    thought, cmds = parse_reply(text)
    assert thought == "I'll create the note."
    assert len(cmds) == 1 and cmds[0].startswith("cat > PostgreSQL.md << 'EOF'")
    assert cmds[0].endswith("psql -U postgres\n```\nEOF")

    thought, cmds = parse_reply("Two.\n```bash\nls\n```\ntext\n```sh\npwd\n```")
    assert cmds == ["ls", "pwd"]
    thought, cmds = parse_reply("No command here.")
    assert cmds == [] and thought == "No command here."


def test_anthropic_system_messages_inside_list_pass_through():
    tool = find_shell_tool([CLAUDE_BASH], "anthropic")
    payload = {
        "system": "You are Claude Code.",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "system",
                "content": [{"type": "text", "text": "# Environment\n- Shell: bash"}],
            },
        ],
    }
    msgs = king_messages(payload, tool, "anthropic")
    assert msgs[-1] == {"role": "system", "content": "# Environment\n- Shell: bash"}


def test_looks_like_tool_attempt():
    from agent.protocols import looks_like_tool_attempt

    assert looks_like_tool_attempt("<Explore> <breadth>medium</breadth> </Explore>")
    assert looks_like_tool_attempt('<invoke name="Bash"><parameter name="command">ls</parameter>')
    assert looks_like_tool_attempt("<tool_call>{}</tool_call>")
    assert not looks_like_tool_attempt(
        "Here is a <b>bold</b> answer with <returncode>0</returncode>."
    )
    assert not looks_like_tool_attempt("Plain text, done.")


def test_only_shell_fences_are_commands():
    from agent.protocols import parse_reply

    text = "Here is the program:\n```cpp\nint main() { return 0; }\n```\nNow compile it:\n```bash\ng++ a.cpp -o a && ./a\n```"
    thought, cmds = parse_reply(text)
    assert cmds == ["g++ a.cpp -o a && ./a"]
    assert parse_reply("Just code:\n```python\nprint(1)\n```")[1] == []
    assert parse_reply("```powershell\nGet-ChildItem\n```")[1] == ["Get-ChildItem"]
    assert parse_reply("```\nls\n```")[1] == ["ls"]
    assert parse_reply("```sql\nSELECT 1;\n```\n```\necho after\n```")[1] == ["echo after"]


def test_preamble_without_command_needs_retry():
    from agent.protocols import needs_command_retry

    assert needs_command_retry(
        "I'll check the existing MD files in the repo to see what needs filling."
    )
    assert needs_command_retry("Let me look at the project structure first.")
    assert needs_command_retry("<Explore> <search>x</search> </Explore>")
    assert not needs_command_retry("There are 80 markdown files in the repo.")
    assert not needs_command_retry("The file was created and printed successfully. Task complete.")
    assert not needs_command_retry("I'll " + "x" * 2000)


def test_fence_glued_to_sentence_is_parsed():
    from agent.protocols import parse_reply

    thought, cmds = parse_reply("I'll check the files.```bash\nGet-ChildItem -Recurse\n```")
    assert thought == "I'll check the files." and cmds == ["Get-ChildItem -Recurse"]


def test_trailing_repeats_counts_identical_commands():
    from agent.protocols import trailing_repeats

    hist = [
        {"role": "assistant", "content": "a\n```bash\nls\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\nx\n</output>"},
        {"role": "assistant", "content": "b\n```bash\ncat f\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\nx\n</output>"},
        {"role": "assistant", "content": "c\n```bash\ncat f\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\nx\n</output>"},
        {"role": "user", "content": "go on"},
    ]
    assert trailing_repeats(hist) == (2, "cat f")
    assert trailing_repeats(hist[:2]) == (1, "ls")
    assert trailing_repeats([{"role": "assistant", "content": "done, no command"}]) == (0, None)


def test_trailing_repeated_thoughts_detects_same_reasoning_with_different_commands():
    from agent.protocols import trailing_repeated_thoughts

    same = "I see the pattern now. The subnet files all have the same template structure, mostly empty."
    hist = []
    for f in ("a", "b", "c"):
        hist.append({"role": "assistant", "content": f"{same}\n\n```bash\ncat {f}.md\n```"})
        hist.append(
            {"role": "user", "content": "<returncode>0</returncode>\n<output>\nx\n</output>"}
        )
    assert trailing_repeated_thoughts(hist) == 3
    hist.append(
        {
            "role": "assistant",
            "content": "Now I will write the summary file with everything I found so far.\n\n```bash\necho hi > s.md\n```",
        }
    )
    assert trailing_repeated_thoughts(hist) == 1
    assert (
        trailing_repeated_thoughts([{"role": "assistant", "content": "ok\n```bash\nls\n```"}]) == 0
    )


def test_preamble_with_intent_in_last_sentence():
    from agent.protocols import looks_like_preamble

    assert looks_like_preamble(
        "The HttpListener requires admin rights on Windows. I'll use a TcpListener-based approach "
        "instead, which works without elevation. Let me write a static file server script."
    )
    assert not looks_like_preamble(
        "The server is running on port 8082. Let me know if you need anything else."
    )
    assert not looks_like_preamble("Done. The file was created and verified.")


def test_collapse_repeats_and_non_shell_block_lang():
    from agent.protocols import collapse_repeats, non_shell_block_lang

    text = "A.\n\nB.\n\nA.\n\nA.\n\nC."
    assert collapse_repeats(text) == "A.\n\nB.\n\nC."
    assert non_shell_block_lang("plan\n\n```python\nprint(1)\n```\n") == "python"
    assert non_shell_block_lang("```python\nprint(1)\n```\n\n```bash\nls\n```") is None
    assert non_shell_block_lang("just text") is None
    assert non_shell_block_lang("```\nls\n```") is None


def test_coerce_to_schema():
    from agent.protocols import coerce_to_schema

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "maxLength": 36},
            "description": {"type": "string"},
        },
        "required": ["title", "description"],
    }
    prose = "Improve website styling\n\nLocate website styles in C:\\x\\website and tidy the CSS"
    data = json.loads(coerce_to_schema(prose, schema))
    assert data == {
        "title": "Improve website styling",
        "description": "Locate website styles in C:\\x\\website and tidy the CSS",
    }
    long = "A" * 60
    assert len(json.loads(coerce_to_schema(long, schema))["title"]) == 36
    good = '```json\n{"title": "T", "description": "D", "extra": 1}\n```'
    assert json.loads(coerce_to_schema(good, schema)) == {"title": "T", "description": "D"}
    assert json.loads(coerce_to_schema("", schema))["title"] == "Task"
    labelled = "Title: Enhance Website Styling\nDescription: Improve the CSS."
    assert json.loads(coerce_to_schema(labelled, schema)) == {
        "title": "Enhance Website Styling",
        "description": "Improve the CSS.",
    }


def test_preamble_detection_after_collapse():
    from agent.protocols import looks_like_preamble

    long_intent = (
        "I apologize for the trouble with the API commands. Since I don't have a working key, "
        "let me use the browser tools directly to check the weather for you.\n\n"
        "I'll navigate to a weather site and take a screenshot.\n\n"
        "I'll check San Francisco weather via the browser now.\n\n"
        "I'll use the browser to look up the conditions and show you a screenshot.\n\n"
        "I'll navigate to weather.com and take a screenshot of the current conditions."
    )
    assert looks_like_preamble(long_intent)
    answer = (
        "There are two HTML files in the folder: index.html and about.html. Both link to style.css. "
        "Let me know if you want me to change anything."
    )
    assert not looks_like_preamble(answer)
    assert not looks_like_preamble("Plan:\n\n```python\nprint(1)\n```")
    assert not looks_like_preamble("x" * 2000)
