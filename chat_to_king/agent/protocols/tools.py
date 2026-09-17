from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agent.protocols.parse import COMPLETE_MARKER, DONE_TEXT, RC_SUFFIX, parse_reply

SHELL_TOOL_NAMES = (
    "run_in_terminal",
    "run_terminal_cmd",
    "execute_command",
    "run_command",
    "exec_command",
    "shell_command",
    "Bash",
    "bash",
    "Shell",
    "shell",
    "powershell",
    "terminal",
)
_SHELL_KEYWORDS = ("terminal", "shell", "bash")
_HELPER_PREFIXES = ("read_", "stop_", "list_", "get_", "kill_", "write_", "send_")
_HINT_DESC_LIMIT = 1_500

HINT_TEMPLATE = (
    "Command execution protocol: to run a command, put exactly ONE fenced code block "
    "(```bash … ``` or ```powershell … ```) in your reply, with a short explanation before it. "
    "The client runs it with its `{name}` tool and returns the result in the next user message as "
    "<returncode>N</returncode> and <output>…</output>. The `{name}` shell is the ONLY tool available "
    "to you: any other tools, sub-agents or file editors mentioned in the instructions cannot be "
    "called, so do everything (reading, searching, creating and editing files) with shell commands, "
    "and never write XML- or JSON-style tool-call syntax. Reply without a code block only when the "
    "work is done or you just need to answer. Notes about the `{name}` tool from the client: "
    "{description}"
)
POWERSHELL_HINT = (
    " The shell is Windows PowerShell and your block is ALREADY running inside it: write cmdlets directly "
    'in a ```powershell block, never wrap them in `powershell -Command "..."` and never use bash syntax. '
    "Use Get-ChildItem, Get-Content, Set-Content, New-Item, Select-String; for HTTP use "
    "`Invoke-RestMethod -Uri '<url>'` (`curl`/`wget` are aliases of Invoke-WebRequest here); write files with "
    "here-strings (Set-Content -Path <file> -Encoding utf8 -Value @'\n...\n'@), never heredocs, no `&&`."
)
_POWERSHELL_RE = re.compile(
    r"\bShell:\s*(?:powershell|pwsh)\b|\bpowershell\.exe\b|\bWindows PowerShell\b", re.IGNORECASE
)


@dataclass(frozen=True)
class ShellTool:
    name: str
    description: str = ""
    properties: dict = field(default_factory=dict)
    required: list = field(default_factory=list)


def _tools(tools: object, kind: str) -> list[ShellTool]:
    if not isinstance(tools, list):
        return []
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if kind == "openai" else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        schema = fn.get("input_schema") if kind == "anthropic" else fn.get("parameters")
        schema = schema if isinstance(schema, dict) else {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        req = schema.get("required") if isinstance(schema.get("required"), list) else []
        out.append(ShellTool(fn["name"], fn.get("description") or "", props, req))
    return out


def find_shell_tool(tools: object, kind: str = "openai") -> ShellTool | None:
    found = _tools(tools, kind)
    by_name = {f.name: f for f in found}
    for name in SHELL_TOOL_NAMES:
        if name in by_name:
            return by_name[name]
    for f in found:
        low = f.name.lower()
        if low.startswith(_HELPER_PREFIXES):
            continue
        if any(k in low for k in _SHELL_KEYWORDS):
            return f
    return None


def tool_hint(tool: ShellTool, powershell: bool = False) -> str:
    desc = " ".join(tool.description.split())[:_HINT_DESC_LIMIT] or "(none)"
    hint = HINT_TEMPLATE.format(name=tool.name, description=desc)
    return hint + POWERSHELL_HINT if powershell else hint


def detect_powershell(payload: dict, tool: ShellTool | None) -> bool:
    if tool is not None and tool.name.lower() in ("powershell", "pwsh"):
        return True
    blob = json.dumps(payload, ensure_ascii=False)[:400_000].replace("\\n", "\n")
    return bool(_POWERSHELL_RE.search(blob))


def shell_lang(payload: dict, tool: ShellTool | None) -> str:
    return "powershell" if detect_powershell(payload, tool) else "bash"


def _command_from_args(args: object) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if not isinstance(args, dict):
        return ""
    cmd = args.get("command")
    if isinstance(cmd, list):
        strs = [c for c in cmd if isinstance(c, str)]
        cmd = strs[2] if len(strs) >= 3 and strs[1] in ("-lc", "-c") else " ".join(strs)
    if not isinstance(cmd, str):
        cmd = next((v for v in args.values() if isinstance(v, str)), "")
    return cmd.removesuffix(RC_SUFFIX)


def build_args(tool: ShellTool, command: str, thought: str, rc_marker: bool) -> dict:
    props = tool.properties
    if "command" in props or not props:
        cmd_key = "command"
    else:
        cmd_key = next(
            (
                k
                for k, v in props.items()
                if isinstance(v, dict) and v.get("type") in ("string", "array")
            ),
            "command",
        )
    spec = props.get(cmd_key) if isinstance(props.get(cmd_key), dict) else {}
    cmd = command + (RC_SUFFIX if rc_marker else "")
    args: dict[str, object] = {
        cmd_key: ["bash", "-lc", cmd] if spec.get("type") == "array" else cmd
    }
    label = " ".join((thought or "Run the next step").split())[:100]
    for key in ("description", "explanation", "reason", "summary"):
        if key in props and key != cmd_key and key in tool.required:
            args[key] = label
    for key in ("isBackground", "is_background", "run_in_background"):
        if key in props:
            args[key] = False
    for key in ("requires_approval", "require_user_approval"):
        if key in props:
            args[key] = True
    for key in tool.required:
        if key in args or key not in props:
            continue
        kind = props[key].get("type") if isinstance(props[key], dict) else None
        args[key] = False if kind == "boolean" else 0 if kind in ("integer", "number") else ""
    return args


def _decide(text: str, tool: ShellTool | None, rc_marker: bool) -> tuple[str, dict | None, str]:
    if tool is None:
        return text.strip(), None, "text"
    thought, commands = parse_reply(text)
    if len(commands) == 1 and COMPLETE_MARKER not in commands[0]:
        return thought, build_args(tool, commands[0], thought, rc_marker), "call"
    if commands and COMPLETE_MARKER in commands[0]:
        return (f"{thought}\n\n{DONE_TEXT}" if thought else DONE_TEXT), None, "done"
    return text.strip(), None, "text"
