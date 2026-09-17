from __future__ import annotations

import re

COMPLETE_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
RC_MARKER = "__albedo_rc="
RC_SUFFIX = f"\necho {RC_MARKER}$?"
DONE_TEXT = "👑 Done. Review the changes in your working tree."
OUTPUT_HEAD = 5_000
OUTPUT_TAIL = 5_000

_OPEN_RE = re.compile(r"```([a-zA-Z0-9_+.-]*)[ \t]*$", re.MULTILINE)
_SHELL_LANGS = {
    "",
    "bash",
    "sh",
    "shell",
    "zsh",
    "powershell",
    "pwsh",
    "ps1",
    "console",
    "cmd",
    "bat",
}
_CLOSE_RE = re.compile(r"^```[ \t]*$", re.MULTILINE)
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
_TAG_RE = re.compile(r"<mswea_bash_command>(.*?)</mswea_bash_command>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_STRAY_THINK_CLOSE = re.compile(r"\A\s*</think>")
_THOUGHT_PREFIX = re.compile(r"\A\s*THOUGHT:\s*", re.IGNORECASE)
_LEADING_COMMENTS = re.compile(r"\A(?:[ \t]*#[^\n]*(?:\n|\Z))+")
_RC_LINE = re.compile(rf"^{RC_MARKER}(\S*)\s*$", re.MULTILINE)
_EXIT_CODE_RE = re.compile(r"exit(?:ed)?(?: with)? code[:= ]+(-?\d+)", re.IGNORECASE)


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            seg["text"]
            for seg in content
            if isinstance(seg, dict) and isinstance(seg.get("text"), str)
        )
    return ""


def _fenced_blocks(body: str) -> tuple[list[str], int]:
    blocks: list[str] = []
    first = -1
    pos = 0
    while True:
        opener = _OPEN_RE.search(body, pos)
        if opener is None:
            break
        start = opener.end() + 1
        close = _CLOSE_RE.search(body, start)
        if close is None:
            break
        for hd in _HEREDOC_RE.finditer(body, start, close.start()):
            term = re.compile(rf"^{re.escape(hd.group(1))}[ \t]*$", re.MULTILINE)
            t = term.search(body, hd.end())
            if t is not None and t.start() > close.start():
                close = _CLOSE_RE.search(body, t.end())
                if close is None:
                    return blocks, first
        if opener.group(1).lower() in _SHELL_LANGS:
            if first < 0:
                first = opener.start()
            blocks.append(body[start : close.start()])
        pos = close.end()
    return blocks, first


def parse_reply(text: str) -> tuple[str, list[str]]:
    body = _STRAY_THINK_CLOSE.sub("", _THINK_RE.sub("", text or ""))
    tagged = list(_TAG_RE.finditer(body))
    if tagged:
        raw, first = [m.group(1) for m in tagged], tagged[0].start()
    else:
        raw, first = _fenced_blocks(body)
    commands = []
    for block in raw:
        cmd = _LEADING_COMMENTS.sub("", block).strip() or block.strip()
        if cmd:
            commands.append(cmd)
    head = body[:first] if commands and first >= 0 else body
    return _THOUGHT_PREFIX.sub("", head).strip(), commands


def split_rc(output: str) -> tuple[int, str]:
    output = output or ""
    last = None
    for last in _RC_LINE.finditer(output):
        pass
    if last is not None:
        raw = last.group(1).strip().lower()
        rc = (
            0 if raw in ("0", "true") else 1 if raw == "false" else int(raw) if raw.isdigit() else 1
        )
        return rc, (output[: last.start()] + output[last.end() :]).rstrip()
    m = None
    for m in _EXIT_CODE_RE.finditer(output):
        pass
    if m is not None:
        return int(m.group(1)), output.rstrip()
    return 0, output.rstrip()


def render_observation(rc: int, output: str) -> str:
    if len(output) > OUTPUT_HEAD + OUTPUT_TAIL:
        cut = len(output) - OUTPUT_HEAD - OUTPUT_TAIL
        output = f"{output[:OUTPUT_HEAD]}\n<Elided {cut} characters.>\n{output[-OUTPUT_TAIL:]}"
    return f"<returncode>{rc}</returncode>\n<output>\n{output}\n</output>"


def _responses_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            seg["text"]
            for seg in content
            if isinstance(seg, dict) and isinstance(seg.get("text"), str)
        )
    return ""
