from __future__ import annotations

import json
import re

from agent.protocols.parse import _CLOSE_RE, _OPEN_RE, _SHELL_LANGS, _THINK_RE, parse_reply

NON_SHELL_BLOCK_TEMPLATE = (
    "NOTE FROM THE HARNESS: your reply contained a ```{lang} code block, but NOTHING was executed - "
    "only ONE ```bash or ```powershell block is run, and shell commands are the only tool (there is "
    "no file editor). To create or edit a file, write it from the shell: PowerShell "
    "`Set-Content -Path <file> -Encoding utf8 -Value @'\n...\n'@`, or bash `cat > <file> <<'EOF' ... EOF`. "
    "Reply now with a short explanation and exactly one shell block that performs the next step."
)
_PREAMBLE_RE = re.compile(
    r"\A\s*(I'll|I will|Let me|Let's|I need to|I'm going to|I am going to|First,? I|Now I|Next,? I|"
    r"I apologi[sz]e|Sorry|The (?:previous|last) command)\b",
    re.IGNORECASE,
)
_PREAMBLE_MAX_CHARS = 1_500
_INTENT_RE = re.compile(
    r"\b(I'll|I will|Let me|Let's|I need to|I'm going to|I am going to|Now I|Next,? I|First,? I)\b",
    re.IGNORECASE,
)
_NOT_INTENT_RE = re.compile(r"\blet me know\b", re.IGNORECASE)
_TOOL_ATTEMPT_RE = re.compile(
    r"<\s*/?\s*(invoke|function_calls?|tool_call|antml:[a-z_]+|Explore|Agent|Task|Read|Edit|Write|"
    r"Grep|Glob|Bash|parameter|arguments?)\b[^>]*>",
    re.IGNORECASE,
)


LOOP_NUDGE_TEMPLATE = (
    "<returncode>0</returncode>\n<output>\nNOTE FROM THE HARNESS: the command you just proposed has "
    "already been executed {n} times in a row with identical output. Repeating it cannot change "
    "anything. Explain what you actually expected to change, then take a DIFFERENT action (inspect "
    "the raw bytes, use a different tool or approach, or report the result to the user).\n</output>"
)
LOOP_STOP_TEMPLATE = (
    "⚠️ The king proposed the same command for the {n}th time in a row:\n\n```\n{command}\n```\n\n"
    "Stopping this turn to avoid an endless loop. Tell it what to do differently, or do that step "
    "yourself."
)


REASONING_NUDGE_TEMPLATE = (
    "<returncode>0</returncode>\n<output>\nNOTE FROM THE HARNESS: your last {n} messages repeated the same "
    "reasoning text while issuing similar commands. You are re-exploring without making progress. Stop "
    "re-reading: summarize what you already learned and take the next concrete step now (if you still need "
    "more files, read them all in ONE command).\n</output>"
)


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def trailing_repeated_thoughts(messages: list[dict]) -> int:
    last: str | None = None
    count = 0
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        thought, cmds = parse_reply(m.get("content") or "")
        key = _norm(thought)
        if not cmds or len(key) < 40 or (last is not None and key != last):
            break
        last = key
        count += 1
    return count


def trailing_repeats(messages: list[dict]) -> tuple[int, str | None]:
    last: str | None = None
    count = 0
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        cmds = parse_reply(m.get("content") or "")[1]
        cmd = cmds[0] if len(cmds) == 1 else None
        if cmd is None or (last is not None and cmd != last):
            break
        last = cmd
        count += 1
    return count, last


def looks_like_tool_attempt(text: str) -> bool:
    return bool(_TOOL_ATTEMPT_RE.search(text or ""))


def looks_like_preamble(text: str) -> bool:
    body = (text or "").strip()
    if not body or len(body) > _PREAMBLE_MAX_CHARS or "```" in body:
        return False
    if _NOT_INTENT_RE.search(body[-250:]):
        return False
    if _PREAMBLE_RE.match(body):
        return True
    if len(_INTENT_RE.findall(body)) >= 2:
        return True
    return bool(_INTENT_RE.search(body[-250:]))


def needs_command_retry(text: str) -> bool:
    return looks_like_tool_attempt(text) or looks_like_preamble(text)


def collapse_repeats(text: str) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        key = " ".join(para.split())
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(para)
    return "\n\n".join(kept)


def json_schema_of(payload: dict) -> dict | None:
    fmt = (
        (payload.get("text") or {}).get("format") if isinstance(payload.get("text"), dict) else None
    )
    if (
        isinstance(fmt, dict)
        and fmt.get("type") == "json_schema"
        and isinstance(fmt.get("schema"), dict)
    ):
        return fmt["schema"]
    return None


def coerce_to_schema(text: str, schema: dict) -> str:
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = [k for k in schema.get("required") or [] if isinstance(k, str)] or list(props)
    body = _THINK_RE.sub("", text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.DOTALL)
    for candidate in (fenced.group(1) if fenced else None, body):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict) and all(k in data for k in required):
            return json.dumps({k: data[k] for k in (required or data)}, ensure_ascii=False)
    label = re.compile(
        r"^\**(?:" + "|".join(map(re.escape, required)) + r")\**\s*:\s*", re.IGNORECASE
    )
    lines = [
        label.sub("", re.sub(r"^[#*\-\d.)\s]+", "", ln.strip()))
        for ln in body.splitlines()
        if ln.strip() and not ln.startswith("```")
    ]
    lines = [ln for ln in lines if ln]
    first = lines[0] if lines else "Task"
    out: dict = {}
    for key in required:
        spec = props.get(key) if isinstance(props.get(key), dict) else {}
        if spec.get("type") not in (None, "string"):
            continue
        value = first if key == required[0] else " ".join(lines[1:]) or first
        limit = spec.get("maxLength")
        if isinstance(limit, int) and len(value) > limit:
            value = value[: limit - 1].rstrip() + "…"
        out[key] = value or first
    return json.dumps(out, ensure_ascii=False)


def non_shell_block_lang(text: str) -> str | None:
    body = _THINK_RE.sub("", text or "")
    if parse_reply(body)[1]:
        return None
    for opener in _OPEN_RE.finditer(body):
        lang = opener.group(1).lower()
        if lang and lang not in _SHELL_LANGS and _CLOSE_RE.search(body, opener.end()):
            return lang
    return None
