from __future__ import annotations

from agent.protocols.parse import _responses_text, _text, render_observation, split_rc
from agent.protocols.tools import ShellTool, _command_from_args, shell_lang, tool_hint


def _assistant_text(content: str, command: str, lang: str = "bash") -> str:
    content = content.strip()
    block = f"```{lang}\n{command}\n```"
    return f"{content}\n\n{block}" if content else block


def _openai_history(messages: list, hint: str, lang: str = "bash") -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), _text(m.get("content"))
        if role in ("system", "developer"):
            out.append({"role": "system", "content": content})
        elif role == "user":
            if content:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            calls = m.get("tool_calls")
            if isinstance(calls, list) and calls:
                fn = calls[0].get("function") if isinstance(calls[0], dict) else None
                cmd = _command_from_args(fn.get("arguments") if isinstance(fn, dict) else None)
                out.append({"role": "assistant", "content": _assistant_text(content, cmd, lang)})
            elif content:
                out.append({"role": "assistant", "content": content})
        elif role == "tool":
            rc, text = split_rc(content)
            out.append({"role": "user", "content": render_observation(rc, text)})
    if hint:
        last_system = max((i for i, m in enumerate(out) if m["role"] == "system"), default=-1)
        out.insert(last_system + 1, {"role": "system", "content": hint})
    return out


def _anthropic_history(payload: dict, hint: str, lang: str = "bash") -> list[dict]:
    out: list[dict] = []
    system = _text(payload.get("system"))
    if system:
        out.append({"role": "system", "content": system})
    if hint:
        out.append({"role": "system", "content": hint})
    for m in payload.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if role == "system":
            text = _text(blocks)
            if text.strip():
                out.append({"role": "system", "content": text})
        elif role == "user":
            texts: list[str] = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    if texts:
                        out.append({"role": "user", "content": "\n".join(texts)})
                        texts = []
                    rc, text = split_rc(_text(b.get("content")))
                    if b.get("is_error") and rc == 0:
                        rc = 1
                    out.append({"role": "user", "content": render_observation(rc, text)})
                elif isinstance(b.get("text"), str):
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            text = "".join(
                b["text"]
                for b in blocks
                if isinstance(b, dict)
                and b.get("type") == "text"
                and isinstance(b.get("text"), str)
            )
            calls = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
            if calls:
                cmd = _command_from_args(calls[0].get("input"))
                out.append({"role": "assistant", "content": _assistant_text(text, cmd, lang)})
            elif text.strip():
                out.append({"role": "assistant", "content": text})
    return out


def _responses_history(payload: dict, hint: str, lang: str = "bash") -> list[dict]:
    out: list[dict] = []
    instructions = _responses_text(payload.get("instructions"))
    if instructions:
        out.append({"role": "system", "content": instructions})
    items = payload.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list):
        items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = it.get("type") or ("message" if it.get("role") else "")
        if kind == "message":
            role, text = it.get("role"), _responses_text(it.get("content"))
            if role in ("system", "developer"):
                out.append({"role": "system", "content": text})
            elif role == "user" and text:
                out.append({"role": "user", "content": text})
            elif role == "assistant" and text.strip():
                out.append({"role": "assistant", "content": text})
        elif kind == "function_call":
            cmd = _command_from_args(it.get("arguments"))
            out.append({"role": "assistant", "content": _assistant_text("", cmd, lang)})
        elif kind == "function_call_output":
            rc, text = split_rc(_responses_text(it.get("output")))
            out.append({"role": "user", "content": render_observation(rc, text)})
    if hint:
        last_system = max((i for i, m in enumerate(out) if m["role"] == "system"), default=-1)
        out.insert(last_system + 1, {"role": "system", "content": hint})
    return out


def king_messages(
    payload: dict, tool: ShellTool | None, kind: str, rc_marker: bool = False
) -> list[dict]:
    lang = shell_lang(payload, tool)
    hint = tool_hint(tool, lang == "powershell") if tool else ""
    if kind == "anthropic":
        return _anthropic_history(payload, hint, lang)
    if kind == "responses":
        return _responses_history(payload, hint, lang)
    return _openai_history(payload.get("messages") or [], hint, lang)
