from __future__ import annotations

from collections.abc import Iterator

from agent.protocols.anthropic_messages import _anthropic_reply
from agent.protocols.openai_chat import _openai_reply
from agent.protocols.openai_responses import _responses_reply
from agent.protocols.tools import ShellTool


def agent_reply(
    kind: str,
    rid: str,
    model: str,
    text: str,
    tool: ShellTool | None,
    usage: dict | None,
    rc_marker: bool = False,
) -> tuple[dict, Iterator[str]]:
    if kind == "anthropic":
        return _anthropic_reply(rid, model, text, tool, usage, rc_marker)
    if kind == "responses":
        return _responses_reply(rid, model, text, tool, usage, rc_marker)
    return _openai_reply(rid, model, text, tool, usage, rc_marker)
