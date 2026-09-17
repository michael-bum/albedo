from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from agent.api.limiter import STATUS_HEADER
from agent.protocols import (
    LOOP_NUDGE_TEMPLATE,
    LOOP_STOP_TEMPLATE,
    NON_SHELL_BLOCK_TEMPLATE,
    REASONING_NUDGE_TEMPLATE,
    agent_reply,
    coerce_to_schema,
    collapse_repeats,
    json_schema_of,
    king_messages,
    looks_like_preamble,
    needs_command_retry,
    non_shell_block_lang,
    parse_reply,
    shell_lang,
    trailing_repeated_thoughts,
    trailing_repeats,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger

if TYPE_CHECKING:
    from agent.config import KingAgentSettings
    from agent.devtools.trace import Trace

_CLIENT_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed")


async def agent_step(
    payload: dict,
    tool,
    stream: bool,
    record,
    kind: str,
    tr: dict,
    *,
    settings: KingAgentSettings,
    client: httpx.AsyncClient,
    engine_url: str,
    trace: Trace,
    notice,
    sampling_defaults: dict,
):
    king_msgs = king_messages(payload, tool, kind, settings.agent_rc_marker)
    shell = shell_lang(payload, tool)
    repeats, repeated_cmd = trailing_repeats(king_msgs) if tool is not None else (0, None)
    if repeats >= settings.agent_dup_nudge:
        logger.info("[king-agent] same command {}x in a row; adding loop note", repeats)
        tr["notes"].append(f"loop note after {repeats} identical commands")
        king_msgs = king_msgs + [{"role": "user", "content": LOOP_NUDGE_TEMPLATE.format(n=repeats)}]
    elif tool is not None:
        same_thoughts = trailing_repeated_thoughts(king_msgs)
        if same_thoughts >= settings.agent_thought_nudge:
            logger.info("[king-agent] same reasoning {}x in a row; adding note", same_thoughts)
            tr["notes"].append(f"reasoning note after {same_thoughts} identical thoughts")
            king_msgs = king_msgs + [
                {"role": "user", "content": REASONING_NUDGE_TEMPLATE.format(n=same_thoughts)}
            ]
    sampling = dict(sampling_defaults)
    for key in _CLIENT_SAMPLING_KEYS:
        if key in payload:
            sampling[key] = payload[key]
    if repeats >= settings.agent_dup_nudge and settings.agent_loop_repetition_penalty != 1.0:
        sampling["repetition_penalty"] = settings.agent_loop_repetition_penalty
    king_payload = {
        "model": settings.served_model_name,
        "messages": king_msgs,
        "max_tokens": payload["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": False},
        **sampling,
    }
    headers = {STATUS_HEADER: "serving"}
    prompt_tokens = completion_tokens = 0
    text, usage, data, prefix = "", None, {}, ""
    for attempt in range(2):
        tr["king_requests"].append(king_payload)
        try:
            upstream = await client.post(
                f"{engine_url}/v1/chat/completions",
                json=king_payload,
            )
        except httpx.HTTPError as exc:
            logger.warning("[king-agent] upstream unreachable: {}", exc)
            record(200, prompt_tokens, completion_tokens)
            tr["notes"].append(f"upstream unreachable: {exc}")
            trace.write(tr)
            return notice(stream)
        if upstream.status_code != 200:
            record(upstream.status_code, prompt_tokens, completion_tokens)
            headers["content-type"] = upstream.headers.get("content-type", "application/json")
            tr["king_responses"].append({"status": upstream.status_code, "body": upstream.text})
            trace.write(tr)
            return Response(upstream.content, status_code=upstream.status_code, headers=headers)
        data = upstream.json()
        tr["king_responses"].append(data)
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        prompt_tokens += int((usage or {}).get("prompt_tokens") or 0)
        completion_tokens += int((usage or {}).get("completion_tokens") or 0)
        collapsed = collapse_repeats(text)
        if collapsed != text.strip():
            logger.info("[king-agent] collapsed repeated paragraphs in the reply")
            tr["notes"].append("collapsed repeated paragraphs")
            text = collapsed
        if tool is None or attempt > 0 or parse_reply(text)[1]:
            break
        lang = non_shell_block_lang(text)
        if lang:
            logger.info("[king-agent] reply had a ```{} block but no command; asking again", lang)
            tr["notes"].append(f"non-shell {lang} block; harness note retry")
            king_payload = {
                **king_payload,
                "messages": king_payload["messages"]
                + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": NON_SHELL_BLOCK_TEMPLATE.format(lang=lang)},
                ],
            }
            continue
        if not needs_command_retry(text):
            break
        logger.info(
            "[king-agent] reply had no command ({!r}); continuing it inside a bash block",
            text[:120],
        )
        tr["notes"].append("no command in reply; prefilled bash continuation")
        prefix = (text.strip() + "\n\n" if looks_like_preamble(text) else "") + f"```{shell}\n"
        king_payload = {
            **king_payload,
            "messages": king_payload["messages"] + [{"role": "assistant", "content": prefix}],
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
    if prefix:
        text = prefix + text
    schema = json_schema_of(payload) if tool is None else None
    if schema is not None:
        text = coerce_to_schema(text, schema)
        tr["notes"].append("json_schema output coerced")
    reply_tool = tool
    if repeats >= settings.agent_dup_stop:
        cmds = parse_reply(text)[1]
        if len(cmds) == 1 and cmds[0] == repeated_cmd:
            logger.warning(
                "[king-agent] loop persists after note ({}x); stopping turn", repeats + 1
            )
            text = LOOP_STOP_TEMPLATE.format(n=repeats + 1, command=cmds[0])
            reply_tool = None
            tr["notes"].append("loop stop: turn ended without a tool call")
    record(200, prompt_tokens, completion_tokens)
    rid = data.get("id") or "chatcmpl-king-agent"
    body, events = agent_reply(
        kind,
        rid,
        settings.agent_model,
        text,
        reply_tool,
        usage,
        settings.agent_rc_marker,
    )
    tr["final_text"] = text
    tr["reply"] = body
    trace.write(tr)
    if stream:
        return StreamingResponse(events, media_type="text/event-stream", headers=headers)
    return JSONResponse(body, headers=headers)
