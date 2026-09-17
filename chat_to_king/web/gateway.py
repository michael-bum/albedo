from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from common.engine_client import EngineState
from common.king_meta import MODEL_ID_PREFIX, KingInfo, KingInfoSource
from common.notices import loading_text, openai_notice
from common.token_meter import TokenMeter
from common.usage import TAIL_BYTES, usage_from_bytes
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from web.config import KingChatSettings

if TYPE_CHECKING:
    from common.engine_client import EngineClient

STATUS_HEADER = "x-albedo-status"


def _message_text(messages: list) -> str:
    parts: list[str] = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for seg in c:
                if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                    parts.append(seg["text"])
    return "\n".join(parts).lower()


def _mentions_albedo(messages: list, keywords: list[str]) -> bool:
    text = _message_text(messages)
    return any(kw in text for kw in keywords)


def _build_knowledge(doc: str, info: KingInfo | None) -> str:
    who = (
        f"You are King {info.roman} (`{info.model_id}`), the model currently crowned on Albedo"
        if info
        else "You are `albedo-king`, the model currently crowned on Albedo"
    )
    return (
        f"{who} (Bittensor subnet SN97). "
        "The user is asking about Albedo. Use the following reference document as authoritative "
        "knowledge about Albedo — its subnet mechanics, mining, validation, scoring, and architecture. "  # noqa: E501
        "Prefer it over your own assumptions; if it doesn't cover something, say so.\n\n"
        "<BEGIN ALBEDO REFERENCE (llms.txt)>\n" + doc + "\n<END ALBEDO REFERENCE>"
    )


def _inject_knowledge(messages: list, block: str) -> list:
    msgs = list(messages)
    if (
        msgs
        and isinstance(msgs[0], dict)
        and msgs[0].get("role") == "system"
        and isinstance(msgs[0].get("content"), str)
    ):
        msgs[0] = {**msgs[0], "content": block + "\n\n" + msgs[0]["content"]}
    else:
        msgs = [{"role": "system", "content": block}, *msgs]
    return msgs


def public_model_id(settings: KingChatSettings, info: KingInfo | None) -> str:
    return info.model_id if info and info.model_id else settings.served_model_name


def create_app(
    settings: KingChatSettings,
    engine: EngineClient,
    king: KingInfoSource,
    meter: TokenMeter | None = None,
) -> FastAPI:
    meter = meter or TokenMeter(settings.database_url, "chat")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0)
        )
        app.state.albedo_keywords = [
            k.strip().lower() for k in settings.llms_keywords.split(",") if k.strip()
        ]
        doc: str | None = None
        try:
            p = Path(settings.llms_path)
            if p.is_file():
                doc = p.read_text(encoding="utf-8")
            elif settings.llms_url:
                r = await app.state.client.get(settings.llms_url)
                if r.status_code == 200:
                    doc = r.text
                else:
                    logger.warning("[king-chat] llms_url HTTP {}", r.status_code)
        except Exception as exc:
            logger.warning("[king-chat] llms.txt load failed: {}", exc)
        if doc and settings.llms_max_chars and len(doc) > settings.llms_max_chars:
            doc = doc[: settings.llms_max_chars]
        app.state.albedo_doc = doc
        if doc:
            logger.info(
                "[king-chat] loaded llms.txt ({:.1f} KB) — Albedo knowledge gated on {}",
                len(doc) / 1024,
                app.state.albedo_keywords,
            )
        else:
            logger.warning("[king-chat] no llms.txt loaded; Albedo knowledge injection disabled")
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(
        title="Albedo King Chat Gateway",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def _notice(stream: bool, state: str):
        info = king.get()
        resp = openai_notice(loading_text(info), public_model_id(settings, info), stream)
        resp.headers[STATUS_HEADER] = state
        resp.headers["retry-after"] = str(settings.retry_after_s)
        return resp

    async def _proxy_or_notice(request: Request, path: str):
        raw = await request.body()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        stream = bool(payload.get("stream"))

        state = (await engine.state()).value
        if state != EngineState.SERVING.value:
            return _notice(stream, state)

        body = raw
        changed = False
        model = payload.get("model")
        if isinstance(model, str) and model.startswith(MODEL_ID_PREFIX):
            payload["model"] = settings.served_model_name
            changed = True
        if stream and meter.enabled and path == "/v1/chat/completions":
            payload["stream_options"] = {
                **(payload.get("stream_options") or {}),
                "include_usage": True,
            }
            changed = True
        if path == "/v1/chat/completions" and app.state.albedo_doc:
            msgs = payload.get("messages")
            if isinstance(msgs, list) and _mentions_albedo(msgs, app.state.albedo_keywords):
                block = _build_knowledge(app.state.albedo_doc, king.get())
                payload["messages"] = _inject_knowledge(msgs, block)
                changed = True
        if changed:
            body = json.dumps(payload).encode("utf-8")

        client: httpx.AsyncClient = app.state.client
        req = client.build_request(
            "POST",
            f"{engine.base_url}{path}",
            content=body,
            headers={"content-type": "application/json"},
        )
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError:
            return _notice(stream, EngineState.LOADING.value)

        async def _relay() -> AsyncIterator[bytes]:
            tail = b""
            try:
                async for chunk in upstream.aiter_raw():
                    tail = (tail + chunk)[-TAIL_BYTES:]
                    yield chunk
            finally:
                await upstream.aclose()
                if upstream.status_code == 200:
                    prompt_tokens, completion_tokens = usage_from_bytes(tail)
                    if prompt_tokens or completion_tokens:
                        meter.record(prompt_tokens, completion_tokens, stream)

        return StreamingResponse(
            _relay(),
            status_code=upstream.status_code,
            headers={
                "content-type": upstream.headers.get("content-type", "application/json"),
                STATUS_HEADER: "serving",
            },
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        info = king.get()
        return {
            "status": "ok",
            "state": (await engine.state()).value,
            "king": info.public() if info else None,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        info = king.get()
        return {
            "object": "list",
            "data": [
                {
                    "id": public_model_id(settings, info),
                    "object": "model",
                    "created": 0,
                    "owned_by": "albedo",
                    "king": info.public() if info else None,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _proxy_or_notice(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy_or_notice(request, "/v1/completions")

    return app
