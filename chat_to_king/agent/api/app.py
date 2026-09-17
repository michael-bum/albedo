from __future__ import annotations

import copy
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from agent.api.agent_step import agent_step
from agent.api.limiter import STATUS_HEADER, Limiter, _error
from agent.api.usage import _TAIL_BYTES, _prompt_chars, _usage_from_bytes
from agent.config import KingAgentSettings
from agent.db import ApiKey, KeyStore, client_family
from agent.devtools.probe import PROBE_REPLY, is_probe
from agent.devtools.trace import Trace, session_key
from agent.protocols import agent_reply, find_shell_tool
from common.engine_client import EngineState
from common.king_meta import KingInfoSource
from common.notices import loading_text, openai_notice
from common.token_meter import TokenMeter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from starlette.concurrency import run_in_threadpool

if TYPE_CHECKING:
    from common.engine_client import EngineClient

_OPENAI_PATHS = ("/v1/chat/completions", "/v1/completions")
_KIND_BY_PATH = {
    "/v1/chat/completions": "openai",
    "/v1/messages": "anthropic",
    "/v1/responses": "responses",
}
_STRIPPED_FIELDS = ("best_of", "prompt_logprobs", "top_logprobs")
_CHARS_PER_TOKEN = 4


def _clean_upstream_error(raw: bytes) -> bytes:
    try:
        body = json.loads(raw)
        message = body["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return raw
    if isinstance(message, str) and "validation error" in message:
        body["error"]["message"] = "Invalid request body."
        return json.dumps(body).encode("utf-8")
    return raw


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def create_ide_app(
    engine: EngineClient,
    settings: KingAgentSettings,
    store: KeyStore | None = None,
    king: KingInfoSource | None = None,
    meter: TokenMeter | None = None,
) -> FastAPI:
    king = king or KingInfoSource(settings.current_king_file)
    meter = meter or TokenMeter(settings.database_url, "agent")
    limiter = Limiter(settings.global_parallel)
    trace = Trace(settings.trace_root)
    keys = store or KeyStore(settings.database_url)
    try:
        _agent_sampling = json.loads(settings.agent_sampling or "{}")
    except json.JSONDecodeError:
        logger.warning("[king-agent] agent_sampling is not valid JSON; ignoring")
        _agent_sampling = {}
    if not isinstance(_agent_sampling, dict):
        _agent_sampling = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0)
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(
        title="Albedo King IDE API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("[king-agent] unhandled error on {}", request.url.path)
        detail = f"{type(exc).__name__}: {exc}" if settings.debug else "Internal server error."
        return _error(500, detail, "internal_error")

    async def _state() -> str:
        return (await engine.state()).value

    def _notice(stream: bool, state: str = EngineState.LOADING.value):
        resp = openai_notice(loading_text(king.get()), settings.served_model_name, stream)
        resp.headers[STATUS_HEADER] = state
        resp.headers["retry-after"] = str(settings.retry_after_s)
        return resp

    def _model_entry(model_id: str, description: str | None = None) -> dict:
        info = king.get()
        entry = {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "albedo",
            "max_model_len": settings.max_model_len,
            "king": info.public() if info else None,
        }
        if description:
            entry["description"] = description
        return entry

    async def _authorize(request: Request) -> ApiKey | JSONResponse:
        raw = _bearer(request)
        if not raw:
            return _error(401, "Missing API key.", "invalid_api_key")
        key = await run_in_threadpool(keys.lookup, raw)
        if key is None or not key.active(time.time()):
            return _error(401, "Invalid, expired or revoked API key.", "invalid_api_key")
        return key

    def _client_meta(request: Request) -> tuple[str | None, str | None]:
        ip = request.headers.get("cf-connecting-ip") or (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip() or None
        )
        return client_family(request.headers.get("user-agent")), ip

    async def _agent_step(payload: dict, tool, stream: bool, record, kind: str, tr: dict):
        return await agent_step(
            payload,
            tool,
            stream,
            record,
            kind,
            tr,
            settings=settings,
            client=app.state.client,
            engine_url=engine.base_url,
            trace=trace,
            notice=lambda stream: _notice(stream),
            sampling_defaults=_agent_sampling,
        )

    async def _complete(request: Request, path: str):
        auth = await _authorize(request)
        if isinstance(auth, JSONResponse):
            return auth
        key = auth
        client_kind, client_ip = _client_meta(request)

        declared = request.headers.get("content-length", "0")
        if declared.isdigit() and int(declared) > settings.max_body_bytes:
            return _error(413, "Request body too large.", "request_too_large")
        raw = await request.body()
        if len(raw) > settings.max_body_bytes:
            return _error(413, "Request body too large.", "request_too_large")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return _error(400, "Body is not valid JSON.", "invalid_request_error")
        if not isinstance(payload, dict):
            return _error(400, "Body must be a JSON object.", "invalid_request_error")

        model = payload.get("model")
        agent_tool = None
        kind = _KIND_BY_PATH.get(path, "openai")
        tr = trace.new(
            path=path,
            kind=kind,
            client=client_kind,
            user_agent=request.headers.get("user-agent", "")[:200],
            account=key.owner,
            key_hint=key.hint,
            model_requested=model,
            stream=bool(payload.get("stream")),
            client_payload=copy.deepcopy(payload) if trace.enabled else None,
            session=session_key(payload, kind, client_kind, key.owner) if trace.enabled else None,
        )
        info = king.get()
        king_ids = (settings.served_model_name, info.model_id if info else None)
        if model not in (None, settings.agent_model, *king_ids):
            logger.info("[king-agent] model {!r} requested; serving the king", model)
        if path in _KIND_BY_PATH:
            agent_tool = find_shell_tool(payload.get("tools"), kind)
        use_agent = path in _KIND_BY_PATH and (
            model not in (None, *king_ids) or path == "/v1/responses" or agent_tool is not None
        )
        payload["model"] = settings.served_model_name
        if "n" in payload:
            payload["n"] = 1
        for field in _STRIPPED_FIELDS:
            payload.pop(field, None)

        prompt_chars = _prompt_chars(payload)
        if settings.max_prompt_chars and prompt_chars > settings.max_prompt_chars:
            return _error(
                400,
                f"Prompt exceeds {settings.max_prompt_chars} characters.",
                "context_length_exceeded",
            )
        if key.max_prompt_tokens and prompt_chars > key.max_prompt_tokens * _CHARS_PER_TOKEN:
            return _error(
                400,
                f"Prompt exceeds this key's limit of {key.max_prompt_tokens} tokens.",
                "context_length_exceeded",
            )
        if settings.debug and path in _KIND_BY_PATH and is_probe(payload, kind):
            body, events = agent_reply(
                kind, "chatcmpl-probe", model or settings.served_model_name, PROBE_REPLY, None, None
            )
            keys.record(
                key,
                path=path,
                status=200,
                stream=bool(payload.get("stream")),
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                client=client_kind,
                ip=client_ip,
            )
            tr["notes"].append("probe")
            tr["reply"] = body
            trace.write(tr)
            if payload.get("stream"):
                return StreamingResponse(
                    events, media_type="text/event-stream", headers={STATUS_HEADER: "serving"}
                )
            return JSONResponse(body, headers={STATUS_HEADER: "serving"})

        max_tokens = payload.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens > settings.max_tokens_cap:
            payload["max_tokens"] = settings.max_tokens_cap
        if path in ("/v1/chat/completions", "/v1/messages"):
            kwargs = payload.get("chat_template_kwargs")
            kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
            kwargs.setdefault("enable_thinking", settings.enable_thinking)
            payload["chat_template_kwargs"] = kwargs

        stream = bool(payload.get("stream"))
        now = time.time()
        used_today = await run_in_threadpool(
            keys.completion_tokens_since, key.account_id, now - 86400.0
        )
        if used_today >= key.daily_completion_tokens:
            return _error(
                429,
                "Daily token quota exhausted.",
                "quota_exceeded",
                {"retry-after": "3600"},
            )
        rejected = limiter.acquire(key, now)
        if rejected is not None:
            reason, wait = rejected
            wait = wait or settings.retry_after_s
            return _error(
                429,
                f"Rate limit ({reason}). Retry after {wait}s.",
                "rate_limit_exceeded",
                {"retry-after": str(wait)},
            )

        started = time.monotonic()

        def _record(status: int, prompt_tokens: int, completion_tokens: int) -> None:
            limiter.release(key.key_hash)
            if prompt_tokens or completion_tokens:
                meter.record(prompt_tokens, completion_tokens, stream)
            keys.record(
                key,
                path=path,
                status=status,
                stream=stream,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=int((time.monotonic() - started) * 1000),
                client=client_kind,
                ip=client_ip,
            )

        state = await _state()
        if state != EngineState.SERVING.value:
            _record(200, 0, 0)
            tr["notes"].append(f"engine {state}: notice returned")
            trace.write(tr)
            return _notice(stream, state)

        if use_agent:
            return await _agent_step(payload, agent_tool, stream, _record, kind, tr)

        if stream and path in _OPENAI_PATHS:
            payload["stream_options"] = {"include_usage": True}
        body = json.dumps(payload).encode("utf-8")
        tr["king_requests"].append(payload)
        client: httpx.AsyncClient = app.state.client
        req = client.build_request(
            "POST",
            f"{engine.base_url}{path}",
            content=body,
            headers={"content-type": "application/json"},
        )
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("[king-agent] upstream unreachable: {}", exc)
            _record(200, 0, 0)
            tr["notes"].append(f"upstream unreachable: {exc}")
            trace.write(tr)
            return _notice(stream)

        headers = {
            "content-type": upstream.headers.get("content-type", "application/json"),
            STATUS_HEADER: "serving",
        }

        if not stream or upstream.status_code >= 400:
            try:
                raw_body = await upstream.aread()
            finally:
                await upstream.aclose()
            if upstream.status_code == 400:
                raw_body = _clean_upstream_error(raw_body)
            prompt_tokens, completion_tokens = _usage_from_bytes(raw_body)
            _record(upstream.status_code, prompt_tokens, completion_tokens)
            if trace.enabled:
                try:
                    tr["king_responses"].append(json.loads(raw_body))
                except ValueError:
                    tr["king_responses"].append(raw_body.decode("utf-8", "replace"))
                tr["notes"].append(f"passthrough {upstream.status_code}")
                trace.write(tr)
            return Response(raw_body, status_code=upstream.status_code, headers=headers)

        async def _relay() -> AsyncIterator[bytes]:
            tail = b""
            full = b""
            try:
                async for chunk in upstream.aiter_raw():
                    tail = (tail + chunk)[-_TAIL_BYTES:]
                    if trace.enabled:
                        full += chunk
                    yield chunk
            finally:
                await upstream.aclose()
                prompt_tokens, completion_tokens = _usage_from_bytes(tail)
                _record(upstream.status_code, prompt_tokens, completion_tokens)
                if trace.enabled:
                    tr["king_stream_raw"] = full.decode("utf-8", "replace")
                    tr["notes"].append(f"passthrough stream {upstream.status_code}")
                    trace.write(tr)

        return StreamingResponse(_relay(), status_code=upstream.status_code, headers=headers)

    @app.get("/health")
    async def health() -> dict[str, object]:
        info = king.get()
        return {"status": "ok", "state": await _state(), "king": info.public() if info else None}

    @app.get("/status")
    async def status() -> dict[str, object]:
        state = await _state()
        info = king.get()
        return {
            "state": state,
            "model": settings.served_model_name,
            "king": info.public() if info else None,
            "max_model_len": settings.max_model_len,
            "retry_after_s": settings.retry_after_s if state != "serving" else 0,
            "notice": None if state == "serving" else loading_text(info),
        }

    @app.get("/v1/models")
    async def models(request: Request):
        auth = await _authorize(request)
        if isinstance(auth, JSONResponse):
            return auth
        info = king.get()
        data = [_model_entry(settings.served_model_name)]
        if info is not None and info.model_id:
            data.append(_model_entry(info.model_id, f"King {info.roman}"))
        data.append(
            _model_entry(settings.agent_model, "king agent loop via the client's terminal tool")
        )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _complete(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _complete(request, "/v1/completions")

    @app.post("/v1/messages")
    async def messages(request: Request):
        return await _complete(request, "/v1/messages")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        return await _complete(request, "/v1/messages/count_tokens")

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await _complete(request, "/v1/responses")

    return app
