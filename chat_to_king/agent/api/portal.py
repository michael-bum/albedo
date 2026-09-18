from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlencode

import httpx
from agent.api.limiter import _error
from agent.config import KingAgentSettings
from agent.db import ApiKey, KeyNameTaken, KeyStore
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from loguru import logger
from starlette.concurrency import run_in_threadpool

GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"
STATE_TTL_S = 600
STATE_COOKIE = "albedo_oauth"
ORIGIN = "portal"
PROVIDER = "github"
NAME_MAX = 40
BODY_MAX = 4096
MAX_ID = 2**62
CREATE_ACTIONS = ("issue", "rotate")


class GitHubOAuth:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret

    def authorize_url(self, redirect_uri: str, state: str) -> str:
        query = urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "scope": "",
                "state": state,
                "allow_signup": "false",
            }
        )
        return f"{GITHUB_AUTHORIZE}?{query}"

    async def fetch_user(self, code: str, redirect_uri: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(
                GITHUB_TOKEN,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"accept": "application/json"},
            )
            token = token_resp.json().get("access_token") if token_resp.is_success else None
            if not token:
                raise PermissionError("github did not return a token")
            user_resp = await client.get(
                GITHUB_USER,
                headers={
                    "authorization": f"Bearer {token}",
                    "accept": "application/vnd.github+json",
                    "user-agent": "albedo-portal",
                },
            )
            if not user_resp.is_success:
                raise PermissionError("github user lookup failed")
            return user_resp.json()


class Signer:
    def __init__(self, secret: str) -> None:
        self._key = secret.encode("utf-8")

    def sign(self, payload: dict[str, Any]) -> str:
        body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(
            b"="
        )
        mac = hmac.new(self._key, body, hashlib.sha256).hexdigest()
        return f"{body.decode()}.{mac}"

    def verify(self, token: str | None, now: float) -> dict[str, Any] | None:
        if not token or "." not in token:
            return None
        body, _, mac = token.rpartition(".")
        expected = hmac.new(self._key, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            return None
        try:
            padded = body + "=" * (-len(body) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict) or payload.get("exp", 0) < now:
            return None
        return payload


def _utc_midnight(now: float) -> float:
    day = datetime.fromtimestamp(now, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return day.timestamp()


def _github_created_at(user: dict[str, Any]) -> float | None:
    raw = user.get("created_at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def create_portal_app(
    settings: KingAgentSettings, keys: KeyStore, github: GitHubOAuth | None = None
) -> FastAPI:
    github = github or GitHubOAuth(settings.github_client_id, settings.github_client_secret)
    signer = Signer(settings.portal_session_secret or secrets.token_hex(32))
    origin = settings.portal_site_origin.rstrip("/")
    public_url = settings.portal_public_url.rstrip("/")
    redirect_uri = f"{public_url}/callback"
    cookie = settings.portal_cookie_name

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError):
        return _error(400, "Invalid request.", "invalid_request_error")

    ip_windows: dict[str, deque[float]] = defaultdict(deque)

    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("cf-connecting-ip") or request.headers.get(
            "x-forwarded-for", ""
        )
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "?"

    def _ip_limited(request: Request, now: float) -> JSONResponse | None:
        if not settings.portal_ip_rpm:
            return None
        if len(ip_windows) > 10_000:
            for stale in [ip for ip, w in ip_windows.items() if not w or w[-1] <= now - 60.0]:
                del ip_windows[stale]
        window = ip_windows[_client_ip(request)]
        while window and window[0] <= now - 60.0:
            window.popleft()
        if len(window) >= settings.portal_ip_rpm:
            return _error(
                429,
                "Too many requests from this address.",
                "rate_limit_exceeded",
                {"retry-after": "60"},
            )
        window.append(now)
        return None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    def _safe_next(value: str | None) -> str:
        if value and (value.startswith(origin + "/") or value == origin):
            return value.split("#")[0]
        return f"{origin}/keys.html"

    def _failed(next_url: str, reason: str) -> RedirectResponse:
        sep = "&" if "?" in next_url else "?"
        return RedirectResponse(
            f"{next_url}{sep}signin=failed&reason={quote(reason)}", status_code=302
        )

    def _set_session(response: Response, account_id: int, now: float) -> None:
        token = signer.sign({"a": account_id, "exp": now + settings.portal_session_ttl_s})
        response.set_cookie(
            cookie,
            token,
            max_age=settings.portal_session_ttl_s,
            httponly=True,
            secure=settings.portal_cookie_secure,
            samesite="lax",
            path="/",
        )

    async def _account(request: Request):
        payload = signer.verify(request.cookies.get(cookie), time.time())
        if payload is None or not isinstance(payload.get("a"), int):
            return None
        account = await run_in_threadpool(keys.get_account, payload["a"])
        if account is None or account.disabled_at is not None:
            return None
        return account

    def _same_site(request: Request) -> bool:
        sent = request.headers.get("origin")
        return sent is None or sent.rstrip("/") == origin

    def _status(key: ApiKey, now: float) -> str:
        if key.revoked_at is not None:
            return "revoked"
        if key.expires_at is not None and key.expires_at <= now:
            return "expired"
        return "active"

    def _visible_keys(account_id: int, now: float) -> list[ApiKey]:
        cutoff = now - settings.portal_history_hours * 3600
        all_keys = keys.list_keys(account_id, origin=ORIGIN)
        live = [k for k in all_keys if k.revoked_at is None]
        live_names = {(k.label or "").lower() for k in live}
        gone = [
            k
            for k in all_keys
            if k.revoked_at is not None
            and k.revoked_at >= cutoff
            and (k.label or "").lower() not in live_names
        ]
        gone = gone[-settings.portal_history_rows :]
        return sorted(live + gone, key=lambda k: k.created_at)

    def _key_json(key: ApiKey, now: float) -> dict[str, Any]:
        week = keys.key_usage(key.id, now - 7 * 86400)
        return {
            "id": key.id,
            "name": key.label,
            "hint_head": key.hint_head,
            "hint": key.hint,
            "status": _status(key, now),
            "created_at": key.created_at,
            "expires_at": key.expires_at,
            "revoked_at": key.revoked_at,
            "last_used_at": week["last_used_at"],
            "requests_7d": int(week["requests"]),
            "tokens_7d": int(week["completion_tokens"]),
        }

    def _max_keys(account) -> int:
        tier = keys.tier(account.tier) or {}
        return int(tier.get("max_keys") or 0)

    def _limits_json(account, active: int, now: float) -> dict[str, Any]:
        tier = keys.tier(account.tier) or {}
        created = keys.count_events(account.id, CREATE_ACTIONS, _utc_midnight(now))
        max_keys = int(tier.get("max_keys") or 0)
        return {
            "max_keys": max_keys,
            "keys_left": max(0, max_keys - active),
            "creations_per_day": settings.portal_creations_per_day,
            "creations_left": max(0, settings.portal_creations_per_day - created),
            "rpm": tier.get("rpm"),
            "parallel": tier.get("parallel"),
            "daily_completion_tokens": tier.get("daily_completion_tokens"),
            "max_prompt_tokens": tier.get("max_prompt_tokens"),
            "key_ttl_days": tier.get("key_ttl_days"),
            "tokens_24h": keys.completion_tokens_since(account.id, now - 86400.0),
            "support_url": settings.support_url or None,
        }

    def _me_json(account, now: float) -> dict[str, Any]:
        visible = _visible_keys(account.id, now)
        active = sum(1 for k in visible if k.active(now))
        return {
            "account": {
                "login": account.contact,
                "provider": PROVIDER,
                "tier": account.tier,
                "created_at": account.created_at,
            },
            "keys": [_key_json(k, now) for k in visible],
            "limits": _limits_json(account, active, now),
        }

    def _issue(account, name: str, now: float, action: str) -> JSONResponse:
        if _limits_json(account, 0, now)["creations_left"] <= 0:
            return _error(429, "Key creation limit reached for today.", "creation_limit")
        try:
            key, shown = keys.issue(
                account.id,
                label=name,
                actor=f"portal:{account.owner}",
                action=action,
                origin=ORIGIN,
            )
        except KeyNameTaken:
            return _error(409, "You already have an active key with that name.", "name_taken")
        body = {"secrets": shown, "key": _key_json(key, now)}
        return JSONResponse(body, status_code=201 if action == "issue" else 200)

    def _clean_name(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        name = " ".join(value.split())
        if not name or len(name) > NAME_MAX or not name.isprintable():
            return None
        return name

    async def _small_body(request: Request) -> Any:
        declared = request.headers.get("content-length", "0")
        if not declared.isdigit() or int(declared) > BODY_MAX:
            return None
        raw = await request.body()
        if len(raw) > BODY_MAX:
            return None
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return None

    @app.get("/login/github")
    async def login(request: Request):
        limited = _ip_limited(request, time.time())
        if limited is not None:
            return limited
        next_url = _safe_next(request.query_params.get("next"))
        nonce = secrets.token_urlsafe(16)
        state = signer.sign({"n": nonce, "next": next_url, "exp": time.time() + STATE_TTL_S})
        response = RedirectResponse(github.authorize_url(redirect_uri, state), status_code=302)
        response.set_cookie(
            STATE_COOKIE,
            nonce,
            max_age=STATE_TTL_S,
            httponly=True,
            secure=settings.portal_cookie_secure,
            samesite="lax",
            path="/portal/callback",
        )
        return response

    @app.get("/callback")
    async def callback(request: Request):
        now = time.time()
        limited = _ip_limited(request, now)
        if limited is not None:
            return limited
        state = signer.verify(request.query_params.get("state"), now)
        next_url = _safe_next(state.get("next") if state else None)
        code = request.query_params.get("code")
        nonce = request.cookies.get(STATE_COOKIE)
        if state is None or not code:
            return _failed(next_url, "sign-in expired, try again")
        if not nonce or not hmac.compare_digest(nonce, str(state.get("n", ""))):
            return _failed(next_url, "sign-in did not start in this browser, try again")
        try:
            user = await github.fetch_user(code, redirect_uri)
        except (PermissionError, httpx.HTTPError) as exc:
            logger.warning("[portal] github sign-in failed: {}", exc)
            return _failed(next_url, "github sign-in failed")
        uid, login_name = user.get("id"), user.get("login")
        if not isinstance(uid, int) or not isinstance(login_name, str):
            return _failed(next_url, "github sign-in failed")
        created = _github_created_at(user)
        if created is None or now - created < settings.portal_account_min_age_days * 86400:
            return _failed(
                next_url,
                f"github account must be at least {settings.portal_account_min_age_days} days old",
            )
        owner = f"{PROVIDER}:{uid}"
        identity = hmac.new(
            settings.portal_session_secret.encode("utf-8"), owner.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        account = await run_in_threadpool(keys.find_account, owner)
        if account is None:
            deleted_at = await run_in_threadpool(keys.last_deleted_at, identity)
            cooldown = settings.portal_recreate_cooldown_hours * 3600
            if deleted_at is not None and now - deleted_at < cooldown:
                hours = max(1, int((deleted_at + cooldown - now) // 3600) + 1)
                return _failed(
                    next_url, f"account deleted recently, you can sign in again in {hours} h"
                )
            account_id = await run_in_threadpool(
                lambda: keys.add_account(
                    owner,
                    tier=settings.portal_tier,
                    contact=login_name,
                    actor="portal",
                    identity_hash=identity,
                )
            )
        elif account.disabled_at is not None:
            return _failed(next_url, "account disabled")
        else:
            account_id = account.id
        response = RedirectResponse(next_url, status_code=302)
        response.delete_cookie(STATE_COOKIE, path="/portal/callback")
        _set_session(response, account_id, now)
        return response

    @app.get("/me")
    async def me(request: Request):
        account = await _account(request)
        if account is None:
            return _error(401, "Not signed in.", "not_signed_in")
        return JSONResponse(await run_in_threadpool(_me_json, account, time.time()))

    @app.post("/logout")
    async def logout(request: Request):
        response = JSONResponse({"ok": True})
        response.delete_cookie(cookie, path="/")
        return response

    @app.post("/account/delete")
    async def delete_account(request: Request):
        account = await _guard(request)
        if isinstance(account, JSONResponse):
            return account
        await run_in_threadpool(keys.delete_account, account.id, actor=f"portal:{account.owner}")
        response = JSONResponse({"ok": True})
        response.delete_cookie(cookie, path="/")
        return response

    async def _guard(request: Request):
        limited = _ip_limited(request, time.time())
        if limited is not None:
            return limited
        if not _same_site(request):
            return _error(403, "Cross-site request refused.", "forbidden")
        account = await _account(request)
        if account is None:
            return _error(401, "Not signed in.", "not_signed_in")
        return account

    @app.post("/key")
    async def create_key(request: Request):
        account = await _guard(request)
        if isinstance(account, JSONResponse):
            return account
        payload = await _small_body(request)
        name = _clean_name(payload.get("name") if isinstance(payload, dict) else None)
        if name is None:
            return _error(400, f"Give the key a name (1-{NAME_MAX} characters).", "invalid_name")
        now = time.time()

        def _create():
            active = sum(1 for k in _visible_keys(account.id, now) if k.active(now))
            max_keys = _max_keys(account)
            if active >= max_keys:
                return _error(
                    409,
                    f"Your tier allows {max_keys} active keys. Revoke one first.",
                    "key_limit",
                )
            return _issue(account, name, now, "issue")

        return await run_in_threadpool(_create)

    def _owned_key(account, key_id: int, now: float) -> ApiKey | JSONResponse:
        if not 0 < key_id < MAX_ID:
            return _error(404, "No such key.", "not_found")
        key = keys.get_key(key_id, account.id)
        if key is None or key.origin != ORIGIN:
            return _error(404, "No such key.", "not_found")
        if not key.active(now):
            return _error(409, "That key is no longer active.", "key_inactive")
        return key

    @app.post("/key/{key_id}/rotate")
    async def rotate_key(key_id: int, request: Request):
        account = await _guard(request)
        if isinstance(account, JSONResponse):
            return account
        now = time.time()

        def _rotate():
            current = _owned_key(account, key_id, now)
            if isinstance(current, JSONResponse):
                return current
            if _limits_json(account, 0, now)["creations_left"] <= 0:
                return _error(429, "Key creation limit reached for today.", "creation_limit")
            keys.revoke(current.id, actor=f"portal:{account.owner}")
            return _issue(account, current.label or "key", now, "rotate")

        return await run_in_threadpool(_rotate)

    @app.post("/key/{key_id}/revoke")
    async def revoke_key(key_id: int, request: Request):
        account = await _guard(request)
        if isinstance(account, JSONResponse):
            return account
        now = time.time()

        def _revoke():
            current = _owned_key(account, key_id, now)
            if isinstance(current, JSONResponse):
                return current
            keys.revoke(current.id, actor=f"portal:{account.owner}")
            return JSONResponse({"ok": True})

        return await run_in_threadpool(_revoke)

    return app
