from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass

_CLIENT_FAMILIES = (
    ("claude", "claude-code"),
    ("codex", "codex"),
    ("copilot", "copilot"),
    ("cursor", "cursor"),
    ("albedo-king-acp", "acp"),
    ("openai", "openai-sdk"),
    ("anthropic", "anthropic-sdk"),
    ("curl", "curl"),
)


@dataclass(frozen=True)
class ApiKey:
    id: int
    account_id: int
    key_hash: str
    hint: str
    label: str | None
    owner: str
    tier: str
    rpm: int
    parallel: int
    daily_completion_tokens: int
    max_prompt_tokens: int
    created_at: float
    expires_at: float | None
    revoked_at: float | None
    account_disabled_at: float | None
    format: str = ""
    hint_head: str | None = None
    origin: str = "cli"

    def active(self, now: float) -> bool:
        if self.revoked_at is not None or self.account_disabled_at is not None:
            return False
        return self.expires_at is None or now < self.expires_at


@dataclass(frozen=True)
class Account:
    id: int
    owner: str
    contact: str | None
    tier: str
    hotkey: str | None
    created_at: float
    disabled_at: float | None
    notes: str | None
    identity_hash: str | None = None


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def split_key(raw: str, prefixes: list[str]) -> tuple[str, str] | None:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if raw.startswith(prefix) and len(raw) > len(prefix):
            return prefix, raw[len(prefix) :]
    return None


def client_family(user_agent: str | None) -> str | None:
    if not user_agent:
        return None
    ua = user_agent.lower()
    for needle, family in _CLIENT_FAMILIES:
        if needle in ua:
            return family
    return "other"


def _ip_hash(ip: str | None, now: float) -> str | None:
    if not ip:
        return None
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    return hashlib.sha256(f"{day}:{ip}".encode()).hexdigest()[:32]


def _to_key(row: Mapping, fmt: str) -> ApiKey:
    return ApiKey(
        id=row["id"],
        account_id=row["account_id"],
        key_hash=row["secret_hash"],
        hint=row["hint"],
        label=row["label"],
        owner=row["owner"],
        tier=row["tier"],
        rpm=row["rpm"],
        parallel=row["parallel"],
        daily_completion_tokens=row["daily_completion_tokens"],
        max_prompt_tokens=row["max_prompt_tokens"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
        account_disabled_at=row["account_disabled_at"],
        format=fmt,
        hint_head=row.get("hint_head"),
        origin=row.get("origin") or "cli",
    )
