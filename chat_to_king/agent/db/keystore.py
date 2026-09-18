from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Mapping

from agent.db.connection import _Db
from agent.db.models import Account, ApiKey, _ip_hash, _to_key, hash_secret, split_key
from psycopg.errors import UniqueViolation

SECRET_BYTES = 32
SECRET_RETRIES = 3
FORMAT_CACHE_S = 60.0

_KEY_SELECT = """
SELECT k.id, k.account_id, k.secret_hash, k.hint, k.hint_head, k.origin, k.label, k.created_at,
       k.expires_at, k.revoked_at,
       a.owner, a.tier, a.disabled_at AS account_disabled_at,
       COALESCE(k.rpm, t.rpm) AS rpm,
       COALESCE(k.parallel, t.parallel) AS parallel,
       COALESCE(k.daily_completion_tokens, t.daily_completion_tokens) AS daily_completion_tokens,
       COALESCE(k.max_prompt_tokens, t.max_prompt_tokens) AS max_prompt_tokens
FROM api_keys k
JOIN accounts a ON a.id = k.account_id
JOIN tiers t ON t.name = a.tier
"""


class KeyNameTaken(ValueError):
    pass


class KeyStore:
    def __init__(self, target: str) -> None:
        self._db = _Db(target)
        self._lock = threading.Lock()
        self._formats: list[tuple[str, str]] = []
        self._formats_at = 0.0
        with self._lock:
            self._db.apply_schema()

    def _event(
        self,
        actor: str,
        action: str,
        account_id: int | None = None,
        key_id: int | None = None,
        detail: str | None = None,
    ) -> None:
        self._db.execute(
            "INSERT INTO events (ts, actor, action, account_id, key_id, detail) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (time.time(), actor, action, account_id, key_id, detail),
        )

    def formats(self, *, enabled_only: bool = True) -> list[tuple[str, str]]:
        now = time.monotonic()
        if enabled_only and self._formats and now - self._formats_at < FORMAT_CACHE_S:
            return self._formats
        where = "WHERE enabled = 1" if enabled_only else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT format, prefix, enabled, note FROM key_formats {where} ORDER BY format"
            ).fetchall()
        result = [(r["format"], r["prefix"]) for r in rows]
        if enabled_only:
            self._formats, self._formats_at = result, now
        return result

    def list_formats(self) -> list[Mapping]:
        with self._lock:
            return self._db.execute("SELECT * FROM key_formats ORDER BY format").fetchall()

    def add_format(self, fmt: str, prefix: str, note: str | None = None, *, actor: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO key_formats (format, prefix, note) VALUES (%s, %s, %s)",
                (fmt, prefix, note),
            )
            self._event(actor, "format_add", detail=f"{fmt}={prefix}")
            self._db.commit()
        self._formats_at = 0.0

    def set_format_enabled(self, fmt: str, enabled: bool, *, actor: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "UPDATE key_formats SET enabled = %s WHERE format = %s", (int(enabled), fmt)
            )
            self._event(actor, "format_enable" if enabled else "format_disable", detail=fmt)
            self._db.commit()
        self._formats_at = 0.0
        return cur.rowcount

    def parse(self, raw: str) -> tuple[str, str] | None:
        formats = self.formats()
        split = split_key(raw, [p for _, p in formats])
        if split is None:
            return None
        prefix, secret = split
        fmt = next(f for f, p in formats if p == prefix)
        return fmt, secret

    def lookup(self, raw: str) -> ApiKey | None:
        parsed = self.parse(raw.strip())
        if parsed is None:
            return None
        fmt, secret = parsed
        with self._lock:
            row = self._db.execute(
                _KEY_SELECT + "WHERE k.secret_hash = %s", (hash_secret(secret),)
            ).fetchone()
        return _to_key(row, fmt) if row else None

    def list_tiers(self) -> list[Mapping]:
        with self._lock:
            return self._db.execute("SELECT * FROM tiers ORDER BY name").fetchall()

    def set_tier(
        self,
        name: str,
        *,
        rpm: int,
        parallel: int,
        daily_completion_tokens: int,
        max_prompt_tokens: int,
        key_ttl_days: int | None,
        max_keys: int = 5,
        actor: str,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO tiers (name, rpm, parallel, daily_completion_tokens, "
                "max_prompt_tokens, key_ttl_days, max_keys) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT(name) DO UPDATE SET "
                "rpm = excluded.rpm, parallel = excluded.parallel, "
                "daily_completion_tokens = excluded.daily_completion_tokens, "
                "max_prompt_tokens = excluded.max_prompt_tokens, "
                "key_ttl_days = excluded.key_ttl_days, max_keys = excluded.max_keys",
                (
                    name,
                    rpm,
                    parallel,
                    daily_completion_tokens,
                    max_prompt_tokens,
                    key_ttl_days,
                    max_keys,
                ),
            )
            self._event(actor, "tier_set", detail=name)
            self._db.commit()

    def add_account(
        self,
        owner: str,
        *,
        tier: str = "standard",
        contact: str | None = None,
        hotkey: str | None = None,
        notes: str | None = None,
        actor: str = "cli",
        identity_hash: str | None = None,
    ) -> int:
        with self._lock:
            if self._db.execute("SELECT 1 FROM tiers WHERE name = %s", (tier,)).fetchone() is None:
                raise ValueError(f"unknown tier {tier!r}")
            account_id = self._db.insert_id(
                "INSERT INTO accounts (owner, contact, tier, hotkey, created_at, notes, "
                "identity_hash) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (owner, contact, tier, hotkey, time.time(), notes, identity_hash),
            )
            self._event(actor, "account_add", account_id, detail=owner)
            self._db.commit()
        return account_id

    def last_deleted_at(self, identity_hash: str) -> float | None:
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(disabled_at) AS t FROM accounts "
                "WHERE identity_hash = %s AND owner LIKE 'deleted:%%'",
                (identity_hash,),
            ).fetchone()
        return row["t"] if row and row["t"] is not None else None

    def set_account_disabled(self, account_id: int, disabled: bool, *, actor: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "UPDATE accounts SET disabled_at = %s WHERE id = %s",
                (time.time() if disabled else None, account_id),
            )
            self._event(actor, "account_disable" if disabled else "account_enable", account_id)
            self._db.commit()
        return cur.rowcount

    def delete_account(self, account_id: int, *, actor: str) -> int:
        now = time.time()
        with self._lock:
            self._db.execute(
                "UPDATE api_keys SET revoked_at = %s WHERE account_id = %s AND revoked_at IS NULL",
                (now, account_id),
            )
            cur = self._db.execute(
                "UPDATE accounts SET owner = 'deleted:' || id, contact = NULL, hotkey = NULL, "
                "notes = NULL, disabled_at = %s WHERE id = %s AND disabled_at IS NULL",
                (now, account_id),
            )
            self._event(actor, "account_delete", account_id)
            self._db.execute(
                "UPDATE events SET actor = 'portal:deleted', detail = NULL "
                "WHERE account_id = %s AND (actor LIKE 'portal:%%' OR action = 'account_add')",
                (account_id,),
            )
            self._db.commit()
        return cur.rowcount

    def set_account_tier(self, account_id: int, tier: str, *, actor: str) -> int:
        with self._lock:
            if self._db.execute("SELECT 1 FROM tiers WHERE name = %s", (tier,)).fetchone() is None:
                raise ValueError(f"unknown tier {tier!r}")
            cur = self._db.execute(
                "UPDATE accounts SET tier = %s WHERE id = %s", (tier, account_id)
            )
            self._event(actor, "tier_change", account_id, detail=tier)
            self._db.commit()
        return cur.rowcount

    def find_account(self, owner: str) -> Account | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM accounts WHERE owner = %s", (owner,)).fetchone()
        return Account(**dict(row)) if row else None

    def get_account(self, account_id: int) -> Account | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM accounts WHERE id = %s", (account_id,)).fetchone()
        return Account(**dict(row)) if row else None

    def count_events(self, account_id: int, actions: tuple[str, ...], since: float) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM events "
                "WHERE account_id = %s AND action = ANY(%s) AND ts >= %s",
                (account_id, list(actions), since),
            ).fetchone()
        return int(row["n"])

    def tier(self, name: str) -> Mapping | None:
        with self._lock:
            return self._db.execute("SELECT * FROM tiers WHERE name = %s", (name,)).fetchone()

    def key_usage(self, key_id: int, since: float) -> Mapping:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) AS requests, "
                "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
                "MAX(ts) AS last_used_at FROM usage WHERE key_id = %s AND ts >= %s",
                (key_id, since),
            ).fetchone()

    def list_accounts(self) -> list[Account]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM accounts ORDER BY id").fetchall()
        return [Account(**dict(r)) for r in rows]

    def issue(
        self,
        account_id: int,
        *,
        label: str | None = None,
        ttl_days: float | None = None,
        rpm: int | None = None,
        parallel: int | None = None,
        daily_completion_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
        actor: str = "cli",
        action: str = "issue",
        origin: str = "cli",
    ) -> tuple[ApiKey, dict[str, str]]:
        now = time.time()
        with self._lock:
            account = self._db.execute(
                "SELECT a.id, t.key_ttl_days FROM accounts a JOIN tiers t ON t.name = a.tier "
                "WHERE a.id = %s",
                (account_id,),
            ).fetchone()
            if account is None:
                raise ValueError(f"unknown account {account_id}")
            if ttl_days is None:
                ttl_days = account["key_ttl_days"]
            expires_at = now + ttl_days * 86400 if ttl_days else None
            for _attempt in range(SECRET_RETRIES):
                secret = secrets.token_urlsafe(SECRET_BYTES)
                try:
                    key_id = self._db.insert_id(
                        "INSERT INTO api_keys (account_id, secret_hash, hint, hint_head, "
                        "origin, label, created_at, expires_at, rpm, parallel, "
                        "daily_completion_tokens, max_prompt_tokens) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            account_id,
                            hash_secret(secret),
                            secret[-4:],
                            secret[:3],
                            origin,
                            label,
                            now,
                            expires_at,
                            rpm,
                            parallel,
                            daily_completion_tokens,
                            max_prompt_tokens,
                        ),
                    )
                    break
                except UniqueViolation as exc:
                    self._db.rollback()
                    if "secret_hash" in (exc.diag.constraint_name or ""):
                        continue
                    raise KeyNameTaken(label or "") from exc
            else:
                raise RuntimeError("could not generate a unique key secret")
            self._event(actor, action, account_id, key_id, detail=label)
            self._db.commit()
            row = self._db.execute(_KEY_SELECT + "WHERE k.id = %s", (key_id,)).fetchone()
        presentations = {fmt: prefix + secret for fmt, prefix in self.formats()}
        return _to_key(row, ""), presentations

    def revoke(self, key_id: int | None = None, *, hint: str | None = None, actor: str) -> int:
        if key_id is None and not hint:
            raise ValueError("key id or hint required")
        cond, arg = ("id = %s", key_id) if key_id is not None else ("hint = %s", hint)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE api_keys SET revoked_at = %s WHERE {cond} AND revoked_at IS NULL",
                (time.time(), arg),
            )
            self._event(actor, "revoke", key_id=key_id, detail=hint)
            self._db.commit()
        return cur.rowcount

    def get_key(self, key_id: int, account_id: int) -> ApiKey | None:
        with self._lock:
            row = self._db.execute(
                _KEY_SELECT + "WHERE k.id = %s AND k.account_id = %s", (key_id, account_id)
            ).fetchone()
        return _to_key(row, "") if row else None

    def list_keys(self, account_id: int | None = None, origin: str | None = None) -> list[ApiKey]:
        clauses = []
        args: tuple = ()
        if account_id is not None:
            clauses.append("k.account_id = %s")
            args += (account_id,)
        if origin is not None:
            clauses.append("k.origin = %s")
            args += (origin,)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._db.execute(_KEY_SELECT + f"{where} ORDER BY k.created_at", args).fetchall()
        return [_to_key(r, "") for r in rows]

    def record(
        self,
        key: ApiKey,
        *,
        path: str,
        status: int,
        stream: bool,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        client: str | None = None,
        ip: str | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO usage (ts, key_id, format, path, client, status, stream, "
                "prompt_tokens, completion_tokens, latency_ms, ip_hash) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    now,
                    key.id,
                    key.format or None,
                    path,
                    client,
                    status,
                    int(stream),
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    _ip_hash(ip, now),
                ),
            )
            self._db.commit()

    def completion_tokens_since(self, account_id: int, since: float) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(u.completion_tokens), 0) AS n FROM usage u "
                "JOIN api_keys k ON k.id = u.key_id WHERE k.account_id = %s AND u.ts >= %s",
                (account_id, since),
            ).fetchone()
        return int(row["n"])

    def report(self, days: float, by: str = "account") -> list[Mapping]:
        group = {
            "account": "a.id",
            "key": "k.id",
            "format": "u.format",
            "client": "u.client",
            "path": "u.path",
        }[by]
        with self._lock:
            return self._db.execute(
                f"SELECT MIN(a.id) AS account_id, MIN(a.owner) AS owner, MIN(k.id) AS key_id, "
                f"MIN(k.hint) AS hint, MIN(u.format) AS format, MIN(u.client) AS client, "
                f"MIN(u.path) AS path, COUNT(u.id) AS requests, "
                f"COALESCE(SUM(u.prompt_tokens), 0) AS prompt_tokens, "
                f"COALESCE(SUM(u.completion_tokens), 0) AS completion_tokens, "
                f"SUM(CASE WHEN u.status >= 400 THEN 1 ELSE 0 END) AS errors, "
                f"COALESCE(AVG(u.latency_ms), 0) AS avg_latency_ms "
                f"FROM usage u JOIN api_keys k ON k.id = u.key_id "
                f"JOIN accounts a ON a.id = k.account_id "
                f"WHERE u.ts >= %s GROUP BY {group} ORDER BY completion_tokens DESC",
                (time.time() - days * 86400,),
            ).fetchall()
