from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.db import KeyStore, client_family, hash_secret, split_key  # noqa: E402


@pytest.fixture
def store(pg_url) -> KeyStore:
    return KeyStore(pg_url)


def test_issue_gives_every_format_for_one_row(store):
    acc = store.add_account("bob", tier="beta", contact="bob@x")
    key, shown = store.issue(acc, label="laptop")
    assert set(shown) == {"albedo", "anthropic", "openai", "openai_project"}
    assert shown["albedo"].startswith("ak-") and shown["anthropic"].startswith("sk-ant-api03-")
    assert shown["openai"].startswith("sk-") and shown["openai_project"].startswith("sk-proj-")
    secret = shown["albedo"][3:]
    assert all(v.endswith(secret) for v in shown.values())
    assert key.key_hash == hash_secret(secret) and key.hint == secret[-4:]
    assert len(store.list_keys()) == 1
    for fmt, value in shown.items():
        rec = store.lookup(value)
        assert rec is not None and rec.id == key.id and rec.format == fmt
        assert rec.owner == "bob" and rec.tier == "beta" and rec.active(time.time())
    assert store.lookup("zz-" + secret) is None
    assert store.lookup("sk-") is None
    assert store.lookup("ak-missing") is None


def test_longest_prefix_wins():
    prefixes = ["sk-", "sk-ant-api03-", "sk-proj-", "ak-"]
    assert split_key("sk-ant-api03-abc", prefixes) == ("sk-ant-api03-", "abc")
    assert split_key("sk-proj-abc", prefixes) == ("sk-proj-", "abc")
    assert split_key("sk-abc", prefixes) == ("sk-", "abc")
    assert split_key("sk-", prefixes) is None
    assert split_key("nope", prefixes) is None


def test_tier_defaults_and_key_overrides(store):
    acc = store.add_account("carol", tier="beta")
    key, _ = store.issue(acc)
    assert (key.rpm, key.parallel, key.daily_completion_tokens) == (30, 2, 200_000)
    assert key.max_prompt_tokens == 131_072 and key.expires_at is not None
    key2, _ = store.issue(acc, rpm=5, ttl_days=None)
    assert key2.rpm == 5 and key2.parallel == 2 and key2.expires_at is not None
    store.set_tier(
        "beta",
        rpm=40,
        parallel=3,
        daily_completion_tokens=1,
        max_prompt_tokens=10,
        key_ttl_days=None,
        actor="test",
    )
    assert store.list_keys(acc)[0].rpm == 40 and store.list_keys(acc)[1].rpm == 5
    with pytest.raises(ValueError):
        store.add_account("x", tier="gold")
    internal = store.add_account("ops", tier="internal")
    k3, _ = store.issue(internal)
    assert k3.expires_at is None


def test_revoke_and_account_disable(store):
    acc = store.add_account("dave")
    k1, shown1 = store.issue(acc, label="a")
    k2, shown2 = store.issue(acc, label="b")
    now = time.time()
    assert store.revoke(k1.id, actor="test") == 1
    assert store.revoke(k1.id, actor="test") == 0
    assert not store.lookup(shown1["openai"]).active(now)
    assert store.lookup(shown2["anthropic"]).active(now)
    assert store.set_account_disabled(acc, True, actor="test") == 1
    assert not store.lookup(shown2["anthropic"]).active(now)
    assert not store.lookup(shown2["albedo"]).active(now)
    store.set_account_disabled(acc, False, actor="test")
    assert store.lookup(shown2["albedo"]).active(now)
    assert store.revoke(hint=k2.hint, actor="test") == 1
    actions = [r["action"] for r in store._db.fetchall("SELECT action FROM events")]
    assert actions.count("revoke") == 3 and "account_disable" in actions


def test_expiry(store):
    acc = store.add_account("erin")
    _, shown = store.issue(acc, ttl_days=0.0001)
    rec = store.lookup(shown["albedo"])
    assert rec.active(time.time()) and not rec.active(time.time() + 60)


def test_usage_quota_per_account_completion_only(store):
    acc = store.add_account("frank")
    _, s1 = store.issue(acc)
    _, s2 = store.issue(acc)
    other = store.add_account("gina")
    _, s3 = store.issue(other)
    k1, k2, k3 = (
        store.lookup(s1["anthropic"]),
        store.lookup(s2["openai"]),
        store.lookup(s3["albedo"]),
    )
    store.record(
        k1,
        path="/v1/messages",
        status=200,
        stream=True,
        prompt_tokens=900,
        completion_tokens=12,
        latency_ms=250,
        client="claude-code",
        ip="1.2.3.4",
    )
    store.record(
        k2,
        path="/v1/responses",
        status=429,
        stream=False,
        prompt_tokens=0,
        completion_tokens=30,
        latency_ms=1,
        client="codex",
    )
    store.record(
        k3,
        path="/v1/chat/completions",
        status=200,
        stream=False,
        prompt_tokens=5,
        completion_tokens=5,
        latency_ms=1,
    )
    assert store.completion_tokens_since(acc, 0) == 42
    assert store.completion_tokens_since(other, 0) == 5
    assert store.completion_tokens_since(acc, time.time() + 1) == 0
    by_account = {r["owner"]: r for r in store.report(1)}
    assert by_account["frank"]["requests"] == 2 and by_account["frank"]["errors"] == 1
    assert by_account["frank"]["prompt_tokens"] == 900
    by_format = {r["format"]: r["requests"] for r in store.report(1, by="format")}
    assert by_format == {"anthropic": 1, "openai": 1, "albedo": 1}
    by_client = {r["client"]: r["requests"] for r in store.report(1, by="client")}
    assert by_client["claude-code"] == 1 and by_client[None] == 1
    row = store._db.fetchone("SELECT ip_hash FROM usage WHERE key_id = %s", (k1.id,))
    assert row["ip_hash"] and "1.2.3.4" not in row["ip_hash"]


def test_formats_can_be_added_and_disabled(store):
    acc = store.add_account("hank")
    _, shown = store.issue(acc)
    secret = shown["albedo"][3:]
    store.add_format("cursor", "cur-", "Cursor", actor="test")
    assert store.lookup("cur-" + secret).format == "cursor"
    assert "cursor" in dict(store.issue(acc)[1])
    store.set_format_enabled("openai", False, actor="test")
    assert store.lookup("sk-" + secret) is None
    assert store.lookup("sk-proj-" + secret).format == "openai_project"
    store.set_format_enabled("openai", True, actor="test")
    assert store.lookup("sk-" + secret).format == "openai"


def test_client_family():
    assert client_family("claude-cli/2.0.1 (external, cli)") == "claude-code"
    assert client_family("codex_cli_rs/0.154.0") == "codex"
    assert client_family("GitHubCopilotChat/0.30") == "copilot"
    assert client_family("curl/8.5") == "curl"
    assert client_family("Mozilla/5.0") == "other"
    assert client_family(None) is None
