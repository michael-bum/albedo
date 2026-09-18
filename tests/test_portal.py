from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from agent.api import create_ide_app  # noqa: E402
from agent.api.portal import Signer  # noqa: E402
from agent.config import KingAgentSettings  # noqa: E402
from agent.db import KeyStore  # noqa: E402
from agent_key_helpers import FakeEngine  # noqa: E402
from common.king_meta import KingInfo, KingInfoSource  # noqa: E402

KING = KingInfo(roman="CXXVI", repo="dendriteholdings/king", sha="a" * 40, hotkey="5Fabc")
SITE = "https://albedo.tech"
OLD = "2020-01-01T00:00:00Z"


class FakeGitHub:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.redirects: list[str] = []

    def authorize_url(self, redirect_uri: str, state: str) -> str:
        self.redirects.append(redirect_uri)
        return f"https://github.test/authorize?state={state}"

    async def fetch_user(self, code: str, redirect_uri: str) -> dict:
        if code not in self.users:
            raise PermissionError("bad code")
        return self.users[code]


def _make_env(pg_url, **overrides):
    settings = KingAgentSettings(
        **{
            "database_url": pg_url,
            "github_client_id": "cid",
            "github_client_secret": "csecret",
            "portal_session_secret": "s" * 32,
            "portal_public_url": "https://api.test/portal",
            "portal_site_origin": SITE,
            "portal_cookie_secure": False,
            "portal_creations_per_day": 3,
            "portal_history_rows": 2,
            **overrides,
        }
    )
    github = FakeGitHub()
    github.users["good"] = {"id": 42, "login": "octocat", "created_at": OLD}
    github.users["young"] = {"id": 43, "login": "newbie", "created_at": "2026-09-10T00:00:00Z"}
    store = KeyStore(pg_url)
    store.set_tier(
        "standard",
        rpm=30,
        parallel=2,
        daily_completion_tokens=200000,
        max_prompt_tokens=131072,
        key_ttl_days=90,
        max_keys=2,
        actor="test",
    )
    app = create_ide_app(
        FakeEngine(), settings, store, king=KingInfoSource(fixed=KING), github=github
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://api.test",
        headers={"origin": SITE},
    )
    return SimpleNamespace(settings=settings, store=store, github=github, client=client, app=app)


@pytest.fixture
def env(pg_url):
    return _make_env(pg_url)


async def _start(env):
    r = await env.client.get("/portal/login/github", params={"next": f"{SITE}/keys.html"})
    assert r.status_code == 302
    return parse_qs(urlparse(r.headers["location"]).query)["state"][0]


async def _sign_in(env, code="good"):
    state = await _start(env)
    r = await env.client.get("/portal/callback", params={"code": code, "state": state})
    assert r.status_code == 302
    return r


@pytest.mark.anyio
async def test_delete_account_revokes_keys_and_forgets_the_github_id(env):
    await _sign_in(env)
    secret = (await _create(env, "laptop")).json()["secrets"]["albedo"]
    r = await env.client.post("/portal/account/delete")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert (await env.client.get("/portal/me")).status_code == 401
    assert env.store.lookup(secret).active(time.time()) is False
    old = env.store.list_accounts()[0]
    assert old.owner == "deleted:1" and old.contact is None and old.disabled_at is not None
    assert env.store.find_account("github:42") is None
    rows = env.store._db.execute(
        "SELECT actor, detail FROM events WHERE account_id = %s", (old.id,)
    ).fetchall()
    assert rows and all("42" not in (r["actor"] or "") for r in rows)
    assert all(r["detail"] is None or "github" not in r["detail"] for r in rows)
    assert env.store._db.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] >= 3
    assert (await env.client.post("/portal/account/delete")).status_code == 401
    state = await _start(env)
    r = await env.client.get("/portal/callback", params={"code": "good", "state": state})
    assert (
        "signin=failed" in r.headers["location"] and "deleted%20recently" in r.headers["location"]
    )
    assert [a.owner for a in env.store.list_accounts()] == ["deleted:1"]
    env.github.users["other"] = {"id": 77, "login": "someone", "created_at": OLD}
    await _sign_in(env, code="other")
    assert [a.owner for a in env.store.list_accounts()] == ["deleted:1", "github:77"]
    env.store._db.execute(
        "UPDATE accounts SET disabled_at = %s WHERE owner = 'deleted:1'", (time.time() - 25 * 3600,)
    )
    env.store._db.commit()
    await _sign_in(env)
    accounts = env.store.list_accounts()
    assert [a.owner for a in accounts] == ["deleted:1", "github:77", "github:42"]
    assert accounts[0].identity_hash == accounts[2].identity_hash
    me = (await env.client.get("/portal/me")).json()
    assert me["keys"] == [] and me["account"]["tier"] == "standard"


@pytest.mark.anyio
async def test_login_creates_account_and_session(env):
    r = await _sign_in(env)
    assert r.headers["location"] == f"{SITE}/keys.html"
    assert env.github.redirects == ["https://api.test/portal/callback"]
    assert env.settings.portal_cookie_name in r.cookies
    me = await env.client.get("/portal/me")
    assert me.status_code == 200
    body = me.json()
    assert body["account"]["login"] == "octocat" and body["account"]["tier"] == "standard"
    assert body["keys"] == [] and body["limits"]["keys_left"] == 2
    assert body["limits"]["creations_left"] == 3 and body["limits"]["rpm"] == 30
    accounts = env.store.list_accounts()
    assert len(accounts) == 1 and accounts[0].owner == "github:42"
    # a second sign-in reuses the account
    await _sign_in(env)
    assert len(env.store.list_accounts()) == 1


@pytest.mark.anyio
async def test_signed_out_and_bad_state(env):
    assert (await env.client.get("/portal/me")).status_code == 401
    assert (await env.client.post("/portal/key", json={"name": "x"})).status_code == 401
    r = await env.client.get("/portal/callback", params={"code": "good", "state": "junk.mac"})
    assert r.status_code == 302 and "signin=failed" in r.headers["location"]
    signer = Signer(env.settings.portal_session_secret)
    expired = signer.sign({"n": "x", "next": f"{SITE}/keys.html", "exp": time.time() - 1})
    r = await env.client.get("/portal/callback", params={"code": "good", "state": expired})
    assert "signin=failed" in r.headers["location"]
    assert env.store.list_accounts() == []


@pytest.mark.anyio
async def test_callback_needs_the_browser_that_started_the_login(env):
    state = await _start(env)
    stranger = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=env.app), base_url="https://api.test"
    )
    r = await stranger.get("/portal/callback", params={"code": "good", "state": state})
    assert r.status_code == 302 and "did%20not%20start" in r.headers["location"]
    assert env.settings.portal_cookie_name not in r.cookies
    assert env.store.list_accounts() == []
    r = await env.client.get("/portal/callback", params={"code": "good", "state": state})
    assert r.headers["location"] == f"{SITE}/keys.html"
    assert "albedo_oauth" not in env.client.cookies
    assert len(env.store.list_accounts()) == 1


@pytest.mark.anyio
async def test_young_account_and_unknown_next_rejected(env):
    r = await _sign_in(env, code="young")
    assert "signin=failed" in r.headers["location"] and "30%20days" in r.headers["location"]
    assert env.store.list_accounts() == []
    r = await env.client.get("/portal/login/github", params={"next": "https://evil.test/x"})
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = await env.client.get("/portal/callback", params={"code": "good", "state": state})
    assert r.headers["location"] == f"{SITE}/keys.html"


async def _create(env, name):
    return await env.client.post("/portal/key", json={"name": name})


@pytest.mark.anyio
async def test_key_lifecycle(env):
    await _sign_in(env)
    r = await _create(env, "  laptop  ")
    assert r.status_code == 201
    body = r.json()
    secrets = body["secrets"]
    assert set(secrets) == {"albedo", "anthropic", "openai", "openai_project"}
    assert secrets["albedo"].startswith("ak-") and secrets["anthropic"].startswith("sk-ant-api03-")
    now = time.time()
    for spelled in secrets.values():
        key = env.store.lookup(spelled)
        assert key is not None and key.active(now) and key.owner == "github:42"
    secret = secrets["albedo"]
    assert body["key"]["name"] == "laptop" and body["key"]["status"] == "active"
    assert body["key"]["hint_head"] == secret[3:6] and body["key"]["hint"] == secret[-4:]

    key = env.store.lookup(secret)
    env.store.record(
        key,
        path="/v1/chat/completions",
        status=200,
        stream=False,
        prompt_tokens=10,
        completion_tokens=7,
        latency_ms=5,
    )
    me = (await env.client.get("/portal/me")).json()
    assert [k["name"] for k in me["keys"]] == ["laptop"]
    assert me["keys"][0]["requests_7d"] == 1 and me["keys"][0]["tokens_7d"] == 7
    assert me["keys"][0]["last_used_at"] is not None
    assert me["limits"]["tokens_24h"] == 7 and me["limits"]["keys_left"] == 1
    assert me["limits"]["creations_left"] == 2

    assert (await _create(env, "laptop")).status_code == 409
    assert (await _create(env, "LAPTOP")).status_code == 409
    assert (await _create(env, "")).status_code == 400
    assert (await _create(env, "x" * 41)).status_code == 400
    assert (await env.client.post("/portal/key", content=b"nope")).status_code == 400
    r = await _create(env, "desktop")
    assert r.status_code == 201
    desktop_id = r.json()["key"]["id"]
    assert (await _create(env, "third")).status_code == 409
    assert (await env.client.get("/portal/me")).json()["limits"]["keys_left"] == 0

    key_id = body["key"]["id"]
    r = await env.client.post(f"/portal/key/{key_id}/rotate")
    assert r.status_code == 200
    rotated = r.json()["secrets"]["albedo"]
    assert rotated != secret and r.json()["key"]["name"] == "laptop"
    assert env.store.lookup(secret).active(time.time()) is False
    assert env.store.lookup(rotated).active(time.time())
    me = (await env.client.get("/portal/me")).json()
    assert [(k["name"], k["status"]) for k in me["keys"]] == [
        ("desktop", "active"),
        ("laptop", "active"),
    ]
    assert key_id not in {k["id"] for k in me["keys"]}
    assert me["limits"]["creations_left"] == 0
    new_id = r.json()["key"]["id"]
    assert (await env.client.post(f"/portal/key/{new_id}/rotate")).status_code == 429
    assert (await env.client.post(f"/portal/key/{key_id}/rotate")).status_code == 409

    assert (await env.client.post(f"/portal/key/{desktop_id}/revoke")).status_code == 200
    assert (await env.client.post(f"/portal/key/{desktop_id}/revoke")).status_code == 409
    assert (await env.client.post("/portal/key/999999/revoke")).status_code == 404
    me = (await env.client.get("/portal/me")).json()
    assert [(k["name"], k["status"]) for k in me["keys"]] == [
        ("desktop", "revoked"),
        ("laptop", "active"),
    ]
    assert me["limits"]["keys_left"] == 1


@pytest.mark.anyio
async def test_key_cap_comes_from_the_tier(env):
    await _sign_in(env)
    account = env.store.find_account("github:42")
    assert (await _create(env, "a")).status_code == 201
    assert (await _create(env, "b")).status_code == 201
    r = await _create(env, "c")
    assert r.status_code == 409 and "2 active keys" in r.json()["error"]["message"]
    env.store.set_account_tier(account.id, "miner", actor="test")
    me = (await env.client.get("/portal/me")).json()
    assert me["account"]["tier"] == "miner"
    assert me["limits"]["max_keys"] == 10 and me["limits"]["keys_left"] == 8
    assert me["limits"]["creations_per_day"] == 3 and me["limits"]["creations_left"] == 1
    assert (await _create(env, "c")).status_code == 201
    assert (await _create(env, "d")).status_code == 429


@pytest.mark.anyio
async def test_revoked_keys_leave_the_table_after_the_history_window(env):
    await _sign_in(env)
    account = env.store.find_account("github:42")
    old, _ = env.store.issue(account.id, label="old", origin="portal")
    env.store.revoke(old.id, actor="test")
    env.store._db.execute(
        "UPDATE api_keys SET revoked_at = %s WHERE id = %s", (time.time() - 25 * 3600, old.id)
    )
    env.store._db.commit()
    fresh, _ = env.store.issue(account.id, label="fresh", origin="portal")
    env.store.revoke(fresh.id, actor="test")
    me = (await env.client.get("/portal/me")).json()
    assert [k["name"] for k in me["keys"]] == ["fresh"]


@pytest.mark.anyio
async def test_revoked_history_is_capped(env):
    await _sign_in(env)
    account = env.store.find_account("github:42")
    for i in range(4):
        key, _ = env.store.issue(account.id, label=f"old{i}", origin="portal")
        env.store.revoke(key.id, actor="test")
    live, _ = env.store.issue(account.id, label="live", origin="portal")
    me = (await env.client.get("/portal/me")).json()
    names = [k["name"] for k in me["keys"]]
    assert names == ["old2", "old3", "live"]
    assert me["keys"][-1]["id"] == live.id and me["limits"]["keys_left"] == 1


@pytest.mark.anyio
async def test_keys_of_other_accounts_and_cli_keys_are_invisible(env):
    other = env.store.add_account("github:7", tier="standard", contact="someone")
    other_key, _ = env.store.issue(other, label="theirs", origin="portal")
    await _sign_in(env)
    mine = env.store.find_account("github:42")
    cli_key, _ = env.store.issue(mine.id, label="admin-issued")
    assert (await env.client.get("/portal/me")).json()["keys"] == []
    assert (await env.client.post(f"/portal/key/{other_key.id}/revoke")).status_code == 404
    assert (await env.client.post(f"/portal/key/{cli_key.id}/revoke")).status_code == 404
    assert env.store.get_key(other_key.id, other).active(time.time())
    assert env.store.get_key(cli_key.id, mine.id).active(time.time())


@pytest.mark.anyio
async def test_public_status_allows_site_origin_and_api_does_not(env):
    r = await env.client.get("/status")
    assert r.headers["access-control-allow-origin"] == SITE
    r = await env.client.get("/health")
    assert r.headers["access-control-allow-origin"] == SITE
    r = await env.client.get("/v1/models")
    assert "access-control-allow-origin" not in r.headers


def test_authorize_url_requests_no_scope():
    from agent.api.portal import GitHubOAuth

    url = GitHubOAuth("cid", "secret").authorize_url("https://api.test/portal/callback", "st")
    assert "scope=&" in url or url.endswith("scope=") or "&scope=" in url
    assert "allow_signup=false" in url


@pytest.mark.anyio
async def test_cross_site_post_and_logout(env):
    await _sign_in(env)
    r = await env.client.post("/portal/key", headers={"origin": "https://evil.test"})
    assert r.status_code == 403
    r = await env.client.options(
        "/portal/me",
        headers={"origin": "https://evil.test", "access-control-request-method": "GET"},
    )
    assert "access-control-allow-origin" not in r.headers
    r = await env.client.options(
        "/portal/me", headers={"origin": SITE, "access-control-request-method": "GET"}
    )
    assert r.headers["access-control-allow-origin"] == SITE
    assert r.headers["access-control-allow-credentials"] == "true"
    assert (await env.client.post("/portal/logout")).status_code == 200
    assert (await env.client.get("/portal/me")).status_code == 401


def test_portal_not_mounted_without_config(pg_url):
    for settings in (
        KingAgentSettings(database_url=pg_url),
        KingAgentSettings(
            database_url=pg_url,
            github_client_id="cid",
            github_client_secret="csecret",
            portal_session_secret="short",
        ),
    ):
        app = create_ide_app(
            FakeEngine(), settings, KeyStore(pg_url), king=KingInfoSource(fixed=KING)
        )
        assert all(getattr(route, "path", "") != "/portal" for route in app.routes)


@pytest.mark.anyio
async def test_hostile_inputs_are_refused_without_detail(env):
    await _sign_in(env)
    r = await env.client.post("/portal/key", json={"name": "bad\x00name"})
    assert r.status_code == 400
    r = await env.client.post("/portal/key", json={"name": "x", "pad": "y" * 5000})
    assert r.status_code == 400
    r = await env.client.post("/portal/key/abc/revoke")
    assert r.status_code == 400 and r.json() == {
        "error": {
            "message": "Invalid request.",
            "type": "invalid_request_error",
            "code": "invalid_request_error",
        }
    }
    r = await env.client.post("/portal/key/99999999999999999999/revoke")
    assert r.status_code in (400, 404)
    r = await env.client.post("/portal/key/-1/revoke")
    assert r.status_code == 404
    name = "drop'; DROP TABLE api_keys; --"
    r = await env.client.post("/portal/key", json={"name": name})
    assert r.status_code == 201 and r.json()["key"]["name"] == name
    assert env.store.list_keys()[0].label == name


@pytest.mark.anyio
async def test_per_ip_cap_on_portal(pg_url):
    env = _make_env(pg_url, portal_ip_rpm=2)
    codes = [
        (await env.client.get("/portal/login/github", params={"next": SITE})).status_code
        for _ in range(3)
    ]
    assert codes == [302, 302, 429]
    other = await env.client.get(
        "/portal/login/github", params={"next": SITE}, headers={"cf-connecting-ip": "9.9.9.9"}
    )
    assert other.status_code == 302
