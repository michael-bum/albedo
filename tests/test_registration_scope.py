from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from pathlib import Path
from types import SimpleNamespace

import asyncpg
import pytest
from nacl.signing import SigningKey

from chain_guard import db as guard_db
from chain_reader.chain import PrivateSignal
from model_validation import db as mv_db
from private_store import intake
from private_store.contracts import activation_signal_payload, registration_id
from private_store.crypto import encode_ss58_public_key

_ADMIN_URL = os.environ.get("ALBEDO_TEST_DATABASE_URL")
_SCHEMA = (Path(__file__).resolve().parents[1] / "schema.sql").read_text()

HOTKEY = encode_ss58_public_key(bytes(SigningKey(b"r" * 32).verify_key))
SETTINGS = SimpleNamespace(max_attempts=3, chain_generation="albedo-mainnet-1")
RID = registration_id(netuid=97, hotkey=HOTKEY, chain_generation=SETTINGS.chain_generation)
OLD_REG, NEW_REG = 1_000, 2_000


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def pool_factory(monkeypatch):
    if not _ADMIN_URL:
        pytest.skip("ALBEDO_TEST_DATABASE_URL is not set")
    monkeypatch.setattr(intake, "_settings", lambda: SETTINGS)
    name = f"t_{secrets.token_hex(6)}"
    base, _, _ = _ADMIN_URL.rpartition("/")

    async def setup():
        admin = await asyncpg.connect(_ADMIN_URL)
        await admin.execute(f"CREATE DATABASE {name}")
        await admin.close()
        conn = await asyncpg.connect(f"{base}/{name}")
        await conn.execute(_SCHEMA)
        await conn.close()

    _run(setup())

    def factory():
        return asyncpg.create_pool(f"{base}/{name}", min_size=1, max_size=2)

    factory.url = f"{base}/{name}"
    yield factory

    async def teardown():
        admin = await asyncpg.connect(_ADMIN_URL)
        await admin.execute(f"DROP DATABASE {name} WITH (FORCE)")
        await admin.close()

    _run(teardown())


async def _miner(conn, *, uid: int, registration_block: int) -> None:
    await conn.execute(
        "INSERT INTO miners (hotkey, uid, netuid, registration_block) VALUES ($1, $2, 97, $3)",
        HOTKEY,
        uid,
        registration_block,
    )


async def _submission(conn, *, block: int, state: str, fault_code: str | None = None):
    commit_id = await conn.fetchval(
        """
        INSERT INTO chain_commits (netuid, block_number, block_hash, uid, hotkey, model_uri,
                                   payload_hash)
        VALUES (97, $1, '0x', 44, $2, 'hf://m', $3) RETURNING id
        """,
        block,
        HOTKEY,
        secrets.token_hex(8),
    )
    return await conn.fetchval(
        """
        INSERT INTO model_submissions (chain_commit_id, netuid, uid, hotkey, model_uri, state,
                                       fault_class, fault_code, fault_message, idempotency_key)
        VALUES ($1, 97, 44, $2, $3, $4, $5, $6, $7, $8) RETURNING id
        """,
        commit_id,
        HOTKEY,
        f"hf://m-{block}",
        state,
        "MINER_FAULT" if fault_code else None,
        fault_code,
        f"prior {fault_code}" if fault_code else None,
        str(uuid.uuid4()),
    )


def test_refresh_moves_the_registration_block_even_while_the_uid_is_stale(pool_factory):
    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, uid=142, registration_block=OLD_REG)
        await guard_db.refresh_registration_blocks(pool, [(44, HOTKEY, NEW_REG)])
        block = await pool.fetchval(
            "SELECT registration_block FROM miners WHERE hotkey = $1", HOTKEY
        )
        await pool.close()
        return block

    assert _run(go()) == NEW_REG


def test_strikes_blocks_and_validated_models_count_only_the_current_registration(pool_factory):
    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, uid=44, registration_block=NEW_REG)
            await _submission(conn, block=1_100, state="TERMINAL_INVALID", fault_code="failed")
            await _submission(conn, block=1_200, state="TERMINAL_INVALID", fault_code="failed")
            await _submission(conn, block=1_300, state="TERMINAL_INVALID", fault_code="duplicate")
            await _submission(conn, block=1_400, state="COMPLETE_LOSS")
            await conn.execute(
                """
                INSERT INTO sanity_results (repo, digest, passed, reason, checked_at)
                VALUES ('hf://m-1100', 'd1', false, 'prompt injection detected', now())
                """
            )
        before = (
            await mv_db.hotkey_preeval_fail_count(pool, HOTKEY),
            await mv_db.hotkey_duplicate_block_reason(pool, HOTKEY),
            await mv_db.hotkey_sanity_block_reason(pool, HOTKEY),
            await mv_db.hotkey_validated(pool, HOTKEY),
        )
        async with pool.acquire() as conn:
            await _submission(conn, block=2_100, state="TERMINAL_INVALID", fault_code="failed")
            await _submission(conn, block=2_200, state="PRE_EVAL_PASSED")
        after = (
            await mv_db.hotkey_preeval_fail_count(pool, HOTKEY),
            await mv_db.hotkey_validated(pool, HOTKEY),
        )
        await pool.close()
        return before, after

    before, after = _run(go())
    assert before == (0, None, None, False)
    assert after == (1, True)


async def _registration(conn, *, state: str, attempt_count: int, activation_block: int, sub=None):
    await conn.execute(
        """
        INSERT INTO private_registrations (netuid, uid, hotkey, registration_id, state,
                                           activation_block, submission_pubkey, attempt_count,
                                           submission_id, model_digest)
        VALUES (97, 142, $1, $2, $3, $4, 'old', $5, $6, 'olddigest')
        """,
        HOTKEY,
        RID,
        state,
        activation_block,
        attempt_count,
        sub,
    )


def _activate(block: int) -> PrivateSignal:
    payload = activation_signal_payload(bytes(SigningKey(b"k" * 32).verify_key))
    return PrivateSignal("activate", 97, block, 44, HOTKEY, payload, None, None)


def test_an_activation_from_a_new_registration_gets_a_fresh_upload_budget(pool_factory):
    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, uid=44, registration_block=NEW_REG)
            sub = await _submission(conn, block=1_500, state="COMPLETE_LOSS")
            # spent every attempt, and the last one reached eval: closed under the old rules
            await _registration(
                conn, state="SUBMITTED", attempt_count=3, activation_block=1_400, sub=sub
            )
        stale = await intake.handle_signals(pool, [_activate(1_900)])
        applied = await intake.handle_signals(pool, [_activate(2_050)])
        replayed = await intake.handle_signals(pool, [_activate(2_050)])
        row = await pool.fetchrow(
            "SELECT * FROM private_registrations WHERE registration_id = $1", RID
        )
        await pool.close()
        return stale, applied, replayed, row

    stale, applied, replayed, row = _run(go())
    assert (stale, applied, replayed) == (0, 1, 0)
    assert row["state"] == "ACTIVATED" and row["uid"] == 44
    assert row["attempt_count"] == 4  # a new prefix: attempts never reuse one
    assert row["attempt_count"] - row["extra_attempts"] == 1  # one used of this registration
    assert row["submission_id"] is None and row["model_digest"] is None


def test_the_same_registration_keeps_its_upload_budget(pool_factory):
    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, uid=44, registration_block=OLD_REG)
            await _registration(conn, state="FAILED", attempt_count=3, activation_block=1_400)
        applied = await intake.handle_signals(pool, [_activate(1_900)])
        await pool.close()
        return applied

    assert _run(go()) == 0


def _commit(block: int, uid: int, block_hash: str = "0xhash"):
    from chain_reader.chain import Commit

    return Commit(
        netuid=97,
        block_number=block,
        block_hash=block_hash,
        extrinsic_hash=None,
        uid=uid,
        hotkey=HOTKEY,
        commit_payload={"digest": "sha256:" + "a" * 64},
        model_uri="hf://m@rev",
        payload_hash="same-payload",
    )


def test_recommitting_a_submitted_payload_leaves_the_old_submission_in_its_registration(
    pool_factory,
):
    from chain_reader import db as chain_db

    async def go():
        pool = await pool_factory()
        await chain_db.insert_new_commits(pool, [_commit(1_500, uid=142)])
        await pool.execute(
            "UPDATE model_submissions SET state = 'COMPLETE_LOSS' WHERE hotkey = $1", HOTKEY
        )
        await pool.execute(
            "UPDATE miners SET uid = 44, registration_block = $2 WHERE hotkey = $1",
            HOTKEY,
            NEW_REG,
        )
        await chain_db.insert_new_commits(pool, [_commit(2_500, uid=44, block_hash="0xnew")])
        commit = await pool.fetchrow("SELECT * FROM chain_commits WHERE hotkey = $1", HOTKEY)
        submissions = await pool.fetchval(
            "SELECT count(*) FROM model_submissions WHERE hotkey = $1", HOTKEY
        )
        validated = await mv_db.hotkey_validated(pool, HOTKEY)
        await pool.close()
        return commit, submissions, validated

    commit, submissions, validated = _run(go())
    assert (commit["block_number"], commit["uid"], commit["block_hash"]) == (1_500, 142, "0xhash")
    assert submissions == 1
    assert validated is False


SWAPPED_IN = encode_ss58_public_key(bytes(SigningKey(b"w" * 32).verify_key))


async def _tick_guard(pool, snapshot, owned: set[str]) -> list:
    """The chain reader's order: detect swaps on the prior state, confirm, ledger, refresh."""
    from chain_guard import swap as guard_swap

    candidates = guard_swap.find_swaps(await guard_db.load_uid_state(pool), snapshot)
    confirmed = [s for s in candidates if s.old_hotkey not in owned]  # confirm_swaps
    await guard_db.record_swaps(pool, confirmed, 97, 9_999)
    await guard_db.refresh_registration_blocks(pool, snapshot)
    return candidates


def test_a_hotkey_swap_is_still_banned_after_the_registration_changes(pool_factory, monkeypatch):
    """swap_hotkey keeps BlockAtRegistration, so the new hotkey cannot pass as a fresh
    registration; the guard must ban it even when it already has a stale miners row."""
    from chain_guard import uploads as guard_uploads
    from chain_reader import db as chain_db
    from chain_reader.chain import Commit

    monkeypatch.setattr(guard_uploads, "put_detection", lambda *a, **k: None)

    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, uid=7, registration_block=OLD_REG)  # HOTKEY holds uid 7
            await conn.execute(
                "INSERT INTO miners (hotkey, uid, netuid, registration_block)"
                " VALUES ($1, 142, 97, 500)",
                SWAPPED_IN,
            )
        candidates = await _tick_guard(pool, [(7, SWAPPED_IN, OLD_REG)], owned=set())
        commit = Commit(97, 1_600, "0x", None, 7, SWAPPED_IN, {}, "hf://w@r", "p-w")
        await chain_db.insert_new_commits(pool, [commit])
        fault = await pool.fetchval(
            "SELECT fault_code FROM model_submissions WHERE hotkey = $1", SWAPPED_IN
        )
        activate = PrivateSignal(
            "activate",
            97,
            1_700,
            7,
            SWAPPED_IN,
            activation_signal_payload(bytes(SigningKey(b"k" * 32).verify_key)),
        )
        activated = await intake.handle_signals(pool, [activate])
        await pool.close()
        return candidates, fault, activated

    candidates, fault, activated = _run(go())
    assert [(s.uid, s.old_hotkey, s.new_hotkey) for s in candidates] == [(7, HOTKEY, SWAPPED_IN)]
    assert fault == "hotkey_swap"
    assert activated == 0


def test_reregistering_elsewhere_in_the_block_someone_takes_the_old_uid_is_not_a_swap(
    pool_factory, monkeypatch
):
    """The refresh gives a stale-uid row its new registration block, so the old uid can
    look swapped; the ownership check (the hotkey is still registered) must reject it."""
    from chain_guard import uploads as guard_uploads

    monkeypatch.setattr(guard_uploads, "put_detection", lambda *a, **k: None)

    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, uid=7, registration_block=OLD_REG)
        # HOTKEY re-registers at uid 44 in block NEW_REG; another hotkey takes uid 7 then too
        snapshot = [(44, HOTKEY, NEW_REG), (7, SWAPPED_IN, NEW_REG)]
        first = await _tick_guard(pool, snapshot, owned={HOTKEY})
        second = await _tick_guard(pool, snapshot, owned={HOTKEY})
        banned = await pool.fetchval("SELECT count(*) FROM used_hotkeys")
        await pool.close()
        return first, second, banned

    first, second, banned = _run(go())
    assert first == []
    assert [(s.uid, s.old_hotkey) for s in second] == [(7, HOTKEY)]  # candidate only
    assert banned == 0
