from __future__ import annotations

import os
import secrets

import pytest

_PG_URL = os.environ.get("KING_AGENT_TEST_DATABASE_URL")


@pytest.fixture
def pg_url():
    if not _PG_URL:
        pytest.skip("KING_AGENT_TEST_DATABASE_URL is not set")
    import psycopg

    name = f"t_{secrets.token_hex(6)}"
    with psycopg.connect(_PG_URL, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {name}")
    base, _, _ = _PG_URL.rpartition("/")
    yield f"{base}/{name}"
    with psycopg.connect(_PG_URL, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE {name} WITH (FORCE)")
