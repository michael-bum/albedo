from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg.rows import dict_row

_HERE = Path(__file__).resolve().parent


class _Db:
    def __init__(self, target: str) -> None:
        if not target:
            raise ValueError("KING_AGENT_DATABASE_URL is required (postgresql://…)")
        self._target = target
        self._conn = None
        self._connect()

    def _connect(self) -> None:
        self._conn = psycopg.connect(self._target, row_factory=dict_row)

    def execute(self, sql: str, params: tuple = ()):
        try:
            return self._conn.execute(sql, params)
        except psycopg.OperationalError:
            self._connect()
            return self._conn.execute(sql, params)

    def fetchone(self, sql: str, params: tuple = ()):
        return self.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list:
        return self.execute(sql, params).fetchall()

    def insert_id(self, sql: str, params: tuple) -> int:
        return int(self.execute(sql + " RETURNING id", params).fetchone()["id"])

    def rowcount(self, sql: str, params: tuple) -> int:
        return self.execute(sql, params).rowcount

    def commit(self) -> None:
        self._conn.commit()

    def apply_schema(self) -> None:
        self._conn.execute((_HERE / "schema.sql").read_text(encoding="utf-8"))
        self._conn.commit()
