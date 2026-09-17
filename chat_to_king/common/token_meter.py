from __future__ import annotations

import threading
import time
from collections.abc import Mapping

from loguru import logger

_DDL = """
CREATE TABLE IF NOT EXISTS token_usage (
    id BIGSERIAL PRIMARY KEY,
    ts DOUBLE PRECISION NOT NULL,
    service TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    stream BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS token_usage_ts ON token_usage (ts);
"""


class TokenMeter:
    def __init__(self, database_url: str, service: str) -> None:
        self.service = service
        self._url = database_url
        self._lock = threading.Lock()
        self._conn = None
        if database_url:
            self._connect()
            self._conn.execute(_DDL)

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def _connect(self) -> None:
        import psycopg

        self._conn = psycopg.connect(self._url, autocommit=True)

    def record(self, prompt_tokens: int, completion_tokens: int, stream: bool) -> None:
        if self._conn is None:
            return
        row = (time.time(), self.service, int(prompt_tokens), int(completion_tokens), bool(stream))
        sql = (
            "INSERT INTO token_usage (ts, service, prompt_tokens, completion_tokens, stream) "
            "VALUES (%s, %s, %s, %s, %s)"
        )
        with self._lock:
            try:
                self._conn.execute(sql, row)
            except Exception:
                try:
                    self._connect()
                    self._conn.execute(sql, row)
                except Exception as exc:
                    logger.warning("[token-meter] could not record usage: {}", exc)


def report(database_url: str, window_hours: float = 4.0, days: float = 2.0) -> list[Mapping]:
    import psycopg
    from psycopg.rows import dict_row

    bucket = window_hours * 3600.0
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        return conn.execute(
            "SELECT floor(ts / %s) * %s AS window_start, service, "
            "SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens, "
            "COUNT(*) AS requests FROM token_usage WHERE ts >= %s "
            "GROUP BY window_start, service ORDER BY window_start, service",
            (bucket, bucket, time.time() - days * 86400.0),
        ).fetchall()


def report_table(rows: list[Mapping], services: tuple[str, ...] = ("chat", "agent")) -> str:
    windows: dict[float, dict[str, Mapping]] = {}
    for r in rows:
        windows.setdefault(float(r["window_start"]), {})[r["service"]] = r
    head = "window (UTC)      " + "".join(
        f"{s + ' in':>12}{s + ' out':>12}{s + ' req':>10}" for s in services
    )
    lines = [head]
    totals = {s: [0, 0, 0] for s in services}
    for start in sorted(windows):
        line = time.strftime("%Y-%m-%d %H:%M", time.gmtime(start)) + "  "
        for s in services:
            r = windows[start].get(s)
            p, c, n = (
                (int(r["prompt_tokens"]), int(r["completion_tokens"]), int(r["requests"]))
                if r
                else (0, 0, 0)
            )
            totals[s][0] += p
            totals[s][1] += c
            totals[s][2] += n
            line += f"{p:>12,}{c:>12,}{n:>10,}"
        lines.append(line)
    lines.append(
        "total             "
        + "".join(f"{t[0]:>12,}{t[1]:>12,}{t[2]:>10,}" for t in totals.values())
    )
    return "\n".join(lines)


def rollup_to_sqlite(
    database_url: str, sqlite_path: str, window_hours: float = 4.0, days: float = 3.0
) -> int:
    import sqlite3
    from pathlib import Path

    bucket = window_hours * 3600.0
    closed_before = (time.time() // bucket) * bucket
    rows = [
        r
        for r in report(database_url, window_hours, days)
        if float(r["window_start"]) < closed_before
    ]
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS token_windows ("
            "window_start INTEGER NOT NULL, window_hours REAL NOT NULL, service TEXT NOT NULL, "
            "prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, requests INTEGER NOT NULL, "
            "updated_at INTEGER NOT NULL, PRIMARY KEY (window_start, window_hours, service))"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO token_windows VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    int(r["window_start"]),
                    window_hours,
                    r["service"],
                    int(r["prompt_tokens"]),
                    int(r["completion_tokens"]),
                    int(r["requests"]),
                    int(time.time()),
                )
                for r in rows
            ],
        )
    return len(rows)
