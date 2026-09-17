#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import get_king_agent_settings  # noqa: E402
from common.token_meter import rollup_to_sqlite  # noqa: E402
from loguru import logger  # noqa: E402


def main() -> None:
    s = get_king_agent_settings()
    parser = argparse.ArgumentParser(prog="token_rollup")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--days", type=float, default=3.0, help="recompute this many days back")
    parser.add_argument("--db", default=s.database_url)
    parser.add_argument("--sqlite", default=str(Path(s.models_dir) / "token_windows.sqlite"))
    args = parser.parse_args()
    if not args.db:
        sys.exit("no database url (KING_AGENT_DATABASE_URL or --db)")
    n = rollup_to_sqlite(args.db, args.sqlite, args.hours, args.days)
    logger.info("[token-rollup] wrote {} closed {}h window rows to {}", n, args.hours, args.sqlite)


if __name__ == "__main__":
    main()
