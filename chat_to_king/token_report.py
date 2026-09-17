#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import get_king_agent_settings  # noqa: E402
from common.token_meter import report, report_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(prog="token_report")
    parser.add_argument("--hours", type=float, default=4.0, help="window size in hours")
    parser.add_argument("--days", type=float, default=2.0, help="how far back")
    parser.add_argument("--db", default=get_king_agent_settings().database_url)
    args = parser.parse_args()
    if not args.db:
        sys.exit("no database url (KING_AGENT_DATABASE_URL or --db)")
    print(report_table(report(args.db, args.hours, args.days)))


if __name__ == "__main__":
    main()
