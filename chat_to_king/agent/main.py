#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402
from agent.api import create_ide_app  # noqa: E402
from agent.config import get_king_agent_settings  # noqa: E402
from common.engine_client import EngineClient  # noqa: E402
from common.king_meta import KingInfoSource  # noqa: E402
from loguru import logger  # noqa: E402


def main() -> None:
    settings = get_king_agent_settings()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if settings.debug else "INFO")
    king = KingInfoSource(settings.current_king_file)
    info = king.get()
    logger.info(
        "[king-agent] starting: api :{} -> engine {} king={} debug={}",
        settings.port,
        settings.engine_url,
        info.roman if info else "?",
        settings.debug,
    )
    app = create_ide_app(EngineClient(settings.engine_url), settings, king=king)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
