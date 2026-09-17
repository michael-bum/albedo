#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402
from common.engine_client import EngineClient  # noqa: E402
from common.king_meta import KingInfoSource  # noqa: E402
from loguru import logger  # noqa: E402
from web.config import get_king_chat_settings  # noqa: E402
from web.gateway import create_app  # noqa: E402


def main() -> None:
    settings = get_king_chat_settings()
    king = KingInfoSource(settings.current_king_file)
    info = king.get()
    logger.info(
        "[king-chat] starting: gateway :{} -> engine {} king={}",
        settings.gateway_port,
        settings.engine_url,
        info.roman if info else "?",
    )
    app = create_app(settings, EngineClient(settings.engine_url), king)
    meter_state = "on" if settings.database_url else "off (no KING_CHAT_DATABASE_URL)"
    logger.info("[king-chat] token meter {}", meter_state)
    uvicorn.run(app, host=settings.gateway_host, port=settings.gateway_port, log_level="info")


if __name__ == "__main__":
    main()
