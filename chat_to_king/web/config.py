from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from common.king_meta import CURRENT_FILE
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KingChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KING_CHAT_", extra="ignore")

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 9202
    engine_url: str = "http://127.0.0.1:9201"
    models_dir: str = Field(
        default="/root/albedo-kings",
        validation_alias=AliasChoices("KING_ENGINE_MODELS_DIR", "KING_CHAT_MODELS_DIR"),
    )
    served_model_name: str = "albedo-king"
    retry_after_s: int = 60
    database_url: str = Field(
        default="",
        validation_alias=AliasChoices("KING_CHAT_DATABASE_URL", "KING_AGENT_DATABASE_URL"),
    )

    @property
    def current_king_file(self) -> Path:
        return Path(self.models_dir) / CURRENT_FILE

    llms_path: str = "website/llms.txt"
    llms_url: str = ""
    llms_keywords: str = "albedo,sn97,subnet 97,netuid 97"
    llms_max_chars: int = 0


@lru_cache
def get_king_chat_settings() -> KingChatSettings:
    return KingChatSettings()
