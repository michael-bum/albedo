from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from common.king_meta import CURRENT_FILE
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KingAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_prefix="KING_AGENT_",
        extra="ignore",
    )

    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 9203
    database_url: str = ""
    engine_url: str = "http://127.0.0.1:9201"
    models_dir: str = Field(
        default="/root/albedo-kings",
        validation_alias=AliasChoices("KING_ENGINE_MODELS_DIR", "KING_AGENT_MODELS_DIR"),
    )
    served_model_name: str = "albedo-king"
    max_model_len: int = 262144

    global_parallel: int = 0
    max_tokens_cap: int = 8192
    max_prompt_chars: int = 0
    max_body_bytes: int = 2_000_000
    retry_after_s: int = 60
    enable_thinking: bool = False
    agent_model: str = "albedo-king-agent"
    agent_rc_marker: bool = False
    agent_dup_nudge: int = 2
    agent_dup_stop: int = 4
    agent_thought_nudge: int = 3
    agent_sampling: str = '{"temperature": 1.0}'
    agent_loop_repetition_penalty: float = 1.0

    trace_dir: str = "/root/king-agent-traces"

    @property
    def current_king_file(self) -> Path:
        return Path(self.models_dir) / CURRENT_FILE

    @property
    def trace_root(self) -> str:
        return self.trace_dir if self.debug else ""


@lru_cache
def get_king_agent_settings() -> KingAgentSettings:
    return KingAgentSettings()
