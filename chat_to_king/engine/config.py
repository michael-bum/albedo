from __future__ import annotations

import sys
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KingEngineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="KING_ENGINE_", extra="ignore")

    model_path: str = ""
    models_dir: str = "/root/albedo-kings"
    served_model_name: str = "albedo-king"

    vllm_host: str = "127.0.0.1"
    vllm_port: int = 9201
    vllm_python: str = sys.executable
    gpu_ids: str = "4,5,6"
    data_parallel_size: int = 3
    enable_expert_parallel: bool = True
    tensor_parallel_size: int = 1
    gpu_util: float = 0.95
    gpu_busy_mib: int = 1024
    api_server_count: int = 2
    vllm_dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    reasoning_parser: str = "qwen3"
    max_model_len: int = 262144
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    vllm_limit_mm: str = '{"image": 0, "video": 0}'
    vllm_moe_backend: str = "triton"
    vllm_quantization: str = ""
    vllm_enforce_eager: bool = False
    cpu_offload_gb: int = 0
    vllm_startup_s: float = 900.0
    health_poll_s: float = 5.0
    unhealthy_restart_after: int = 3
    download_timeout_s: float = 3600.0

    hf_namespace: str = Field(
        default="dendriteholdings",
        validation_alias=AliasChoices("KING_ENGINE_HF_NAMESPACE", "ALBEDO_KING_HF_NAMESPACE"),
    )
    hf_repo_prefix: str = Field(
        default="albedo-qwen3.6-35b-king",
        validation_alias=AliasChoices("KING_ENGINE_HF_REPO_PREFIX", "ALBEDO_KING_HF_REPO_PREFIX"),
    )

    @property
    def gpu_list(self) -> list[int]:
        return [int(g) for g in self.gpu_ids.split(",") if g.strip()]


@lru_cache
def get_king_engine_settings() -> KingEngineSettings:
    return KingEngineSettings()
