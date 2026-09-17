from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
from loguru import logger

from engine.config import KingEngineSettings

TEMPLATE_NAME = "albedo_chat_template.jinja"


def build_command(
    s: KingEngineSettings, model_path: str, served_names: list[str], template: str
) -> list[str]:
    cmd = [
        s.vllm_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        *served_names,
        "--host",
        s.vllm_host,
        "--port",
        str(s.vllm_port),
        "--gpu-memory-utilization",
        str(s.gpu_util),
        "--dtype",
        s.vllm_dtype,
        "--max-model-len",
        str(s.max_model_len),
        "--kv-cache-dtype",
        s.kv_cache_dtype,
        "--chat-template",
        template,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    if s.data_parallel_size > 1:
        cmd += ["--data-parallel-size", str(s.data_parallel_size)]
        if s.enable_expert_parallel:
            cmd += ["--enable-expert-parallel"]
    if s.tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(s.tensor_parallel_size)]
    if s.api_server_count > 1:
        cmd += ["--api-server-count", str(s.api_server_count)]
    if s.max_num_seqs > 0:
        cmd += ["--max-num-seqs", str(s.max_num_seqs)]
    if s.max_num_batched_tokens > 0:
        cmd += ["--max-num-batched-tokens", str(s.max_num_batched_tokens)]
    if s.reasoning_parser:
        cmd += ["--reasoning-parser", s.reasoning_parser]
    if s.vllm_limit_mm:
        cmd += ["--limit-mm-per-prompt", s.vllm_limit_mm]
    if s.cpu_offload_gb > 0:
        cmd += ["--cpu-offload-gb", str(s.cpu_offload_gb)]
    if s.vllm_quantization:
        cmd += ["--quantization", s.vllm_quantization]
    if s.vllm_enforce_eager:
        cmd += ["--enforce-eager"]
    if s.vllm_moe_backend:
        cmd += ["--moe-backend", s.vllm_moe_backend]
    return cmd


def busy_gpus(gpu_ids: list[int], max_used_mib: int, run=subprocess.run) -> list[tuple[int, int]]:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(g) for g in gpu_ids),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    busy = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        used = int(float(parts[1]))
        if used > max_used_mib:
            busy.append((int(parts[0]), used))
    return busy


class VllmProcess:
    def __init__(self, settings: KingEngineSettings) -> None:
        self._s = settings
        self._proc: subprocess.Popen | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def start(self, model_path: str, served_names: list[str]) -> None:
        from albedo_eval_service.remote.prompt_remote import QWEN3_CHAT_TEMPLATE

        s = self._s
        Path(s.models_dir).mkdir(parents=True, exist_ok=True)
        template = os.path.join(s.models_dir, TEMPLATE_NAME)
        with open(template, "w", encoding="utf-8") as fh:
            fh.write(QWEN3_CHAT_TEMPLATE)
        cmd = build_command(s, model_path, served_names, template)
        logger.info(
            "[king-engine] starting vLLM :{} gpus={} dp={} tp={} names={}",
            s.vllm_port,
            s.gpu_ids,
            s.data_parallel_size,
            s.tensor_parallel_size,
            served_names,
        )
        self._proc = subprocess.Popen(
            cmd,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": s.gpu_ids,
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
            start_new_session=True,
        )

    async def healthy(self) -> bool:
        if not self.alive:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"http://127.0.0.1:{self._s.vllm_port}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def wait_healthy(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.alive:
                raise RuntimeError(f"vLLM exited with code {self._proc.returncode} on startup")
            if await self.healthy():
                return
            await asyncio.sleep(2.0)
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s")

    def kill(self) -> None:
        if not self._proc:
            return
        logger.info("[king-engine] killing vLLM pid={}", self._proc.pid)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            pass
        self._proc = None

    def kill_port_squatter(self) -> None:
        try:
            with socket.socket() as sk:
                sk.settimeout(0.5)
                if sk.connect_ex(("127.0.0.1", self._s.vllm_port)) != 0:
                    return
        except Exception:
            return
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{self._s.vllm_port}"], capture_output=True, text=True, timeout=5
            )
            for pid_str in result.stdout.split():
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                    logger.info(
                        "[king-engine] killed orphan pid={} on :{}", pid_str, self._s.vllm_port
                    )
                except Exception:
                    pass
        except Exception:
            pass
