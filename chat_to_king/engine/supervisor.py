#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.king_meta import (  # noqa: E402
    KingInfo,
    current_king_file,
    read_king_info,
    write_king_file,
)
from loguru import logger  # noqa: E402

from engine.config import KingEngineSettings, get_king_engine_settings  # noqa: E402
from engine.vllm_process import VllmProcess, busy_gpus  # noqa: E402


def served_names(settings: KingEngineSettings, info: KingInfo | None) -> list[str]:
    names = [settings.served_model_name]
    if info is not None and info.model_id and info.model_id not in names:
        names.append(info.model_id)
    return names


def check_model_path(settings: KingEngineSettings) -> KingInfo | None:
    if not settings.model_path:
        sys.exit("KING_ENGINE_MODEL_PATH is empty — run engine/king_ctl.py download <roman>")
    if not (Path(settings.model_path) / "config.json").is_file():
        sys.exit(f"KING_ENGINE_MODEL_PATH={settings.model_path} has no config.json")
    info = read_king_info(settings.model_path)
    current = current_king_file(settings.models_dir)
    if info is None:
        logger.warning("[king-engine] no albedo-king.json next to the weights; no roman to serve")
        current.unlink(missing_ok=True)
    else:
        write_king_file(current, info)
    return info


async def wait_gpus_free(settings: KingEngineSettings, timeout_s: float) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        busy = busy_gpus(settings.gpu_list, settings.gpu_busy_mib)
        if not busy:
            return
        if asyncio.get_event_loop().time() >= deadline:
            sys.exit(
                "[king-engine] refusing to start: GPUs still busy "
                + ", ".join(f"gpu{i}={used} MiB" for i, used in busy)
            )
        logger.warning("[king-engine] waiting for GPUs to free up: {}", busy)
        await asyncio.sleep(5.0)


async def run(settings: KingEngineSettings) -> None:
    info = check_model_path(settings)
    names = served_names(settings, info)
    proc = VllmProcess(settings)
    logger.info(
        "[king-engine] model {} king={} names={} gpus={} dp={} util={}",
        settings.model_path,
        info.roman if info else "?",
        names,
        settings.gpu_ids,
        settings.data_parallel_size,
        settings.gpu_util,
    )
    while True:
        proc.kill_port_squatter()
        await wait_gpus_free(settings, timeout_s=120.0)
        proc.start(settings.model_path, names)
        try:
            await proc.wait_healthy(settings.vllm_startup_s)
        except RuntimeError as exc:
            logger.error("[king-engine] {} — retrying in 30s", exc)
            proc.kill()
            await asyncio.sleep(30.0)
            continue
        logger.info("[king-engine] serving {} on :{}", names, settings.vllm_port)
        unhealthy = 0
        while proc.alive:
            await asyncio.sleep(settings.health_poll_s)
            if await proc.healthy():
                unhealthy = 0
                continue
            unhealthy += 1
            if unhealthy >= settings.unhealthy_restart_after:
                logger.error("[king-engine] unhealthy {}x — restarting vLLM", unhealthy)
                break
        proc.kill()
        logger.warning("[king-engine] vLLM down — restarting from {}", settings.model_path)


def main() -> None:
    asyncio.run(run(get_king_engine_settings()))


if __name__ == "__main__":
    main()
