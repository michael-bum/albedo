from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chat_to_king"))

from common.engine_client import EngineClient, EngineState  # noqa: E402
from common.king_meta import (  # noqa: E402
    SIDECAR,
    KingInfo,
    KingInfoSource,
    read_king_info,
    write_king_info,
)
from common.notices import loading_text, openai_notice  # noqa: E402
from engine.config import KingEngineSettings  # noqa: E402
from engine.supervisor import check_model_path, served_names  # noqa: E402
from engine.vllm_process import build_command, busy_gpus  # noqa: E402


def test_sidecar_roundtrip_and_model_id(tmp_path):
    info = KingInfo(
        roman="CXXV",
        repo="dendriteholdings/albedo-qwen3.6-35b-king-cxxv",
        sha="a" * 40,
        hotkey="5F",
    )
    write_king_info(tmp_path, info)
    assert json.loads((tmp_path / SIDECAR).read_text())["roman"] == "CXXV"
    back = read_king_info(tmp_path)
    assert back == info and back.model_id == "albedo-king-cxxv"
    assert back.public()["model_id"] == "albedo-king-cxxv" and back.public()["hotkey"] == "5F"
    assert read_king_info(tmp_path / "missing") is None
    (tmp_path / SIDECAR).write_text("{}")
    assert read_king_info(tmp_path) is None


def test_king_info_source_follows_file_changes(tmp_path):
    src = KingInfoSource(tmp_path / SIDECAR)
    assert src.get() is None
    write_king_info(tmp_path, KingInfo(roman="cxxv", repo="r", sha="s"))
    assert src.get().roman == "CXXV"
    import os
    import time

    write_king_info(tmp_path, KingInfo(roman="CXXVI", repo="r", sha="s"))
    os.utime(tmp_path / SIDECAR, (time.time() + 5, time.time() + 5))
    assert src.get().roman == "CXXVI"
    assert KingInfoSource(fixed=KingInfo(roman="X", repo="r", sha="s")).get().roman == "X"
    assert KingInfoSource().get() is None


def test_served_names_and_command_flags(tmp_path):
    s = KingEngineSettings(
        model_path=str(tmp_path), gpu_ids="4,5,6", data_parallel_size=3, api_server_count=2
    )
    info = KingInfo(roman="CXXV", repo="r", sha="s")
    assert served_names(s, info) == ["albedo-king", "albedo-king-cxxv"]
    assert served_names(s, None) == ["albedo-king"]
    cmd = build_command(s, "/m", served_names(s, info), "/t.jinja")
    joined = " ".join(cmd)
    assert "--served-model-name albedo-king albedo-king-cxxv --host" in joined
    assert "--data-parallel-size 3 --enable-expert-parallel" in joined
    assert "--api-server-count 2" in joined
    single = build_command(KingEngineSettings(data_parallel_size=1), "/m", ["albedo-king"], "/t")
    assert "--enable-expert-parallel" not in " ".join(single)
    assert "--tensor-parallel-size" not in joined and "--generation-config" not in joined
    assert "--gpu-memory-utilization 0.95" in joined and "--max-model-len 262144" in joined
    assert "--reasoning-parser qwen3" in joined and "--moe-backend triton" in joined
    assert s.gpu_list == [4, 5, 6]


def test_busy_gpus_parses_nvidia_smi():
    def fake_run(cmd, **kw):
        assert cmd[-1] == "4,5,6" and "--query-gpu=index,memory.used" in cmd
        return SimpleNamespace(stdout="4, 3\n5, 89725\n6, 512\n", returncode=0)

    assert busy_gpus([4, 5, 6], 1024, run=fake_run) == [(5, 89725)]
    assert busy_gpus([4, 5, 6], 100_000, run=fake_run) == []


@pytest.mark.anyio
async def test_engine_client_state(monkeypatch):
    status = {"code": 503}

    def handler(request):
        return httpx.Response(status["code"])

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = transport
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    c = EngineClient("http://engine:9201/", cache_s=0.0)
    assert await c.state() == EngineState.LOADING
    status["code"] = 200
    assert await c.state() == EngineState.SERVING
    assert c.base_url == "http://engine:9201"


@pytest.mark.anyio
async def test_notice_text_and_shapes():
    assert "King CXXV" in loading_text(KingInfo(roman="CXXV", repo="r", sha="s"))
    assert "resend" in loading_text(None)
    body = json.loads(openai_notice("hi", "albedo-king-cxxv", False).body)
    assert body["model"] == "albedo-king-cxxv" and body["choices"][0]["message"]["content"] == "hi"
    resp = openai_notice("hi", "m", True)
    chunks = "".join([c if isinstance(c, str) else c.decode() async for c in resp.body_iterator])
    assert chunks.count("data: ") == 3 and chunks.endswith("data: [DONE]\n\n")


def test_supervisor_publishes_current_king(tmp_path):
    model = tmp_path / "hf" / "ns" / "repo" / "sha"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    write_king_info(model, KingInfo(roman="CXXV", repo="ns/repo", sha="sha"))
    s = KingEngineSettings(model_path=str(model), models_dir=str(tmp_path))
    info = check_model_path(s)
    assert info.roman == "CXXV"
    assert KingInfoSource(tmp_path / "current_king.json").get().roman == "CXXV"
    (model / SIDECAR).unlink()
    assert check_model_path(s) is None and not (tmp_path / "current_king.json").exists()
