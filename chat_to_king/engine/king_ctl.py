#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.king_meta import SIDECAR, KingInfo, read_king_info, write_king_info  # noqa: E402

from engine.config import KingEngineSettings, get_king_engine_settings  # noqa: E402

_SETTINGS = get_king_engine_settings()
os.environ["ALBEDO_MODEL_CACHE_DIR"] = _SETTINGS.models_dir
os.environ["CV_MODEL_CACHE_DIR"] = _SETTINGS.models_dir

from engine.materialize import King, materialize, model_complete  # noqa: E402
from engine.mirror import mirror_origin, mirror_repo_id, mirror_revision  # noqa: E402


def resolve(settings: KingEngineSettings, roman: str, repo: str, revision: str) -> King:
    if not repo:
        repo = mirror_repo_id(roman, settings)
    sha = revision or mirror_revision(repo, "", None)
    return King(repo=repo, sha=sha, roman=roman.upper())


def on_disk(settings: KingEngineSettings) -> list[tuple[str, str, str, float, Path]]:
    found = []
    for cfg in Path(settings.models_dir).rglob("config.json"):
        d = cfg.parent
        if not model_complete(str(d)) or len(d.parts) < 3:
            continue
        info = read_king_info(d)
        repo = info.repo if info else "/".join(d.parts[-3:-1])
        roman = info.roman if info else ""
        size = sum(f.stat().st_size for f in d.glob("*.safetensors")) / 2**30
        found.append((roman, repo, d.name, size, d))
    return sorted(found)


def engine_models(settings: KingEngineSettings) -> list[str] | None:
    try:
        r = httpx.get(f"http://127.0.0.1:{settings.vllm_port}/v1/models", timeout=3.0)
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return None


def cmd_download(settings: KingEngineSettings, args) -> None:
    king = resolve(settings, args.roman, args.repo, args.revision)
    print(f"resolved {king.roman or king.repo} -> {king.repo}@{king.sha}")
    dest = materialize(king)
    original_repo, hotkey = mirror_origin(king.repo, king.sha)
    info = KingInfo(
        roman=king.roman,
        repo=king.repo,
        sha=king.sha,
        hotkey=hotkey,
        original_repo=original_repo,
        downloaded_at=time.time(),
    )
    write_king_info(dest, info)
    print(f"ready at {dest} ({SIDECAR} written: {info.model_id})")
    print("to serve it: put this line in the repo .env, then `pm2 restart albedo-king-engine`:")
    print(f"KING_ENGINE_MODEL_PATH={dest}")


def cmd_list(settings: KingEngineSettings, _args) -> None:
    served = engine_models(settings)
    current = Path(settings.model_path).resolve() if settings.model_path else None
    for roman, repo, sha, size, d in on_disk(settings):
        marks = []
        if current and d.resolve() == current:
            marks.append("env")
        if served and roman and f"albedo-king-{roman.lower()}" in served:
            marks.append("serving")
        print(f"{roman or '?':<10} {repo}@{sha[:12]}  {size:6.1f} GiB  {' '.join(marks)}")
    print(f"env: {settings.model_path or '(KING_ENGINE_MODEL_PATH not set)'}")
    served_txt = "down" if served is None else "serving " + ", ".join(served)
    print(f"engine :{settings.vllm_port}: {served_txt}")


def cmd_status(settings: KingEngineSettings, _args) -> None:
    info = read_king_info(settings.model_path) if settings.model_path else None
    print(f"model_path: {settings.model_path or '-'}")
    print(f"king: {info.public() if info else None}")
    served = engine_models(settings)
    print(f"engine: {'down' if served is None else 'serving ' + ', '.join(served)}")


def cmd_prune(settings: KingEngineSettings, args) -> None:
    target = args.roman.upper()
    current = Path(settings.model_path).resolve() if settings.model_path else None
    victims = [(r, d) for r, _repo, _sha, _size, d in on_disk(settings) if r == target]
    if not victims:
        sys.exit(f"{target} is not on disk")
    for roman, d in victims:
        if current and d.resolve() == current:
            sys.exit(f"{roman} is in KING_ENGINE_MODEL_PATH; point the env elsewhere first")
        shutil.rmtree(d, ignore_errors=True)
        print(f"removed {roman} at {d}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="king_ctl")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download")
    p.add_argument("roman", nargs="?", default="")
    p.add_argument("--repo", default="")
    p.add_argument("--revision", default="")
    p.set_defaults(fn=cmd_download)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("prune")
    p.add_argument("roman")
    p.set_defaults(fn=cmd_prune)
    args = parser.parse_args()
    if getattr(args, "roman", None) == "" and not getattr(args, "repo", ""):
        parser.error("give a roman numeral (e.g. CXXV) or --repo")
    args.fn(_SETTINGS, args)


if __name__ == "__main__":
    main()
