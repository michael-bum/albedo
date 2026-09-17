from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from sanity_remote.worker import _inject_seed_processor_files, _model_present


@dataclass(frozen=True)
class King:
    repo: str
    sha: str
    roman: str = ""


def model_complete(model_dir: str) -> bool:
    return _model_present(model_dir) and (Path(model_dir) / "config.json").exists()


def model_dir_for(king: King) -> str:
    from model_validation.storage import cache_dir, make_ref

    return str(cache_dir(make_ref(king.repo, king.sha)))


def materialize(king: King) -> str:
    from model_validation.storage import download_config, download_full, make_ref

    ref = make_ref(king.repo, king.sha)
    dest = model_dir_for(king)
    if model_complete(dest):
        logger.info("[king-engine] reusing on-disk model at {} — skipping download", dest)
    else:
        logger.info("[king-engine] downloading {} rev={:.16} to {}", king.repo, king.sha, dest)
        download_config(ref)
        dest = download_full(ref)
    _inject_seed_processor_files(dest)
    return dest
