from __future__ import annotations

from pathlib import Path

from loguru import logger

from config_validation.models import ModelRef
from config_validation.storage import download_config, list_files
from model_validation.validate import (
    check_chat_template,
    check_dtype,
    check_genesis,
    check_index,
    check_repo,
)


def _result(checks: dict[str, tuple[bool, str]]) -> tuple[bool, dict]:
    ok = all(passed for passed, _ in checks.values())
    return ok, {name: {"ok": passed, "reason": msg} for name, (passed, msg) in checks.items()}


def validate_local(path: str, files: list[str] | None = None) -> tuple[bool, dict]:
    """Every check the validator runs except dedup, in the validator's order. Keys are the
    fault codes the validator would report, so a local FAIL names the on-chain rejection.

    `files` is the inventory the backend will actually publish, relative to `path`. Pass it
    whenever the uploader's file set differs from the directory's own — the private store
    drops dotfiles and uploads subdirectories, so judging the raw directory there both
    rejects files it will strip and misses files it will send.
    """
    logger.info(f"validating local model: {path}")
    if files is None:
        files = [p.name for p in Path(path).iterdir() if p.is_file()]
    logger.info("checking file manifest…")
    files_res = check_repo(files)
    logger.info("checking weight dtype…")
    dtype_res = check_dtype(path)
    logger.info("checking chat template…")
    template_res = check_chat_template(path, files)
    logger.info("checking genesis metadata…")
    genesis_res = check_genesis(path, files)
    logger.info("checking safetensors index…")
    index_res = check_index(path)
    return _result(
        {
            "file_manifest": files_res,
            "weight_dtype": dtype_res,
            "chat_template_hash": template_res,
            "metadata_hash": genesis_res,
            "safetensors_index": index_res,
        }
    )


def require_valid(
    path: str, *, files: list[str] | None = None, skip: bool = False, log=print
) -> bool:
    """Gate a publish path on validate_local. Advisory only: the validator re-runs every
    check, plus the near-duplicate check that cannot run locally."""
    if skip:
        log("model check skipped (--skip-check)")
        return True
    ok, res = validate_local(path, files)
    for name, v in res.items():
        log(f"[{'PASS' if v['ok'] else 'FAIL'}] {name}" + ("" if v["ok"] else f" — {v['reason']}"))
    if not ok:
        log("model is INVALID — fix it, or re-run with --skip-check")
    return ok


def validate_remote(repo: str, digest: str) -> tuple[bool, dict]:
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    ref = ModelRef(repo=repo, digest=digest)
    logger.info(f"validating remote repo: {repo}@{digest} (backend={ref.backend})")
    try:
        logger.info("listing repo files…")
        files = list_files(ref)
    except RevisionNotFoundError:
        return _result(
            {
                "file_manifest": (False, f"digest not found: {digest} is not in {repo}"),
            }
        )
    except RepositoryNotFoundError:
        return _result(
            {
                "file_manifest": (False, f"repo not found: {repo}"),
            }
        )
    except Exception as exc:
        return _result(
            {
                "file_manifest": (False, f"could not read {repo}@{digest}: {exc}"),
            }
        )

    logger.info("checking file manifest…")
    checks = {"file_manifest": check_repo(files)}
    try:
        logger.info("downloading config files…")
        cfg_dir = download_config(ref)
    except Exception as exc:
        failed = (False, f"could not download config files: {exc}")
        return _result({**checks, "chat_template_hash": failed, "metadata_hash": failed})
    logger.info("checking chat template…")
    checks["chat_template_hash"] = check_chat_template(cfg_dir, files)
    logger.info("checking genesis metadata…")
    checks["metadata_hash"] = check_genesis(cfg_dir, files)
    return _result(checks)
