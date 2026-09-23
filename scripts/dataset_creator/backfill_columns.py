"""Rewrite every published chunk with the score columns and re-upload them.

The columns are re-derived from the run artifacts on disk: a chunk row is matched to its
artifact row by (sample_id, assistant turns), so a row keeps null only where its artifact never
recorded the fact (rows before 2026-09-22 have no per-run score or earned flags). Pending rows are
enriched the same way. One commit per model split, plus the dataset card when --readme is given.

    python scripts/dataset_creator/backfill_columns.py \\
        --artifacts-dir /root/albedo-remote/artifacts --upload \\
        --readme scripts/dataset_creator/dataset_card.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
from config import Config  # noqa: E402
from extract import extract_rows  # noqa: E402
from store import COLUMNS, _load_pending, _save_pending, write_chunk_file  # noqa: E402

EXTRA = [c for c in COLUMNS if c not in ("sample_id", "messages")]


def row_key(sample_id: str, messages: list[dict]) -> tuple:
    return (sample_id, tuple(m["content"] for m in messages if m["role"] == "assistant"))


def build_index(artifacts_dir: Path) -> dict[tuple, dict]:
    index: dict[tuple, dict] = {}
    run_dirs = sorted(
        d
        for d in artifacts_dir.iterdir()
        if (d / "scoring-results.jsonl").exists() and (d / "generated-samples.jsonl").exists()
    )
    for n, run_dir in enumerate(run_dirs, start=1):
        try:
            by_model = extract_rows(run_dir)
        except Exception as exc:  # a corrupt artifact must not stop the rest
            print(f"  skip {run_dir.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for rows in by_model.values():
            for row in rows:
                index[row_key(row["sample_id"], row["messages"])] = {k: row.get(k) for k in EXTRA}
        if n % 50 == 0:
            print(f"  indexed {n}/{len(run_dirs)} runs, {len(index)} rows", flush=True)
    return index


def enrich(row: dict, index: dict[tuple, dict]) -> tuple[dict, bool]:
    extra = index.get(row_key(row["sample_id"], row["messages"]))
    return {**{k: None for k in EXTRA}, **row, **(extra or {})}, extra is not None


def rewrite_chunk(path: Path, index: dict[tuple, dict]) -> tuple[int, int]:
    rows = pq.read_table(path).to_pylist()
    enriched = [enrich(r, index) for r in rows]
    tmp = path.with_suffix(".parquet.tmp")
    write_chunk_file([r for r, _ in enriched], tmp)
    os.replace(tmp, path)
    return sum(1 for _, hit in enriched if hit), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--artifacts-dir", required=True, type=Path)
    ap.add_argument("--upload", action="store_true", help="re-upload rewritten chunks to HF")
    ap.add_argument("--readme", type=Path, help="dataset card to upload as README.md")
    args = ap.parse_args()
    cfg = Config()
    print("indexing artifacts", flush=True)
    index = build_index(args.artifacts_dir)
    print(f"index: {len(index)} rows", flush=True)
    for model_dir in sorted(p for p in cfg.out_dir.iterdir() if p.is_dir()):
        matched = total = 0
        for chunk in sorted(model_dir.glob("*.parquet")):
            m, t = rewrite_chunk(chunk, index)
            matched += m
            total += t
        pending = [enrich(r, index)[0] for r in _load_pending(cfg, model_dir.name)]
        _save_pending(cfg, model_dir.name, pending)
        print(
            f"{model_dir.name}: {total} rows rewritten, {matched} matched an artifact, "
            f"{len(pending)} pending rows enriched",
            flush=True,
        )
    if not args.upload:
        return 0
    from huggingface_hub import HfApi

    token = os.environ.get("DATASET_CREATOR_HF_TOKEN") or os.environ.get("HF_TOKEN")
    repo = os.environ.get("DATASET_CREATOR_HF_REPO")
    if not token or not repo:
        raise RuntimeError("DATASET_CREATOR_HF_TOKEN and DATASET_CREATOR_HF_REPO must be set")
    api = HfApi(token=token)
    for model_dir in sorted(p for p in cfg.out_dir.iterdir() if p.is_dir()):
        api.upload_folder(
            folder_path=str(model_dir),
            path_in_repo=f"data/{model_dir.name}",
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=["*.parquet"],
            commit_message=f"add score columns to {model_dir.name}",
        )
        print(f"uploaded data/{model_dir.name}", flush=True)
    if args.readme:
        api.upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message="document the score columns",
        )
        print("uploaded README.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
