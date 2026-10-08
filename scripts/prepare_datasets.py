#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("prepare_datasets")


# benchmark tasks (or the same repo + PR under another id) found in our sources by
# scripts/benchmark_leaks.py; rerun it whenever a source or a benchmark changes. Excluded from every
# source, since the sampler pools all sources' rows by instance id
LEAKS = frozenset(Path(__file__).with_name("benchmark_leaks.txt").read_text().split())

# every source pins each of its upstream repos to its own revision, so adding a source that
# shares a repo (open-swe-traces-v1.1 vs -v1.0) never moves another source's snapshot
SOURCES: dict[str, dict] = {
    "mini-coder": {
        "language": "python",
        "repos": {"ricdomolm/mini-coder-trajs-400k": "c03c1fd5016d59be5f4acf6d4492d943d0633923"},
        "shard_glob": "data/train-*.parquet",
    },
    "open-swe-traces-v1.0": {
        "repos": {"nvidia/Open-SWE-Traces": "9c0e4579a4ee0effa3e5f7a552494a045f29377d"},
        "shard_glob": "data/train-*.parquet",
        "raw_glob": "data/**/train-*.parquet",
        "render": True,
        "family": "pr",
    },
    "open-swe-traces-v1.1": {
        "repos": {"nvidia/Open-SWE-Traces": "f8fb5b3d2c787f85f8a00f5fe04fe3f1a11088ef"},
        "shard_glob": "data/train-*.parquet",
        "raw_glob": (
            "data/*/qwen36_27b/*/train-*.parquet",
            "data/*/deepseek_v4_flash/*/train-*.parquet",
        ),
        "raw_dir": "Open-SWE-Traces-v1.1",
        "render": True,
        "family": "pr",
    },
    "open-swe-traces-v1.2": {
        "repos": {"nvidia/Open-SWE-Traces": "f8fb5b3d2c787f85f8a00f5fe04fe3f1a11088ef"},
        "shard_glob": "data/train-*.parquet",
        "raw_glob": ("data/*/qwen38_27b/*/train-*.parquet",),
        "raw_dir": "Open-SWE-Traces-v1.2",
        "render": True,
        "family": "pr",
    },
    "mini-coder-rs": {
        "language": "rust",
        "repos": {
            "AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories": (
                "dfd98db8db1970d485d6897626648a90b54e453b"
            ),
            "AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories": (
                "d4c902a41c7a73b230613932827e1908e06d162d"
            ),
            "AlienKevin/SWE-smith-rs-gemini-3-flash-trajectories": (
                "0b2f075e7f65670b5f284e5d4264c4eabe05d91e"
            ),
        },
        "shard_glob": "data/train-*.parquet",
        "render": True,
    },
    "swe-hero": {
        "language": "python",
        "repos": {
            "nvidia/SWE-Hero-openhands-trajectories": "150bc119e52c647216fce285fd801f16b6fd745b"
        },
        "shard_glob": "data/train-*.parquet",
        "render": True,
        "family": "pr",
        "exclude_upstream": ("nebius/SWE-rebench",),
        "repo_cap": 200,
    },
    # Affine's corpus (data.affine.io, epoch 84) and SWE-Lego's transcripts, pre-converted to
    # mini-coder's returncode turns in affine-dedup (no render step), solution leaks already
    # dropped. One source per kind of machine, since the scaffold is chosen by source name
    # (see DATASETS.md); swesmith tasks sit under `mini-coder-affine-*`, the only names their ids
    # parse under.
    "affine-openhands": {
        "repos": {"dendriteholdings/affine-openhands": "96acd3252883acd12920567b8c4b9055778dd679"},
        "shard_glob": "data/train-*.parquet",
    },
    "affine-mswea": {
        "repos": {"dendriteholdings/affine-mswea": "b3b887739c09a8c0e2aa9af090d3654ee3e9c396"},
        "shard_glob": "data/train-*.parquet",
    },
    "affine-bash": {
        "repos": {"dendriteholdings/affine-bash": "d3c630d9b2c7f46fe2aaab127f0a2d1313526872"},
        "shard_glob": "data/train-*.parquet",
    },
    "affine-tools": {
        "repos": {"dendriteholdings/affine-tools": "04da3db81f5cafaeec3e4702194d718ae557d25a"},
        "shard_glob": "data/train-*.parquet",
    },
    "mini-coder-affine-mswea": {
        "repos": {
            "dendriteholdings/mini-coder-affine-mswea": "999978e2a610217a3e600585696986ff1115344f",
        },
        "shard_glob": "data/train-*.parquet",
    },
    "mini-coder-affine-bash": {
        "repos": {
            "dendriteholdings/mini-coder-affine-bash": "7b92128f54d8fe5cf7ba04dc2829d4ee4475178b",
        },
        "shard_glob": "data/train-*.parquet",
    },
    "mini-coder-affine-tools": {
        "repos": {
            "dendriteholdings/mini-coder-affine-tools": "3f65a975428b9f6b470313299504b47a07608335",
        },
        "shard_glob": "data/train-*.parquet",
    },
}


def _repo_of(name: str) -> str:
    return next(iter(SOURCES[name]["repos"]))


def raw_globs(spec: dict) -> tuple[str, ...]:
    globs = spec.get("raw_glob", spec["shard_glob"])
    return (globs,) if isinstance(globs, str) else tuple(globs)


def raw_dir(spec: dict, repo_id: str) -> str:
    """Where a render source's raw snapshot of `repo_id` lives under the raw root. A source that
    shares its repo with another names its own directory, so neither globs the other's files."""
    return spec.get("raw_dir") or repo_id.split("/")[-1]


def _enable_fast_transfer() -> None:

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _expected_parquet_shards(repo_id: str, globs: tuple[str, ...], revision: str) -> set[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return {f for f in files if any(fnmatch.fnmatch(f, g) for g in globs)}


def _local_parquet_shards(dest: Path, globs: tuple[str, ...]) -> set[str]:
    return {p.relative_to(dest).as_posix() for g in globs for p in dest.glob(g)}


def download_source(
    name: str,
    repo_id: str,
    globs: tuple[str, ...],
    root: Path,
    *,
    revision: str,
    force: bool,
    max_workers: int,
    dest_name: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    dest = root / (dest_name or name)

    expected = _expected_parquet_shards(repo_id, globs, revision)
    if not expected:
        raise RuntimeError(f"{name}: no shards in repo {repo_id} matching {globs!r}")
    present = _local_parquet_shards(dest, globs)
    to_fetch = sorted(expected) if force else sorted(expected - present)

    if not to_fetch:
        log.info(
            "%s: complete (%d/%d shards present) — skipping", name, len(present), len(expected)
        )
        return dest
    log.info(
        "%s: %d/%d present, downloading %d missing -> %s (%d workers) rev=%s",
        name,
        len(present),
        len(expected),
        len(to_fetch),
        dest,
        max_workers,
        revision[:12],
    )

    def _one(rel: str) -> None:
        hf_hub_download(
            repo_id,
            rel,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest),
            force_download=force,
            token=os.environ.get("HF_TOKEN"),
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, rel) for rel in to_fetch]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(to_fetch):
                log.info("%s: %d/%d downloaded", name, done, len(to_fetch))

    still_missing = expected - _local_parquet_shards(dest, globs)
    if still_missing:
        raise RuntimeError(
            f"{name}: {len(still_missing)} shard(s) still missing after download, "
            f"e.g. {sorted(still_missing)[:3]}"
        )
    log.info("%s: done (%d shards)", name, len(expected))
    return dest


def _upload_manifest(manifest_path: Path, key: str) -> str:
    import boto3
    from botocore.config import Config

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from albedo_config import get_model_validation_settings

    hv = get_model_validation_settings()

    if not (hv.S3_BUCKET and hv.S3_ACCESS_KEY and hv.S3_SECRET_KEY):
        raise SystemExit(
            "--upload needs S3 credentials: set ALBEDO_S3_BUCKET, ALBEDO_S3_ACCESS_KEY and "
            "ALBEDO_S3_SECRET_KEY (in albedo/.env). ALBEDO_S3_ENDPOINT decides WHERE it lands: "
            "prod points it at R2, served as https://albedo.tech/<key>. Left unset it defaults to "
            "the legacy s3.hippius.com, which is not what anyone reads."
        )

    body = manifest_path.read_bytes()
    client = boto3.client(
        "s3",
        endpoint_url=hv.S3_ENDPOINT,
        aws_access_key_id=hv.S3_ACCESS_KEY,
        aws_secret_access_key=hv.S3_SECRET_KEY,
        region_name="decentralized",
        config=Config(
            connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}
        ),
    )
    client.put_object(
        Bucket=hv.S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
        ACL="public-read",
    )
    log.info("uploaded %s (%s) sha256 %s", key, hv.S3_ENDPOINT, hashlib.sha256(body).hexdigest())
    return f"s3://{hv.S3_BUCKET}/{key}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download eval datasets from HuggingFace and build the combined manifest locally."  # noqa: E501
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dir to download <source>/data/*.parquet into and write manifest.json (local only).",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Where render sources keep their raw HF snapshot (default: --dataset-root).",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SOURCES),
        help=f"Comma-separated source names to fetch (default: all of {','.join(SOURCES)}).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if shards already exist."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Concurrent files downloaded in parallel (default: 16). Raise for many small shards.",
    )
    parser.add_argument(
        "--skip-manifest", action="store_true", help="Only download; do not build manifest.json."
    )
    parser.add_argument(
        "--out", default=None, help="Manifest output path (default: <dataset-root>/manifest.json)."
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload manifest.json and manifest.meta.json to the albedo bucket (ALBEDO_S3_* creds); does not upload the shards.",  # noqa: E501
    )
    parser.add_argument(
        "--upload-key",
        default="datasets/manifest.json",
        help="Destination key for --upload (default: datasets/manifest.json). The meta is published beside it.",  # noqa: E501
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _enable_fast_transfer()

    root = Path(args.dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}; known: {list(SOURCES)}")

    for name in names:
        meta = SOURCES[name]
        for repo_id, revision in meta["repos"].items():
            download_source(
                name,
                repo_id,
                raw_globs(meta),
                Path(args.raw_root) if args.raw_root and meta.get("render") else root,
                revision=revision,
                force=args.force,
                max_workers=args.max_workers,
                dest_name=raw_dir(meta, repo_id) if meta.get("render") else name,
            )
        if meta.get("render"):
            from render_trajectories import render_source

            stats = render_source(name, Path(args.raw_root or root), root)
            log.info(
                "%s: rendered %s/%s rows -> %s shards",
                name,
                stats["rows_out"],
                stats["rows_in"],
                stats["shards_written"],
            )

    out_path = Path(args.out) if args.out else root / "manifest.json"

    if args.skip_manifest:
        log.info("skipping manifest build (--skip-manifest)")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from build_manifest import print_manifest_summary, write_manifest

        out_path, manifest, digest = write_manifest(
            root, names, out_path=out_path, max_workers=args.max_workers
        )
        print_manifest_summary(out_path, manifest, digest)

    if args.upload:
        if not out_path.exists():
            raise SystemExit(
                f"--upload: no manifest at {out_path} (build one first, or drop --skip-manifest)."
            )
        _upload_manifest(out_path, args.upload_key)
        meta_path = out_path.with_name("manifest.meta.json")
        if meta_path.exists():
            _upload_manifest(meta_path, args.upload_key.removesuffix(".json") + ".meta.json")


if __name__ == "__main__":
    main()
