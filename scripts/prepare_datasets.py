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


_MINI_CODER_LEAKS = ("modin-project__modin.8c7799fd.pr_7434", "scrapy__scrapy.35212ec5.pr_6671")
_OPEN_SWE_LEAKS = (
    "agronholm__anyio-935",
    "astropy__ccdproc-901",
    "beeware__briefcase-2302",
    "beeware__briefcase-2401",
    "conan-io__conan-18327",
    "conan-io__conan-18444",
    "ethereum__web3.py-3690",
    "geopandas__geopandas-3591",
    "matthewwithanm__python-markdownify-230",
    "pdm-project__pdm-3575",
    "pydata__sparse-870",
)

SOURCES: dict[str, dict] = {
    "mini-coder": {
        "language": "python",
        "repos": ["ricdomolm/mini-coder-trajs-400k"],
        "shard_glob": "data/train-*.parquet",
        "exclude_ids": _MINI_CODER_LEAKS,
    },
    "open-swe-traces": {
        "repos": ["nvidia/Open-SWE-Traces"],
        "shard_glob": "data/train-*.parquet",
        "raw_glob": "data/**/train-*.parquet",
        "render": True,
        "family": "pr",
        "exclude_ids": _OPEN_SWE_LEAKS,
    },
    "mini-coder-rs": {
        "language": "rust",
        "repos": [
            "AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories",
            "AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories",
            "AlienKevin/SWE-smith-rs-gemini-3-flash-trajectories",
        ],
        "shard_glob": "data/train-*.parquet",
        "render": True,
    },
    "swe-hero": {
        "language": "python",
        "repos": ["nvidia/SWE-Hero-openhands-trajectories"],
        "shard_glob": "data/train-*.parquet",
        "render": True,
        "family": "pr",
        "exclude_upstream": ("nebius/SWE-rebench",),
        "repo_cap": 200,
    },
}

REVISIONS: dict[str, str] = {
    "ricdomolm/mini-coder-trajs-400k": "c03c1fd5016d59be5f4acf6d4492d943d0633923",
    "nvidia/Open-SWE-Traces": "9c0e4579a4ee0effa3e5f7a552494a045f29377d",
    "nvidia/SWE-Hero-openhands-trajectories": "150bc119e52c647216fce285fd801f16b6fd745b",
    "AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories": "dfd98db8db1970d485d6897626648a90b54e453b",
    "AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories": "d4c902a41c7a73b230613932827e1908e06d162d",
    "AlienKevin/SWE-smith-rs-gemini-3-flash-trajectories": "0b2f075e7f65670b5f284e5d4264c4eabe05d91e",  # noqa: E501
}


def _repo_of(name: str) -> str:
    return SOURCES[name]["repos"][0]


def _enable_fast_transfer() -> None:

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _expected_parquet_shards(repo_id: str, shard_glob: str, revision: str) -> set[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return {f for f in files if fnmatch.fnmatch(f, shard_glob)}


def _local_parquet_shards(dest: Path, shard_glob: str) -> set[str]:
    return {p.relative_to(dest).as_posix() for p in dest.glob(shard_glob)}


def download_source(
    name: str,
    repo_id: str,
    shard_glob: str,
    root: Path,
    *,
    revision: str,
    force: bool,
    max_workers: int,
    dest_name: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    dest = root / (dest_name or name)

    expected = _expected_parquet_shards(repo_id, shard_glob, revision)
    if not expected:
        raise RuntimeError(f"{name}: no shards in repo {repo_id} matching {shard_glob!r}")
    present = _local_parquet_shards(dest, shard_glob)
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

    still_missing = expected - _local_parquet_shards(dest, shard_glob)
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
        glob = meta.get("raw_glob", meta["shard_glob"])
        for repo_id in meta["repos"]:
            download_source(
                name,
                repo_id,
                glob,
                Path(args.raw_root) if args.raw_root and meta.get("render") else root,
                revision=REVISIONS[repo_id],
                force=args.force,
                max_workers=args.max_workers,
                dest_name=repo_id.split("/")[-1] if meta.get("render") else name,
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
