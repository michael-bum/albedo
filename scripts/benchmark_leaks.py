#!/usr/bin/env python3
"""Find training-trajectory rows whose task is (or derives from) a benchmark task we evaluate on.

Benchmark fingerprints are loaded from HuggingFace once and cached as JSON (``--cache``). Each
dataset is given as ``label=glob`` and every row is matched on three keys:

  id        normalized instance id equality (``owner__repo-N``, case-folded, ``_interface`` dropped)
  repo_pr   (owner/repo, PR number) taken from the id, the ``repo`` column or an AACR PR url
  text      word 8-shingle containment of a benchmark problem statement / instruction in the
            first user message (Terminal-Bench, Aider Polyglot and the SWE benchmarks)

plus ``exercism_repo`` (a trajectory on an ``exercism/*`` or polyglot-benchmark repository).
``--ids-out`` writes every matched id of every label in the report, one per line: run it over all
sources (or add them with ``--update``) to refresh ``scripts/benchmark_leaks.txt``.

    uv run scripts/benchmark_leaks.py --out /tmp/osw/leaks.json \\
        --ids-out scripts/benchmark_leaks.txt \\
        'open-swe-v1.0=/data/raw/Open-SWE-Traces/data/*/*.parquet' \\
        'mini-coder=/data/mini-coder/data/*.parquet'
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_CACHE = Path.home() / ".cache" / "albedo" / "benchmark_fingerprints.json"
CACHE_VERSION = 2

SHINGLE = 8
TEXT_MIN_CONTAINMENT = 0.5
TEXT_MIN_SHINGLES = 20  # or 80% of a short task's shingles, but never fewer than:
TEXT_FLOOR_SHINGLES = 8
TEXT_MAX_GROUP_DF = 2  # shingles shared by more benchmark tasks than this are boilerplate
MAX_TEXT_CHARS = 60_000

# --------------------------------------------------------------------------- normalization

_SMITH_RE = re.compile(r"^(?P<owner>.+?)__(?P<repo>.+?)\.[0-9a-f]{7,8}\.(?P<tail>.+)$")
_SWE_RE = re.compile(r"^(?P<owner>.+?)__(?P<repo>.+)-(?P<n>\d+)(?:_[a-z]+)?$")
_SCALE_RE = re.compile(r"_pr(?P<n>\d+)$")
_AACR_RE = re.compile(r"^(?P<slug>.+)-pr-(?P<n>\d+)(?:-[0-9a-f]+)?(?:__\w+)?$")
_PR_URL_RE = re.compile(r"github\.com/(?P<repo>[^/]+/[^/]+)/pull/(?P<n>\d+)")


def norm_id(instance_id: str) -> str:
    iid = instance_id.strip().lower()
    return re.sub(r"_(interface)$", "", iid)


def repo_slug(repo: str) -> str:
    """``FreeCAD/FreeCAD`` and AACR's ``freecad-freecad`` collapse to the same slug."""
    return re.sub(r"[^a-z0-9]+", "-", repo.strip().lower().replace("__", "/")).strip("-")


def pr_key(repo: str, number: int | str) -> str:
    return f"{repo_slug(repo)}#{int(number)}"


def parse_id(instance_id: str) -> tuple[str, int] | None:
    """(owner/repo, PR number) from SWE-bench, SWE-smith ``pr_N`` or AACR style ids."""
    iid = instance_id.strip()
    smith = _SMITH_RE.match(iid)
    if smith:
        tail = re.fullmatch(r"pr_(\d+)", smith["tail"])
        return (f"{smith['owner']}/{smith['repo']}", int(tail[1])) if tail else None
    swe = _SWE_RE.match(iid)
    if swe:
        return f"{swe['owner']}/{swe['repo']}", int(swe["n"])
    aacr = _AACR_RE.match(iid)
    if aacr and "__" not in aacr["slug"]:
        return aacr["slug"], int(aacr["n"])
    return None


def row_pr_keys(instance_id: str, repo: str | None) -> set[str]:
    keys: set[str] = set()
    parsed = parse_id(instance_id)
    if parsed:
        keys.add(pr_key(*parsed))
    if repo:
        number = parsed[1] if parsed else None
        if number is None:
            scale = _SCALE_RE.search(instance_id)
            number = int(scale["n"]) if scale else None
        if number is not None:
            keys.add(pr_key(repo, number))
    return keys


def norm_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    words = norm_text(text[:MAX_TEXT_CHARS]).split()
    return {" ".join(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def polyglot_instruction(text: str) -> str:
    """Drop the per-language 'Use the instructions above to modify ...' footer."""
    return re.split(r"\n-{4,}\n", text)[0]


# --------------------------------------------------------------------------- benchmark loading


def _task(repo=None, number=None, text=None, **extra) -> dict:
    return {
        "pr": pr_key(repo, number) if repo and number is not None else None,
        "text": norm_text(text) if text else None,
        **extra,
    }


def _hf_parquets(repo_id: str, prefix: str = "data/") -> list[tuple[str, str]]:
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return [
        (f, hf_hub_download(repo_id, f, repo_type="dataset"))
        for f in sorted(files)
        if f.startswith(prefix) and f.endswith(".parquet")
    ]


def _swe_tasks(repo_id: str) -> dict:
    tasks: dict[str, dict] = {}
    cols = ["instance_id", "repo", "problem_statement"]
    for name, path in _hf_parquets(repo_id):
        split = name.split("/")[-1].split("-")[0]
        for row in pq.read_table(path, columns=cols).to_pylist():
            iid = norm_id(row["instance_id"])
            parsed = parse_id(row["instance_id"])
            if iid in tasks:
                tasks[iid].setdefault("splits", []).append(split)
                continue
            tasks[iid] = _task(
                row["repo"],
                parsed[1] if parsed else None,
                row["problem_statement"],
                splits=[split],
            )
    return tasks


def _aacr_tasks() -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    tasks: dict[str, dict] = {}
    path = hf_hub_download("Alibaba-Aone/aacr-bench", "dataset.json", repo_type="dataset")
    for entry in json.loads(Path(path).read_text()):
        m = _PR_URL_RE.search(entry.get("pr_url") or "")
        if m:
            key = pr_key(m["repo"], m["n"])
            tasks.setdefault(key, _task(m["repo"], m["n"], url=m.group(0)))
    try:
        files = HfApi().list_repo_files("osolmaz/aacr-bench-harbor", repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"aacr-bench-harbor unavailable ({exc}); using Alibaba-Aone/aacr-bench only")
        files = []
    for task_id in sorted({f.split("/")[1] for f in files if f.startswith("tasks/")}):
        parsed = parse_id(task_id)
        if parsed:
            tasks.setdefault(pr_key(*parsed), _task(*parsed, harbor=task_id))
    return tasks


def _instruction_tasks(repo_id: str, pattern: str) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns=[pattern]))
    return {p.parent.name: p.read_text(errors="replace") for p in root.glob(pattern)}


def _terminal_bench_tasks() -> dict:
    found = _instruction_tasks("harborframework/terminal-bench-2.1", "tasks/*/instruction.md")
    return {name: _task(text=text) for name, text in sorted(found.items())}


def _polyglot_tasks() -> dict:
    found = _instruction_tasks("DCAgent2/aider_polyglot", "polyglot_*/instruction.md")
    tasks = {}
    for name, text in sorted(found.items()):
        exercise = name.split("_", 2)[2]
        tasks[name] = _task(text=polyglot_instruction(text), group=f"polyglot:{exercise}")
    return tasks


LOADERS = {
    "swe-rebench-leaderboard": (
        "nebius/SWE-rebench-leaderboard",
        lambda: _swe_tasks("nebius/SWE-rebench-leaderboard"),
    ),
    "swe-bench-verified": (
        "SWE-bench/SWE-bench_Verified",
        lambda: _swe_tasks("SWE-bench/SWE-bench_Verified"),
    ),
    "swe-bench-multilingual": (
        "SWE-bench/SWE-bench_Multilingual",
        lambda: _swe_tasks("SWE-bench/SWE-bench_Multilingual"),
    ),
    "aacr-bench": ("Alibaba-Aone/aacr-bench + osolmaz/aacr-bench-harbor", _aacr_tasks),
    "terminal-bench-2.1": ("harborframework/terminal-bench-2.1", _terminal_bench_tasks),
    "aider-polyglot": ("DCAgent2/aider_polyglot", _polyglot_tasks),
}


def _first_user(messages: list) -> str:
    for m in messages or []:
        if isinstance(m, dict) and str(m.get("role", "")).lower() in ("user", "human"):
            return str(m.get("content") or "")
    return ""


def load_benchmarks(cache: Path, refresh: bool = False) -> dict:
    data = json.loads(cache.read_text()) if cache.exists() and not refresh else {}
    if data.get("version") != CACHE_VERSION:
        data = {"version": CACHE_VERSION, "benchmarks": {}}
    for name, (source, loader) in LOADERS.items():
        if name in data["benchmarks"]:
            continue
        tasks = loader()
        data["benchmarks"][name] = {"source": source, "tasks": tasks}
        print(f"loaded {name}: {len(tasks)} tasks from {source}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data["benchmarks"]


# --------------------------------------------------------------------------- index + matching


class Index:
    def __init__(self, benchmarks: dict):
        self.by_id: dict[str, list] = defaultdict(list)
        self.by_pr: dict[str, list] = defaultdict(list)
        self.by_shingle: dict[str, list] = defaultdict(list)
        self.text_size: dict[tuple, int] = {}
        task_shingles: dict[tuple, set[str]] = {}
        group_df: dict[str, set] = defaultdict(set)
        for bench, spec in benchmarks.items():
            for task_id, task in spec["tasks"].items():
                ref = (bench, task_id)
                if "#" not in task_id:
                    self.by_id[norm_id(task_id)].append(ref)
                if task.get("pr"):
                    self.by_pr[task["pr"]].append(ref)
                if task.get("text"):
                    sh = shingles(task["text"])
                    task_shingles[ref] = sh
                    for s in sh:
                        group_df[s].add(task.get("group") or ref)
        for ref, sh in task_shingles.items():
            kept = {s for s in sh if len(group_df[s]) <= TEXT_MAX_GROUP_DF}
            if kept:
                self.text_size[ref] = len(kept)
                for s in kept:
                    self.by_shingle[s].append(ref)

    def match(self, instance_id: str, repo: str | None, text: str) -> list[dict]:
        hits: list[dict] = []
        for ref in self.by_id.get(norm_id(instance_id), ()):
            hits.append({"benchmark": ref[0], "key": "id", "detail": ref[1]})
        for key in sorted(row_pr_keys(instance_id, repo)):
            for ref in self.by_pr.get(key, ()):
                hits.append({"benchmark": ref[0], "key": "repo_pr", "detail": f"{ref[1]} ({key})"})
        hits.extend(self.match_text(text))
        if repo and (
            repo.lower().startswith("exercism/") or repo.lower() == "aider-ai/polyglot-benchmark"
        ):
            hits.append({"benchmark": "aider-polyglot", "key": "exercism_repo", "detail": repo})
        return hits

    def match_text(self, text: str) -> list[dict]:
        if not text or not self.by_shingle:
            return []
        counts: Counter = Counter()
        for s in shingles(text):
            for ref in self.by_shingle.get(s, ()):
                counts[ref] += 1
        hits = []
        for ref, n in counts.items():
            size = self.text_size[ref]
            need = max(TEXT_FLOOR_SHINGLES, min(TEXT_MIN_SHINGLES, 0.8 * size))
            if n >= need and n / size >= TEXT_MIN_CONTAINMENT:
                hits.append(
                    {"benchmark": ref[0], "key": "text", "detail": f"{ref[1]} ({n}/{size})"}
                )
        return hits


# --------------------------------------------------------------------------- dataset scan

_INDEX: Index | None = None


def _scan_file(path: str) -> tuple[int, dict, dict, dict]:
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    turns = next(c for c in ("messages", "trajectory", "conversation") if c in names)
    cols = ["instance_id"] + [c for c in ("repo", "hf_dataset_name", "dataset") if c in names]
    cols += [f"{turns}.list.element.{f}" for f in ("role", "content")]
    rows, hits, upstream, examples = 0, {}, {}, {}
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        for row in batch.to_pylist():
            rows += 1
            iid = str(row.get("instance_id") or "")
            text = _first_user(row.get(turns))
            found = _INDEX.match(iid, row.get("repo"), text)
            up = row.get("hf_dataset_name") or row.get("dataset") or ""
            upstream.setdefault(iid, up)
            if found:
                seen = hits.setdefault(iid, [])
                for h in found:
                    if h not in seen:
                        seen.append(h)
                if any(h["key"] == "text" for h in found) and iid not in examples:
                    examples[iid] = text[:1500]
    return rows, hits, upstream, examples


def scan(globs: list[str], index: Index, workers: int) -> dict:
    global _INDEX
    _INDEX = index
    files = sorted({f for g in globs for f in glob.glob(g, recursive=True)})
    if not files:
        raise SystemExit(f"no parquet files match {globs}")
    rows, hits, upstream, examples = 0, {}, {}, {}
    if workers <= 1:
        results = map(_scan_file, files)
    else:
        pool = multiprocessing.get_context("fork").Pool(workers)
        results = pool.imap_unordered(_scan_file, files)
    for r, h, u, e in results:
        rows += r
        for iid, found in h.items():
            seen = hits.setdefault(iid, [])
            seen += [x for x in found if x not in seen]
        for iid, up in u.items():
            upstream.setdefault(iid, up)
        for iid, ex in e.items():
            examples.setdefault(iid, ex)
    if workers > 1:
        pool.close()
    return {
        "files": len(files),
        "rows": rows,
        "hits": hits,
        "upstream": upstream,
        "examples": examples,
    }


def summarize(label: str, result: dict) -> None:
    hits, upstream = result["hits"], result["upstream"]
    print(f"\n== {label}: {result['rows']} rows, {len(upstream)} ids, {result['files']} files")
    by_key: Counter = Counter()
    by_bench: Counter = Counter()
    by_up: Counter = Counter()
    for iid, hs in hits.items():
        for b in {h["benchmark"] for h in hs}:
            by_bench[b] += 1
            by_up[(b, upstream.get(iid, ""))] += 1
        for bk in {(h["benchmark"], h["key"]) for h in hs}:
            by_key[bk] += 1
    for bench in sorted(by_bench):
        keys = ", ".join(f"{k}={n}" for (b, k), n in sorted(by_key.items()) if b == bench)
        ups = ", ".join(f"{u or '?'}={n}" for (b, u), n in sorted(by_up.items()) if b == bench)
        print(f"  {bench:<26}{by_bench[bench]:>6} ids  [{keys}]  upstream: {ups}")
    print(f"  -> {len(hits)} ids to exclude")
    text_only = [i for i, hs in hits.items() if {h["key"] for h in hs} == {"text"}]
    for iid in sorted(text_only)[:5]:
        detail = "; ".join(h["detail"] for h in hits[iid])
        snippet = " ".join(result["examples"].get(iid, "").split())[:200]
        print(f"     text-only {iid}: {detail}\n        {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datasets", nargs="+", help="label=parquet-glob (repeatable)")
    parser.add_argument("--out", required=True, help="JSON report {label: {instance_id: [hit]}}")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="benchmark fingerprint JSON")
    parser.add_argument("--refresh", action="store_true", help="re-download benchmark lists")
    parser.add_argument("--ids-out", help="write every matched id in the report, one per line")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--update", action="store_true", help="keep other labels already in --out")
    args = parser.parse_args()

    benchmarks = load_benchmarks(Path(args.cache), refresh=args.refresh)
    for bench, spec in benchmarks.items():
        print(f"{bench:<26}{len(spec['tasks']):>7} tasks  ({spec['source']})")

    index = Index(benchmarks)
    out = Path(args.out)
    report = json.loads(out.read_text()) if args.update and out.exists() else {}
    for spec in args.datasets:
        label, sep, pattern = spec.partition("=")
        if not sep:
            raise SystemExit(f"dataset must be label=glob, got {spec!r}")
        result = scan(pattern.split(","), index, args.workers)
        summarize(label, result)
        report[label] = {
            iid: [{**h, "upstream": result["upstream"].get(iid, "")} for h in hs]
            for iid, hs in sorted(result["hits"].items())
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {out}")
    if args.ids_out:
        ids = sorted({iid for hits in report.values() for iid in hits})
        Path(args.ids_out).write_text("".join(f"{iid}\n" for iid in ids))
        print(f"wrote {len(ids)} ids to {args.ids_out}")


if __name__ == "__main__":
    main()
