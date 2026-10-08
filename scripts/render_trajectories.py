#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_datasets import LEAKS, SOURCES, raw_dir, raw_globs

from albedo_eval_service.simulator.prompt_simulator import COMPLETE_MARKER

_EDIT_SUBCOMMANDS = {"create", "str_replace", "insert", "undo_edit", "write"}
_BASH_TOOLS = {"bash", "execute_bash", "run_bash_cmd", "shell"}
_EDITOR_TOOLS = {"str_replace_editor", "str_replace_based_edit_tool", "edit_file", "file_editor"}
_THINK_TOOLS = {"think"}
_DONE_TOOLS = {"finish", "submit", "complete"}

_BASH_EDIT_RE = re.compile(
    r"sed\s+-i|tee\s+[\w./-]|cat\s*>|git apply|patch\s+-p|applypatch|"
    r"cp\s+[\w./-]|mv\s+[\w./-]|(?<![-\d&])>>?\s*(?!/dev/)[\w.][\w./-]*"
)

_THOUGHT_LOGGED = "your thought has been logged"


def _arguments(tool_call: Any) -> tuple[str, dict]:
    if not isinstance(tool_call, dict):
        return "", {}
    function = tool_call.get("function") or {}
    name = str(function.get("name") or tool_call.get("name") or "")
    args = function.get("arguments")
    if args is None:
        args = tool_call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    return name, args if isinstance(args, dict) else {}


def _bash_block(command: str) -> str:
    return f"```bash\n{command.strip()}\n```"


def _render_editor(args: dict) -> tuple[str, bool]:
    command = str(args.get("command") or "").strip()
    path = str(args.get("path") or args.get("file_path") or "").strip()
    if command == "view":
        rng = args.get("view_range")
        if isinstance(rng, list) and len(rng) == 2:
            return _bash_block(f"sed -n '{rng[0]},{rng[1]}p' {path} | cat -n"), False
        return _bash_block(f"cat -n {path}"), False
    if command in {"create", "write"}:
        body = str(args.get("file_text") or args.get("content") or "")
        return _bash_block(f"cat > {path} <<'EOF'\n{body}\nEOF"), True
    if command == "insert":
        body = str(args.get("new_str") or args.get("insert_line_text") or "")
        line = args.get("insert_line")
        return _bash_block(f"sed -i '{line}a\\\n{body}' {path}"), True
    if command == "str_replace":
        old = str(args.get("old_str") or "")
        new = str(args.get("new_str") or "")
        return (
            f"Editing `{path}`:\n\n```\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n```",  # noqa: E501
            True,
        )
    if command == "undo_edit":
        return _bash_block(f"git checkout -- {path}"), True
    return _bash_block(f"# {command} {path}".strip()), command in _EDIT_SUBCOMMANDS


def _render_call(tool_call: Any) -> tuple[str, bool, str]:
    name, args = _arguments(tool_call)
    lowered = name.lower()
    if lowered in _THINK_TOOLS:
        return str(args.get("thought") or args.get("text") or ""), False, "think"
    if lowered in _DONE_TOOLS:
        return _bash_block(COMPLETE_MARKER), False, "done"
    if lowered in _BASH_TOOLS:
        command = str(args.get("command") or args.get("cmd") or "")
        return _bash_block(command), bool(_BASH_EDIT_RE.search(command)), "action"
    if lowered in _EDITOR_TOOLS:
        text, is_edit = _render_editor(args)
        return text, is_edit, "action"
    return "", False, "unknown"


def _turn_calls(turn: dict) -> list:
    calls = turn.get("tool_calls")
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except (json.JSONDecodeError, ValueError):
            calls = None
    return [c for c in (calls or []) if isinstance(c, dict)]


def _thought(turn: dict) -> str:
    for key in ("content", "reasoning_content", "think"):
        value = turn.get(key)
        if value:
            return str(value).strip()
    return ""


_MSWEA_KEYS = {"returncode", "output", "output_head", "output_tail", "elided_chars", "warning"}


def _mswea_observation(content: str) -> dict | None:
    """mini-swe-agent's tool-calling harness reports a command as a JSON object."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or "returncode" not in data:
        return None
    if not set(data) - {"exception_info"} <= _MSWEA_KEYS:
        return None
    return data


def _render_mswea(data: dict) -> str:
    """The JSON as mini-swe-agent's text harness prints it (the `<returncode>` format mini-coder
    is recorded in), so the eval's returncode simulation serves these samples too."""
    head = f"<returncode>{data['returncode']}</returncode>\n"
    if data.get("exception_info"):
        head += f"<exception_info>{data['exception_info']}</exception_info>\n"
    if "output_head" in data:
        return (
            f"{head}<warning>\n{data.get('warning') or ''}\n</warning>"
            f"<output_head>\n{data['output_head']}\n</output_head>\n"
            f"<elided_chars>\n{data.get('elided_chars')} characters elided\n</elided_chars>\n"
            f"<output_tail>\n{data.get('output_tail') or ''}\n</output_tail>"
        )
    return f"{head}<output>\n{data.get('output') or ''}</output>"


def _is_mswea(turns: list) -> bool:
    for turn in turns:
        if isinstance(turn, dict) and str(turn.get("role") or "").lower() == "tool":
            content = str(turn.get("content") or "").strip()
            if content:
                return _mswea_observation(content) is not None
    return False


def _observation(turn: dict, mswea: bool) -> str:
    observation = str(turn.get("content") or "").strip()
    if mswea:
        data = _mswea_observation(observation)
        if data is not None:
            return _render_mswea(data)
    return observation


def render_turns(turns: list, *, stats: Counter | None = None) -> tuple[list[dict], int]:
    stats = stats if stats is not None else Counter()
    out: list[dict] = []
    pending_thought = ""
    assistant_index = 0
    first_edit = 0
    turns = [t for t in turns if isinstance(t, dict)]
    mswea = _is_mswea(turns)
    if mswea:
        stats["mswea_rows"] += 1

    def observe(turn: dict) -> None:
        observation = _observation(turn, mswea)
        if observation and _THOUGHT_LOGGED not in observation.lower():
            out.append({"role": "user", "content": observation})

    index = 0
    while index < len(turns):
        turn = turns[index]
        index += 1
        role = str(turn.get("role") or "").lower()

        if role in {"system", "user"}:
            content = _thought(turn)
            if content:
                out.append({"role": role, "content": content})
            continue

        if role == "tool":
            observe(turn)
            continue

        if role != "assistant":
            continue

        calls = _turn_calls(turn)
        thought = _thought(turn)
        if not calls:
            pending_thought = "\n\n".join(p for p in (pending_thought, thought) if p)
            continue

        # a turn that runs several commands at once becomes one turn per command, each followed
        # by its own result, so no command or result of the turn is dropped
        results: list[dict] = []
        while index < len(turns) and str(turns[index].get("role") or "").lower() == "tool":
            results.append(turns[index])
            index += 1
        if len(calls) > 1:
            stats["multi_call_split"] += 1

        for call, result in zip_longest(calls, results[: len(calls)]):
            rendered, is_edit, kind = _render_call(call)
            if kind == "think":
                pending_thought = "\n\n".join(p for p in (pending_thought, thought, rendered) if p)
                thought = ""
                stats["think_folded"] += 1
                continue
            if kind == "unknown" or not rendered:
                stats["unknown_tool"] += 1
                if result is not None:
                    observe(result)
                continue

            full_thought = "\n\n".join(p for p in (pending_thought, thought) if p)
            pending_thought = thought = ""
            body = f"THOUGHT: {full_thought}\n\n{rendered}" if full_thought else rendered
            out.append({"role": "assistant", "content": body})
            assistant_index += 1
            if is_edit and not first_edit:
                first_edit = assistant_index
            stats[f"kind_{kind}"] += 1
            if result is not None:
                observe(result)
        for result in results[len(calls) :]:
            observe(result)

    return out, first_edit


def smith_family(instance_id: str) -> str:
    if "." not in instance_id:
        return "pr"
    tail = instance_id.rsplit(".", 1)[-1]
    for prefix, family in (("pr_", "pr"), ("lm_", "lm"), ("combine", "combine")):
        if tail.startswith(prefix):
            return family
    return "mechanical"


def _repo_of(row: dict, instance_id: str) -> str:
    repo = row.get("repo")
    if repo:
        return str(repo).replace("/", "__")
    return instance_id.split(".")[0]


def _keep(row: dict, instance_id: str, spec: dict, repo_tasks: dict[str, set[str]]) -> str | None:
    if instance_id in LEAKS:
        return "excluded_id"
    upstream = str(row.get("hf_dataset_name") or row.get("dataset") or "")
    if any(bad in upstream for bad in spec.get("exclude_upstream", ())):
        return "excluded_upstream"
    # the cap counts a repository's tasks, so every rollout of a task already admitted is kept
    cap = spec.get("repo_cap")
    tasks = repo_tasks.get(_repo_of(row, instance_id), set())
    if cap and instance_id not in tasks and len(tasks) >= cap:
        return "repo_cap"
    return None


def _trajectory_key(instance_id: str, messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps([instance_id, messages]).encode("utf-8")).hexdigest()


def _raw_shards(raw_root: Path, spec: dict) -> list[Path]:
    files: list[Path] = []
    for repo in spec["repos"]:
        base = raw_root / raw_dir(spec, repo)
        files.extend(sorted({p for g in raw_globs(spec) for p in base.glob(g)}))
    return files


def render_source(
    name: str,
    raw_root: Path,
    out_root: Path,
    *,
    max_rows_per_shard: int = 2500,
    limit_shards: int | None = None,
) -> dict:
    spec = SOURCES[name]
    if not spec.get("render"):
        raise SystemExit(f"{name}: not a render source (already in eval shard format)")
    shards = _raw_shards(raw_root, spec)
    if limit_shards:
        shards = shards[:limit_shards]
    if not shards:
        raise SystemExit(f"{name}: no raw shards under {raw_root} for {spec['repos']}")

    out_dir = out_root / name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    # every rollout of a task is kept (the sampler draws one per task per eval); only a
    # rollout that renders identically to one already kept is dropped
    repo_tasks: dict[str, set[str]] = {}
    seen_trajectories: set[str] = set()
    buffer: list[dict] = []
    written = 0

    def flush() -> None:
        nonlocal buffer, written
        if not buffer:
            return
        table = pa.table(
            {
                "instance_id": [r["instance_id"] for r in buffer],
                "messages": [r["messages"] for r in buffer],
                "first_edit": [r["first_edit"] for r in buffer],
                "family": [r["family"] for r in buffer],
                "repo": [r["repo"] for r in buffer],
                "language": [r["language"] for r in buffer],
            }
        )
        pq.write_table(table, out_dir / f"train-{written:05d}.parquet", row_group_size=64)
        written += 1
        buffer = []

    for shard in shards:
        parquet = pq.ParquetFile(shard)
        columns = [
            c
            for c in ("instance_id", "repo", "hf_dataset_name", "dataset", "language", "verified")
            if c in parquet.schema_arrow.names
        ]
        turns_col = next(
            c for c in ("messages", "trajectory", "conversation") if c in parquet.schema_arrow.names
        )
        for batch in parquet.iter_batches(batch_size=64, columns=columns + [turns_col]):
            for row in batch.to_pylist():
                stats["rows_in"] += 1
                instance_id = str(row.get("instance_id") or "")
                if not instance_id:
                    stats["no_instance_id"] += 1
                    continue
                reason = _keep(row, instance_id, spec, repo_tasks)
                if reason:
                    stats[reason] += 1
                    continue
                messages, first_edit = render_turns(row.get(turns_col) or [], stats=stats)
                assistant = sum(1 for m in messages if m["role"] == "assistant")
                if assistant < 2:
                    stats["too_few_assistant_turns"] += 1
                    continue
                if any(not m["content"] for m in messages):
                    stats["empty_content_dropped"] += 1
                    continue
                key = _trajectory_key(instance_id, messages)
                if key in seen_trajectories:
                    stats["duplicate_trajectory"] += 1
                    continue
                seen_trajectories.add(key)
                repo = _repo_of(row, instance_id)
                repo_tasks.setdefault(repo, set()).add(instance_id)
                family = spec.get("family") or smith_family(instance_id)
                buffer.append(
                    {
                        "instance_id": instance_id,
                        "messages": messages,
                        "first_edit": first_edit,
                        "family": family,
                        "repo": repo,
                        "language": str(row.get("language") or spec.get("language") or "unknown"),
                    }
                )
                stats["rows_out"] += 1
                if len(buffer) >= max_rows_per_shard:
                    flush()
    flush()
    stats["shards_written"] = written
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        required=True,
        help=f"one of: {','.join(n for n, s in SOURCES.items() if s.get('render'))}",
    )
    parser.add_argument(
        "--raw-root", required=True, help="Dir holding <repo-name>/data/*.parquet snapshots."
    )
    parser.add_argument(
        "--out-root", required=True, help="Dir to write <source>/data/train-*.parquet into."
    )
    parser.add_argument("--max-rows-per-shard", type=int, default=2500)
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Only read the first N raw shards (smoke runs).",
    )
    args = parser.parse_args()

    stats = render_source(
        args.source,
        Path(args.raw_root),
        Path(args.out_root),
        max_rows_per_shard=args.max_rows_per_shard,
        limit_shards=args.limit_shards,
    )
    print(
        f"{args.source}: {stats['rows_out']}/{stats['rows_in']} rows -> {stats['shards_written']} shards"  # noqa: E501
    )
    for key, value in sorted(stats.items()):
        print(f"  {key:<28}{value}")


if __name__ == "__main__":
    main()
