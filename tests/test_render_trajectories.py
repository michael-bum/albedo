import importlib.util
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service.shared.observation_format import (
    RETURNCODE,
    classify,
    is_scaffold_truncated,
    valid_output,
)


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load_script("render_trajectories")
P = _load_script("prepare_datasets")

TASK = [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix it"}]


def _call(command: str) -> dict:
    return {"function": {"name": "bash", "arguments": json.dumps({"command": command})}}


def _mswea(output: str, returncode: int = 0) -> dict:
    return {"role": "tool", "content": json.dumps({"returncode": returncode, "output": output})}


def test_a_single_command_turn_renders_as_before():
    turns = TASK + [
        {"role": "assistant", "content": "look", "tool_calls": [_call("ls")]},
        {"role": "tool", "content": "a.py\nb.py"},
    ]
    messages, _ = R.render_turns(turns)
    assert messages[2:] == [
        {"role": "assistant", "content": "THOUGHT: look\n\n```bash\nls\n```"},
        {"role": "user", "content": "a.py\nb.py"},
    ]


def test_a_turn_of_several_commands_becomes_one_turn_per_command():
    stats = Counter()
    turns = TASK + [
        {"role": "assistant", "content": "both", "tool_calls": [_call("ls"), _call("pwd")]},
        {"role": "tool", "content": "a.py"},
        {"role": "tool", "content": "/testbed"},
        {"role": "assistant", "content": "edit", "tool_calls": [_call("sed -i 's/a/b/' a.py")]},
        {"role": "tool", "content": "done"},
    ]
    messages, first_edit = R.render_turns(turns, stats=stats)
    assert messages[2:] == [
        {"role": "assistant", "content": "THOUGHT: both\n\n```bash\nls\n```"},
        {"role": "user", "content": "a.py"},
        {"role": "assistant", "content": "```bash\npwd\n```"},
        {"role": "user", "content": "/testbed"},
        {"role": "assistant", "content": "THOUGHT: edit\n\n```bash\nsed -i 's/a/b/' a.py\n```"},
        {"role": "user", "content": "done"},
    ]
    assert first_edit == 3
    assert stats["multi_call_split"] == 1


def test_mini_swe_agent_json_is_rendered_in_the_returncode_format():
    stats = Counter()
    turns = TASK + [
        {"role": "assistant", "content": "look", "tool_calls": [_call("ls")]},
        _mswea("a.py\nb.py\n"),
        {"role": "assistant", "content": "", "tool_calls": [_call("true")]},
        _mswea(""),
    ]
    messages, _ = R.render_turns(turns, stats=stats)
    observations = [m["content"] for m in messages[2:] if m["role"] == "user"]
    assert observations == [
        "<returncode>0</returncode>\n<output>\na.py\nb.py\n</output>",
        "<returncode>0</returncode>\n<output>\n</output>",
    ]
    assert all(classify(o) == RETURNCODE and valid_output(o, RETURNCODE) for o in observations)
    assert stats["mswea_rows"] == 1


def test_mini_swe_agent_truncated_and_failed_commands_keep_their_fields():
    truncated = R._render_mswea(
        {
            "returncode": 0,
            "warning": "Output too long.",
            "output_head": "head",
            "output_tail": "tail",
            "elided_chars": 42,
        }
    )
    assert truncated == (
        "<returncode>0</returncode>\n<warning>\nOutput too long.\n</warning>"
        "<output_head>\nhead\n</output_head>\n<elided_chars>\n42 characters elided\n"
        "</elided_chars>\n<output_tail>\ntail\n</output_tail>"
    )
    assert is_scaffold_truncated(truncated)
    failed = R._render_mswea(
        {"returncode": -1, "output": "", "exception_info": "Command 'x' timed out after 60 seconds"}
    )
    assert failed == (
        "<returncode>-1</returncode>\n"
        "<exception_info>Command 'x' timed out after 60 seconds</exception_info>\n"
        "<output>\n</output>"
    )


def test_json_output_of_another_harness_is_left_alone():
    """Only a row whose results are mini-swe-agent's JSON is converted: a command that happens to
    print such an object elsewhere keeps its bytes."""
    printed = json.dumps({"returncode": 0, "output": "x"})
    turns = TASK + [
        {"role": "assistant", "content": "", "tool_calls": [_call("ls")]},
        {"role": "tool", "content": "a.py"},
        {"role": "assistant", "content": "", "tool_calls": [_call("cat r.json")]},
        {"role": "tool", "content": printed},
    ]
    messages, _ = R.render_turns(turns)
    assert messages[-1]["content"] == printed


def test_each_open_swe_version_reads_only_its_own_raw_files(tmp_path):
    for name in (
        "Open-SWE-Traces/data/minimax_m25_openhands_trajectories/train-00000-of-00001.parquet",
        "Open-SWE-Traces-v1.1/data/openhands/qwen36_27b/scale-swe/train-00000-of-00001.parquet",
        "Open-SWE-Traces-v1.1/data/minisweagent/qwen38_27b/scale-swe/train-00000-of-00001.parquet",
        "Open-SWE-Traces-v1.2/data/minisweagent/qwen38_27b/scale-swe/train-00000-of-00001.parquet",
        "Open-SWE-Traces-v1.2/data/openhands/qwen36_27b/scale-swe/train-00000-of-00001.parquet",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def shards(source: str) -> list[str]:
        files = R._raw_shards(tmp_path, P.SOURCES[source])
        return [str(f.relative_to(tmp_path)).split("/data/")[1].split("/train")[0] for f in files]

    assert shards("open-swe-traces-v1.0") == ["minimax_m25_openhands_trajectories"]
    assert shards("open-swe-traces-v1.1") == ["openhands/qwen36_27b/scale-swe"]
    assert shards("open-swe-traces-v1.2") == ["minisweagent/qwen38_27b/scale-swe"]


def test_every_distinct_rollout_is_kept_and_identical_ones_once(tmp_path, monkeypatch):
    def rollout(iid: str, command: str) -> dict:
        turns = TASK + [
            {"role": "assistant", "content": "", "tool_calls": [_call("ls")]},
            {"role": "tool", "content": "a.py"},
            {"role": "assistant", "content": "", "tool_calls": [_call(command)]},
            {"role": "tool", "content": "ok"},
        ]
        return {"instance_id": iid, "repo": iid.split("__")[0], "trajectory": json.dumps(turns)}

    rows = [
        rollout("a__x-1", "pwd"),
        rollout("a__x-1", "pwd"),  # the same run twice
        rollout("a__x-1", "id"),  # another run of the same task
        rollout("a__x-2", "pwd"),  # a second task of the capped repository
        rollout("b__y-3", "pwd"),
    ]
    raw = tmp_path / "raw" / "Fake" / "data"
    raw.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {k: [r[k] for r in rows] for k in ("instance_id", "repo")}
            | {"trajectory": [json.loads(r["trajectory"]) for r in rows]}
        ),
        raw / "train-0.parquet",
    )
    spec = {
        "repos": {"o/Fake": "0" * 40},
        "shard_glob": "data/train-*.parquet",
        "render": True,
        "family": "pr",
        "repo_cap": 1,
    }
    monkeypatch.setitem(R.SOURCES, "fake", spec)
    stats = R.render_source("fake", tmp_path / "raw", tmp_path / "out")
    out = pq.read_table(tmp_path / "out" / "fake" / "data" / "train-00000.parquet").to_pylist()
    assert [r["instance_id"] for r in out] == ["a__x-1", "a__x-1", "b__y-3"]
    assert (stats["duplicate_trajectory"], stats["repo_cap"]) == (1, 1)
