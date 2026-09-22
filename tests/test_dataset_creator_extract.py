"""The dataset creator turns eval artifacts into reference-trajectory rows for the HF upload.

It reads `question_source` from scoring-results.jsonl; the 2026-09-07 format change broke the
upload for days, so both shapes it must understand are pinned here: `reference_steps` (runs as
{assistant, observation} steps) and the older rendered `reference_trajectories` text.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "scripts" / "dataset_creator"
sys.path.insert(0, str(_DIR))  # the creator's modules import each other by bare name


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"dataset_creator_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract = _load("extract")
store = _load("store")
backfill = _load("backfill_columns")

PROMPT = "<|im_start|>system\nSYS\n<|im_end|>\n<|im_start|>user\nTASK\n<|im_end|>\n"
STEPS = [
    {
        "assistant": "```bash\nls\n```",
        "observation": "<returncode>0</returncode>\n<output>\na\n</output>",
    },
    {"assistant": "```bash\ncat a\n```", "observation": ""},
]


def _run_dir(tmp_path: Path, question_source: dict) -> Path:
    (tmp_path / "generated-samples.jsonl").write_text(
        json.dumps({"sample_id": "src/data/x.parquet:1:1#r1", "prompt": PROMPT}) + "\n"
    )
    records = [
        {
            "sample_id": f"src/data/x.parquet:1:1{suffix}",
            "question_source": question_source,
            "king_score": 0.75,
            "questions": [
                {"id": "q_01", "text": "Did it look?", "milestone": "m1", "rung": 1, "weight": 1.0}
            ],
        }
        for suffix in ("", "#r2")  # every rollout record repeats the sample's references
    ]
    (tmp_path / "scoring-results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return tmp_path


def test_reference_steps_become_alternating_messages_once_per_distinct_run(tmp_path):
    other = [{"assistant": "```bash\npwd\n```", "observation": "/repo"}]
    source = {
        "reference_steps": [
            {"run": 1, "model": "z-ai/glm-5.2", "made_edit": True, "steps": STEPS},
            {"run": 2, "model": "z-ai/glm-5.2", "made_edit": False, "steps": STEPS},
            {"run": 3, "model": "z-ai/glm-5.2", "made_edit": False, "steps": other},
        ]
    }
    rows = extract.extract_rows(_run_dir(tmp_path, source))
    assert set(rows) == {"glm_5_2"}
    assert len(rows["glm_5_2"]) == 2, "identical runs and repeated records collapse to one row each"
    messages = rows["glm_5_2"][0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant"]
    assert messages[:2] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "TASK"},
    ]
    assert messages[-1]["content"] == "```bash\ncat a\n```", (
        "a step without observation ends the row"
    )


def test_rendered_reference_trajectories_still_extract(tmp_path):
    rendered = (
        "REFERENCE STEP 1:\n```bash\nls\n```\nENVIRONMENT OBSERVATION:\n<output>\na\n</output>\n"
        "REFERENCE STEP 2:\n```bash\ncat a\n```\n"
    )
    source = {"reference_models": ["z-ai/glm-5.2"], "reference_trajectories": [rendered]}
    rows = extract.extract_rows(_run_dir(tmp_path, source))
    messages = rows["glm_5_2"][0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant"]
    assert messages[2]["content"] == "```bash\nls\n```"


def test_new_artifacts_fill_every_score_column(tmp_path):
    source = {
        "reference_steps": [
            {"run": 1, "model": "z-ai/glm-5.2", "made_edit": True, "steps": STEPS},
            {"run": 2, "model": "z-ai/glm-5.2", "made_edit": False, "steps": STEPS[:1]},
        ],
        "reference_scoring": {"q_01": [1]},
        "reference_self_scores": [{"run": 1, "model": "z-ai/glm-5.2", "yes_rate": 1.0}],
        "milestones": [
            {"id": "m1", "statement": "Looked at the file", "category": "explore", "evidence": [1]}
        ],
    }
    rows = {r["run"]: r for r in extract.extract_rows(_run_dir(tmp_path, source))["glm_5_2"]}
    assert rows[1]["made_edit"] is True and rows[1]["score"] == 1.0
    assert rows[2]["made_edit"] is False and rows[2]["score"] is None, (
        "run 2 gave no readable verdict"
    )
    assert json.loads(rows[1]["questions"]) == [
        {
            "id": "q_01",
            "text": "Did it look?",
            "milestone": "m1",
            "rung": 1,
            "weight": 1.0,
            "earned": True,
        }
    ]
    assert json.loads(rows[2]["questions"])[0]["earned"] is False
    assert json.loads(rows[1]["milestones"]) == [
        {"id": "m1", "statement": "Looked at the file", "category": "explore"}
    ]
    assert rows[1]["king_score"] == 0.75 and rows[1]["eval_run_id"] == tmp_path.name


def test_old_artifacts_leave_unrecorded_columns_null(tmp_path):
    rendered = (
        "REFERENCE STEP 1:\n```bash\nls\n```\nENVIRONMENT OBSERVATION:\n<output>\na\n</output>\n"
    )
    source = {"reference_models": ["z-ai/glm-5.2"], "reference_trajectories": [rendered]}
    row = extract.extract_rows(_run_dir(tmp_path, source))["glm_5_2"][0]
    assert (
        row["run"] == 1
        and row["made_edit"] is None
        and row["score"] is None
        and row["milestones"] is None
    )
    assert json.loads(row["questions"])[0]["earned"] is None, (
        "no join table yet, so unknown, not False"
    )
    assert row["king_score"] == 0.75


def test_chunks_share_one_schema_even_when_rows_lack_columns(tmp_path):
    """The HF loader casts a split to its first file: a chunk of old rows must type its empty
    columns exactly like a chunk of new ones."""
    old_row = {"sample_id": "s:1:1", "messages": [{"role": "user", "content": "t"}]}
    new_row = {
        **old_row,
        "run": 2,
        "made_edit": True,
        "score": 0.5,
        "questions": "[]",
        "milestones": "[]",
        "king_score": 0.7,
        "eval_run_id": "r",
    }
    store.write_chunk_file([old_row], tmp_path / "old.parquet")
    store.write_chunk_file([new_row], tmp_path / "new.parquet")
    import pyarrow.parquet as pq

    assert pq.read_schema(tmp_path / "old.parquet").equals(pq.read_schema(tmp_path / "new.parquet"))
    assert pq.read_schema(tmp_path / "old.parquet").equals(store.SCHEMA)
    assert pq.read_table(tmp_path / "old.parquet").to_pylist()[0]["score"] is None


def test_backfill_enriches_matching_rows_and_nulls_the_rest():
    messages = [
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "```bash\nls\n```"},
    ]
    index = {backfill.row_key("s:1:1", messages): {"run": 1, "score": 0.25, "eval_run_id": "r"}}
    hit, matched = backfill.enrich({"sample_id": "s:1:1", "messages": messages}, index)
    miss, unmatched = backfill.enrich({"sample_id": "s:9:9", "messages": messages}, index)
    assert matched and hit["score"] == 0.25 and hit["run"] == 1 and hit["made_edit"] is None
    assert not unmatched and miss["score"] is None and set(miss) >= set(store.COLUMNS)
