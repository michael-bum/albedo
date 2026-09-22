"""The dataset creator turns eval artifacts into reference-trajectory rows for the HF upload.

It reads `question_source` from scoring-results.jsonl; the 2026-09-07 format change broke the
upload for days, so both shapes it must understand are pinned here: `reference_steps` (runs as
{assistant, observation} steps) and the older rendered `reference_trajectories` text.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "dataset_creator_extract",
    Path(__file__).resolve().parents[1] / "scripts" / "dataset_creator" / "extract.py",
)
extract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extract)

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
        {"sample_id": f"src/data/x.parquet:1:1{suffix}", "question_source": question_source}
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
