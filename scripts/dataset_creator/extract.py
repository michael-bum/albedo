from __future__ import annotations

import json
import re
from pathlib import Path

BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
TRAJ_MARKER_RE = re.compile(r"^(REFERENCE STEP \d+:|ENVIRONMENT OBSERVATION:)\s*$", re.M)


def parse_system_user(prompt: str) -> tuple[str, str]:
    blocks = BLOCK_RE.findall(prompt)
    if len(blocks) < 2 or blocks[0][0] != "system" or blocks[1][0] != "user":
        raise ValueError(f"unexpected prompt structure: roles={[r for r, _ in blocks[:3]]}")
    return blocks[0][1].strip(), blocks[1][1].strip()


def parse_trajectory(traj: str) -> list[dict]:
    parts = TRAJ_MARKER_RE.split(traj)
    turns: list[dict] = []
    role = "assistant"
    for i, part in enumerate(parts):
        if i % 2 == 1:
            role = "assistant" if part.startswith("REFERENCE STEP") else "user"
            continue
        content = part.strip()
        if not content:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})
    return turns


def sanitize(model: str) -> str:
    return re.sub(r"\W+", "_", model.split("/", 1)[-1]).strip("_").lower()


def _turns(steps: list[dict]) -> list[dict]:
    turns = []
    for step in steps:
        turns.append({"role": "assistant", "content": step["assistant"]})
        if step.get("observation"):
            turns.append({"role": "user", "content": step["observation"]})
    return turns


def _references(qs: dict) -> list[tuple[str, list[dict]]]:
    """(model, turns) pairs. `reference_steps` carries each run as steps; artifacts written before
    it carry the runs as rendered text, which has to be re-split to get the same thing back."""
    runs = qs.get("reference_steps")
    if runs:
        return [(run.get("model", ""), _turns(run["steps"])) for run in runs]
    models = qs.get("reference_models") or [qs.get("reference_model")]
    trajs = qs.get("reference_trajectories") or [qs.get("reference_trajectory")]
    return [(m, parse_trajectory(t)) for m, t in zip(models, trajs) if m and t]


def extract_rows(run_dir: Path) -> dict[str, list[dict]]:
    run_id = run_dir.name
    prompts = {}
    with open(run_dir / "generated-samples.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            prompts[rec["sample_id"].split("#")[0]] = rec["prompt"]
    by_model: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    with open(run_dir / "scoring-results.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            sample_id = rec["sample_id"].split("#")[0]
            if sample_id not in prompts:
                continue
            for ref_model, turns in _references(rec.get("question_source") or {}):
                key = (sample_id, tuple(t["content"] for t in turns if t["role"] == "assistant"))
                if key in seen:
                    continue
                seen.add(key)
                system, user = parse_system_user(prompts[sample_id])
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    *turns,
                ]
                by_model.setdefault(sanitize(ref_model), []).append(
                    {"sample_id": sample_id, "messages": messages, "_run_id": run_id}
                )
    return by_model
