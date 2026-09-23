---
pretty_name: albedo
configs:
- config_name: default
  data_files:
  - split: glm_5_2
    path: data/glm_5_2/*.parquet
  - split: gpt_5_6_luna
    path: data/gpt_5_6_luna/*.parquet
  - split: mimo_v2_5_pro
    path: data/mimo_v2_5_pro/*.parquet
---

# albedo — SWE-agent reference trajectories

Reference trajectories from SN97 (albedo) evaluations. Each row is a
chat-format sample:

- `sample_id` — source shard and row of the underlying task
  (`<dataset>/<shard>.parquet:<row>:<turn>`)
- `messages` — `list<{role, content}>`: a `system` prompt, the `user` task
  (PR description + instructions), then alternating `assistant` reference
  steps and `user` environment observations
- `run` — which of the sample's reference runs this row is (1-3)
- `made_edit` — whether the run changed any file (null for rows recorded
  before 2026-09-22)
- `score` — the run's score on the sample's final checklist, 0-1, on the same
  scale miners are scored on (null before 2026-09-22)
- `questions` — JSON list of `{id, text, milestone, rung, weight, earned}`:
  the checklist the sample was scored on and which items this run earned
  (`earned` is null before 2026-09-22; `milestone`, `rung` and `weight`
  are null for the oldest rows)
- `milestones` — JSON list of `{id, statement, category}` the checklist was
  built from (null before 2026-09-22)
- `king_score` — the reigning model's score on the same sample in the same
  evaluation, a difficulty anchor
- `eval_run_id` — the evaluation the row comes from

One split per reference model. Every parquet contains exactly 100 samples;
new files are appended as evaluations complete.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("dendriteholdings/albedo", split="glm_5_2")
print(ds[0]["messages"][0])
```
