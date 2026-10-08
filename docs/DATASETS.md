# Datasets & observation formats (SN97)

What models are evaluated on, how samples are drawn, and the format rule the environment simulator
must obey. Sources of truth: `scripts/prepare_datasets.py`, `scripts/render_trajectories.py`,
`scripts/build_manifest.py`, `src/albedo_eval_service/shared/sampling.py`,
`src/albedo_eval_service/shared/observation_format.py`, and `src/repo_context_service/` for grounding.

For how the resulting trajectories are turned into a score, see [SCORING.md](SCORING.md).

---

## Corpora

Thirteen sources, declared in `prepare_datasets.SOURCES` (the six below plus the seven Affine ones). All are real agent trajectories on real
repositories — never synthetic prompts. Each source pins every upstream repo it reads to its own
revision (`repos: {repo: revision}`), so two sources sharing a repo never move each other's
snapshot; a render source sharing a repo also names its own `raw_dir`, so neither globs the
other's files.

| source | language | upstream | notes |
|---|---|---|---|
| `mini-coder` | python | `ricdomolm/mini-coder-trajs-400k` | mini-swe-agent format, used as-is |
| `mini-coder-rs` | rust | `AlienKevin/SWE-smith-rs-*` (3 repos) | rendered; **named `mini-coder-*` deliberately** |
| `open-swe-traces-v1.0` | mixed | `nvidia/Open-SWE-Traces` @ `9c0e4579` | rendered; the four v1.0 arms (minimax-m2.5, qwen3.5 × openhands, swe-agent); SWE-rebench-V2 tasks only; pinned before upstream's 08/26 removal of git-hacking trajectories |
| `open-swe-traces-v1.1` | mixed | `nvidia/Open-SWE-Traces` @ `f8fb5b3d` | rendered; qwen3.6-27b (openhands, swe-agent, mini-swe-agent) and deepseek-v4-flash (openhands); SWE-rebench-V2 and Scale-SWE tasks |
| `open-swe-traces-v1.2` | mixed | `nvidia/Open-SWE-Traces` @ `f8fb5b3d` | rendered; qwen3.8-27b mini-swe-agent; SWE-rebench-V2 and Scale-SWE tasks |
| `swe-hero` | python | `nvidia/SWE-Hero-openhands-trajectories` | rendered; `repo_cap: 200` (pandas alone was 37.9% of the pool) |

**Affine** (`dendriteholdings/<source>` on Hugging Face). Affine's corpus (`data.affine.io`, view
`duel_turns@v4`) and SWE-Lego's transcripts, pre-converted in `affine-dedup` to mini-coder's returncode
turns (no render step). One source per kind of machine, since the scaffold is the only thing that tells
machines apart: `affine-openhands` (SWE-Lego), `affine-mswea` (mini-swe-agent text harness, dash),
`affine-bash` (`bash -c`), `affine-tools` (claude_code, pi, kimi_code, hermes_agent; `/bin/bash -c`),
and `mini-coder-affine-<machine>` for the swesmith tasks (their ids parse only under a `mini-coder*`
name). Their machines differ from albedo's other sources for the same ids in two places, both in
`repo_context_service.core`: an R2E fix-commit id resolves to its parent (as for `swe-hero`), and a
swesmith id resolves to the branch head even for Rust (Affine's Rust machines did not sit at `Bug
Patch`), so these sources cache resolved SHAs apart (`_OWN_SHA_CACHE`). Runs that got the fix from
outside their own work (swesmith `Bug Patch` history, the task's own PR in `git log --all`, fetching
upstream code) were dropped before conversion.

Rendering keeps **every distinct rollout** of a task (upstream ships several per task: separate
runs, models and harnesses); it drops only a rollout that renders identically to one already
kept (`duplicate_trajectory`). Picking one rollout per task is the sampler's job:
`_instance_pool` groups all rows of all sources by `instance_id` and draws one with the eval's
seeded RNG, so a task is used at most once per eval, every task is equally likely whatever its
rollout count, and different evals see different rollouts of it. `repo_cap` counts a repository's
tasks, not its rows. The three Open-SWE sources share most tasks but no rollout (no
`trajectory_id` repeats across versions).

Benchmark leaks are excluded from every source (`prepare_datasets.LEAKS`, applied when rendering
and again as `blocked` in the manifest), since the sampler pools all sources' rows by
`instance_id`: a task leaked in one source would leak through another's copy of it. Plus
`exclude_upstream` for `swe-hero` (rebench is our benchmark). The ids live in
`scripts/benchmark_leaks.txt`, written by `scripts/benchmark_leaks.py --ids-out`, which matches
every row against SWE-rebench (leaderboard), SWE-bench Verified, SWE-bench Multilingual,
AACR-bench, Terminal-Bench 2.1 and Aider Polyglot on instance id, repository + PR number and
problem-statement text. Rerun it over all sources whenever a source or a benchmark changes:
Scale-SWE rewrites issue text and ships the same PR under another id (`owner_repo_pr<N>`), so only
the repo + PR key catches its copies.

### Rendering normalizes actions, not observations

`render_trajectories.py` converts every upstream tool call into a **bash block** — `_render_call`
maps bash tools to ` ```bash `, editor `view` to `cat -n`, `create` to `cat > … <<'EOF'`,
`str_replace` to a SEARCH/REPLACE block — and folds `think` calls into the next action's `THOUGHT:`.
A turn that runs several tool calls at once becomes one turn per call, each followed by its own
result (in order; upstream results do not name their calls), so no command and no result is
dropped.

Tool **observations** are copied through verbatim. That asymmetry is the single most important
fact about this data: **every trajectory keeps its upstream observation format all the way into
eval.** The one exception is mini-swe-agent's tool-calling harness (the Open-SWE v1.1/v1.2
`minisweagent` arms), which reports each command as a JSON object; a row whose results are that
JSON is rendered in mini-swe-agent's own text template, the `RETURNCODE` format `mini-coder` is
recorded in (`<warning>`/`<output_head>`/`<elided_chars>`/`<output_tail>` for a clipped output,
and an `<exception_info>` line for a command the harness killed).

---

## Observation formats

Three formats survive in the corpora. `Observation:` (SWE-ZERO) was retired with that corpus.

| format id | shape | who writes it |
|---|---|---|
| `RETURNCODE` | `<returncode>N</returncode>` + `<output>…</output>` | `mini-coder`, `mini-coder-rs`, `open-swe-traces-*` — `minisweagent` arms |
| `SWE_AGENT` | `OBSERVATION:` then the output | `open-swe-traces-*` — `sweagent` arms |
| `OPENHANDS` | bare tool output; **bash** calls close with an exit-code trailer | `open-swe-traces-*` — `openhands` arms, `swe-hero` |

The OpenHands trailer is structured and must be reproduced:

```
[The command completed with exit code 0.]
[Current working directory: /workspace/<repo>]
[Command finished with exit code 0]
```

Editor-style OpenHands observations carry no trailer — they open with
`File created successfully at: PATH` or `Here's the result of running `cat -n` on PATH:`.

### Format is detected per sample, never from the source name

`open-swe-traces-*` merges SWE-agent, OpenHands and mini-swe-agent arms under one name, so the source name cannot
determine the format, and the manifest cannot carry it (manifest `rows_meta` only ever reaches the
sampler, never the worker or judge API).

`observation_format.detect_format(sample_id, messages)` instead reads the format off **the
trajectory's own first environment turn** — the first `user` message that follows an `assistant`
message; the leading `user` message is the task. This is always safe because the sampler never cuts
at the first assistant turn, so every sampled prefix carries at least one real observation. The
fallback (no observation in the transcript) is `RETURNCODE` for `mini-coder*` ids and `OPENHANDS`
otherwise — OpenHands because its check is the permissive one, so a wrong guess cannot reject an
otherwise good observation.

### What the simulator must emit

`judge_api` builds the simulator request with `simulation_messages`: the system prompt is
`BASE_PROMPT` + the detected format's `OUTPUT FORMAT` section, and the user message holds the
transcript followed by the repo-context block (when grounding is available) and any retry note.
Keeping the per-turn block out of the system prompt lets providers cache the transcript prefix
across turns. Output is then gated by `observation_format.valid_output`:

| format | accepted |
|---|---|
| `RETURNCODE` | starts `<returncode>`, contains `</returncode>` and `<output>\n`, ends `\n</output>` |
| `SWE_AGENT` | starts `OBSERVATION:` |
| `OPENHANDS` | anything non-empty that does **not** open with another format's marker |

Rejected output is retried, then falls back to `empty_output(fmt)` for that format. Synthetic
observations the harness injects itself — task submitted, no bash command found, empty output — go
through `wrap(body, fmt)` so they match the trajectory too.

### Grounding comes first: the command is actually executed

The simulator is now the *fallback*, not the first resort. `ObservationSimulationService.simulate`
resolves each turn in this order:

1. **Absent tool** — `absent_tool_output(command)` recognises commands whose tool does not exist in
   this environment (a missing `pytest`, `pip install` in a sealed box) and returns that tool's
   canonical refusal text with the right returncode. No model call.
2. **Grounded execution** — the **repo-context service** (`src/repo_context_service/`) resolves the
   sample id to a repository + commit, fetches a snapshot, and *runs the command against it*:
   - `core.py:_resolve_sha` picks the tree the agent actually worked on, which is **not** the commit
     named in the id for three sources. `mini-coder*` ids name a branch of the swesmith mirror
     (`github.com/swesmith/<owner>__<repo>.<short-commit>`, history `Initial commit` (clean) <-
     `Bug Patch` [<- `Remove F2P Tests`]): Python/Go trajectories ran at the branch head (bug in
     place, fail-to-pass test files deleted), the Rust ones (`mini-coder-rs`) at `Bug Patch`, because
     a Rust "test file" is the whole source file and the head deletes it. `swe-hero` ids name the
     R2E-Gym *fix* commit; the agent worked on its parent. Verified 2026-09-21 against real
     trajectories: 40/40 pre-edit reads of the bug files match the chosen tree (upstream commit:
     16/40; swe-hero fix commit: 383/722 lines vs 722/722 for the parent). Cached resolutions carry
     `rule=_SHA_RULE`; bump it when this mapping changes and old entries are ignored.
   - `command_search.py` executes `find` / `grep` / `ls` / `sed` / `cat` and friends — BRE→Python
     translation, `-prune`/`-o` rewriting, `-name`/`-iname`/`-path`, POSIX classes, `2>/dev/null`.
   - `git_sim/` executes a subset of `git` (log, show, diff, status, branch…) against the snapshot,
     including patch/diff rendering and a session view of the working tree. `git show <sha>` only
     renders commits from the history already served for the task. A swesmith mirror's history is
     served as a single commit with no parent and no commit patches, because its `Bug Patch` and
     `Remove F2P Tests` commits are the answer and the hidden tests. That commit is the upstream
     commit the mirror was built from (real sha and subject, cached under `<cache>/upstream/`); it
     falls back to the mirror head as `Initial commit` when GitHub no longer has that commit or, for
     a `pr_<N>` task, when its subject names `#N`. `git remote -v` names the upstream repository.
   - `overlay.py` keeps an in-memory write overlay, so the candidate's *own* edits — including full
     `sed -i` emulation — are visible to its later reads.
   - `core.py:SCAFFOLDS` records how each source's machines print what the snapshot cannot show:
     `ls` widths, git ref decoration, short-hash length, the branch line of `git status`, the
     shell's error wording. `scaffold_for(source, fmt, instance_id)` takes the most specific entry:
     the source itself (`open-swe-traces-v1.2`) before its family (`open-swe-traces`, see
     `source_family`), and the instance's task set before the source's default. Open-SWE's
     machines differ by task set (`task_source`: `owner_repo_pr<N>` ids are Scale-SWE): SWE-rebench-V2
     ones are at a detached HEAD git cannot name (`Not currently on any branch.`), Scale-SWE ones are
     on the harness's `scaleswe` branch under openhands and v1.2's mini-swe-agent and at a named
     detached HEAD (`HEAD detached at <short>`) under swe-agent.
     mini-swe-agent runs commands in dash (`/bin/sh: 1: cd: can't cd to X`, exit 2), off a terminal
     (one name per `ls` line, no ref decoration). Where one entry covers machines the id cannot
     tell apart, it follows the most common one: v1.1's mini-swe-agent Scale-SWE rows are detached
     at the commit (58%) or on `scaleswe`, v1.2's SWE-rebench-V2 rows detached (67%) or on `main`,
     and Affine's PR tasks on the repository's own branch (SWE-Lego, 47%), detached at the commit
     (R2E-Gym, Multi-SWE) or detached unnamed (SWE-rebench-V2).

   When this produces an exact result, the response carries `exact_output` + `exact_returncode` and
   **that is the observation** (wrapped in the trajectory's format). No LLM is involved at all. The
   grounding block also carries `GIT SEMANTICS` notes and, for `&&` chains, per-stage `CHAIN
   EVIDENCE`.
3. **Transcribe** — if grounding produced a `COMMAND OUTPUT` block but not an exact result, the
   simulator is handed only `$ <command>` plus that block and told to transcribe it
   (`TRANSCRIBE_PROMPT`), which removes its freedom to invent.
4. **Simulate** — otherwise the LLM ladder: `ALBEDO_JUDGE_SIMULATION_MODEL`
   (`deepseek/deepseek-v4-flash-0731`) over **one rung per provider** in
   `ALBEDO_JUDGE_SIMULATION_PROVIDERS` (`deepseek,cloudflare`), rotated so each rung leads with a
   different provider and rungs after the first are forced through OpenRouter; then the evaluator
   model as the final rung. Each rung gets a single parse attempt — the ladder itself is the retry
   mechanism, because every extra in-rung attempt sits on the turn barrier's critical path.

Candidates are not merely accepted or rejected: each is repaired (`repair_output` +
`repair_to_contract`) and then **ranked** against the command's output contract, and the best-ranked
candidate across the whole ladder is kept. The contract comes from `command_contract(command)` and
`output_expectation(command)`:

| contract | meaning |
|---|---|
| `must_print` | this command always writes something; an empty observation is rejected and re-asked (`MUST_PRINT_RETRY`) |
| `may_be_silent` | silence is a legitimate result (a successful `sed -i`, a matchless `grep -q`) |
| `not_derivable` | the output cannot be inferred from the repo at all |

Looping output (25 identical consecutive lines, or a fully periodic 512-char tail) is collapsed
before use.

---

## Sampling

Deterministic and seeded by the submission's `block_hash` — the same submission always draws the
same samples, and no miner can influence the draw.

`multi_source_manifest_sample_ids` pools **one random rollout per unique `instance_id` across all
sources**, then fills a stratified grid:

- **phase** (`STEP_TRIM`) — where the trajectory is cut, anchored on the instance's `first_edit`:
  `cold` 65% (turn 1 or 2), `pre_edit` 15% (`first_edit - 2`), `at_edit` 20% (`first_edit`).
- **bug family** (`FAMILY_MIX`) — `pr` 50%, `lm` 15%, `combine` 10%, `mechanical` 25%.
- `REPO_CAP = 2` — at most two samples from any one repository.
- `NON_BENCHMARK_LANGUAGE_FRACTION = 0.30` — 30% of the draw is non-`python`.
- `MAX_PREFIX_CHARS = 54_000` — prefixes above this are skipped.

Sample ids are `"<source>/data/train-XXXXX.parquet:<row>:<turn>"`. Defaults: `sample_count = 100`
(`ALBEDO_EVAL_SAMPLE_COUNT`; the function's own default is 64). The number of candidate turns per
sample is **not** a single constant — `HORIZON_STRATA = (12, 16)` in
`evaluator/shared/questions.py` assigns a horizon round-robin within each phase bucket, and
`ALBEDO_REMOTE_TRAJECTORY_ASSISTANT_TURNS` (8) is only the fallback when no horizon is assigned.

## The manifest

`scripts/build_manifest.py` writes `manifest.json` — per source: repo, shard list, per-shard
sha256, and `rows_meta` (one entry per row: `instance_id`, `first_edit`, `family`, `repo`,
`language`, `verified`, prefix sizes). The sampler requires `rows_meta`; single-source manifests are
rejected.

The manifest is **hash-pinned**: `ALBEDO_EVAL_DATASET_MANIFEST_HASH` (and the sanity service's copy)
must match the file's sha256, so validators cannot silently drift onto different data. Rebuilding
prints the new hash and the settings to repin. A rows_meta-stripped `manifest.meta.json` is written
alongside for the dashboard.
