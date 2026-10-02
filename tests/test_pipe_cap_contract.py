from __future__ import annotations

import asyncio

from test_repo_context_core import FULL_SHA, make_service, make_snapshot

from albedo_config import JudgeSettings
from albedo_eval_service.judge_api import (
    Grounding,
    ObservationSimulationService,
    SimulateObservationRequest,
)
from albedo_eval_service.judge_llm_client import JudgeRawResponse
from albedo_eval_service.shared.observation_format import RETURNCODE, observation_body

VIEW = [f"{n:>6}\tdef test_view_{n}(self): pass" for n in range(1, 61)]
RUN_TAIL = ["pyramid.exceptions.PredicateMismatch: view_7", "fallback view never tried"]
TRACEBACK = ["Traceback (most recent call last):", '  File "repro.py", line 9, in <module>']
APP = (
    "def load(path):\n    return open(path).read()\n\n\n"
    "def save(path, text):\n    open(path, 'w').write(text)\n"
)


class Scripted:
    def __init__(self, raw: str):
        self.raw = raw

    async def complete(self, **kwargs):
        return JudgeRawResponse(model=kwargs["model"], provider="fake", raw=self.raw)


class SnapshotContext:
    """The repo-context service over a local snapshot, answering the way /repo-context does."""

    def __init__(self, service):
        self.service = service

    async def context_for(self, sample_id, assistant_output, messages=None):
        found = self.service.repo_context_for_instance(
            "swe-zero", "o__r-1", assistant_output, messages
        )
        return Grounding(
            found.context,
            found.exact_output,
            found.exact_returncode,
            found.state,
            found.leading_output,
        )


def simulate(command: str, body: list[str], repo_context=None) -> list[str]:
    client = Scripted("<returncode>1</returncode>\n<output>\n" + "\n".join(body) + "\n</output>")
    service = ObservationSimulationService(JudgeSettings(simulation_model=""), client, repo_context)
    observation = asyncio.run(
        service.simulate(
            SimulateObservationRequest(
                eval_run_id="run",
                sample_id="mini-coder/data/train-00000-of-00001.parquet:0:0",
                prompt="task",
                messages=[{"role": "user", "content": "task"}],
                assistant_output=f"```bash\n{command}\n```",
            )
        )
    )
    return observation_body(observation, RETURNCODE).splitlines()


def snapshot_context(tmp_path, monkeypatch) -> SnapshotContext:
    service = make_service(tmp_path)
    make_snapshot(service, {"src/app.py": APP})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    return SnapshotContext(service)


def test_a_pipe_cap_on_the_last_stage_of_a_chain_keeps_the_earlier_stages_and_the_program_output():
    # the shell caps only the script: the 60-line view prints above the last two lines of its run
    served = simulate(
        "sed -n '1,60p' tests/test_views.py && python repro.py 2>&1 | tail -2", VIEW + RUN_TAIL
    )

    assert served == VIEW + RUN_TAIL


def test_a_tail_cap_keeps_the_last_lines_even_behind_a_cd_prefix():
    # too many lines for `tail -2`: the repair must keep the error at the end, not the first lines
    served = simulate("cd /workspace/repo && python repro.py 2>&1 | tail -2", TRACEBACK + RUN_TAIL)

    assert served == RUN_TAIL


def test_the_last_stage_cap_cuts_only_the_program_output_after_grounded_stages(
    tmp_path, monkeypatch
):
    # the grep runs against the snapshot, so its two lines are known: only the script's four lines
    # after them are held to `tail -2`
    grep = ["1:def load(path):", "5:def save(path, text):"]
    served = simulate(
        'grep -n "def " src/app.py && python repro.py 2>&1 | tail -2',
        grep + TRACEBACK + RUN_TAIL,
        snapshot_context(tmp_path, monkeypatch),
    )

    assert served == grep + RUN_TAIL


def test_the_last_stage_cap_is_skipped_when_the_observation_does_not_open_with_grounded_output(
    tmp_path, monkeypatch
):
    # the model rendered the grep differently: nothing marks where the script's output starts, so
    # cutting would eat into the earlier stage
    rendered = ["src/app.py:1:def load(path):", "src/app.py:5:def save(path, text):"]
    served = simulate(
        'grep -n "def " src/app.py && python repro.py 2>&1 | tail -2',
        rendered + TRACEBACK + RUN_TAIL,
        snapshot_context(tmp_path, monkeypatch),
    )

    assert served == rendered + TRACEBACK + RUN_TAIL
