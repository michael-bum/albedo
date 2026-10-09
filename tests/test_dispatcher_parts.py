"""Pre-eval simulates a command the repository answers in parts the way the eval does."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from albedo_config import JudgeSettings
from albedo_eval_service.repo_context_client import Grounding
from albedo_eval_service.shared.observation_format import PYTEST_MISSING, RETURNCODE
from sanity_service import dispatcher as D

PARTS = [
    {"kind": "exact", "separator": "", "output": "SIZE = 1\n", "returncode": 0},
    {"kind": "gap", "separator": "&&", "command": "python run.py", "context": "GAP CONTEXT"},
    {"kind": "exact", "separator": "&&", "output": "# demo\n", "returncode": 0},
    {"kind": "exact", "separator": "||", "output": "fallback ran\n", "returncode": 0},
]
CHAIN = "cat a.py && python run.py && cat README.md || echo fallback ran"


class Repo:
    def __init__(self, grounding: Grounding):
        self.grounding = grounding
        self.calls = 0

    async def context_for(self, sample_id, assistant_output, messages=None):
        self.calls += 1
        return self.grounding


class ScriptedClient:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(raw=self.answers.pop(0), error=None)


def _simulate(command: str, client, repo: Repo, monkeypatch) -> str:
    monkeypatch.setattr(D, "detect_format", lambda *_a, **_k: RETURNCODE)
    state = SimpleNamespace(
        sample_id="mini-coder/data/train-00000.parquet:0:1",
        prompt="fix the bug",
        messages=[{"role": "user", "content": "fix the bug"}],
        turns=[],
        shared_messages=1,
    )
    return asyncio.run(
        D._simulate_observation_uncached(
            client=client,
            settings=JudgeSettings(engy_api_key="", simulation_model="stub-model"),
            eval_run_id="run",
            state=state,
            assistant_output=f"```bash\n{command}\n```",
            repo_context=repo,
        )
    )


def test_a_gap_is_simulated_alone_and_the_exact_parts_are_set_around_it(monkeypatch):
    client = ScriptedClient(["<returncode>0</returncode>\n<output>\nran ok\n</output>"])
    repo = Repo(Grounding("WHOLE COMMAND CONTEXT", None, None, "", None, PARTS))
    observation = _simulate(CHAIN, client, repo, monkeypatch)
    assert (
        observation == "<returncode>0</returncode>\n<output>\nSIZE = 1\nran ok\n# demo\n</output>"
    )
    assert len(client.calls) == 1
    asked = "\n".join(m["content"] for m in client.calls[0]["messages"])
    assert "GAP CONTEXT" in asked and "WHOLE COMMAND CONTEXT" not in asked
    # the eval's layout: the sample's messages, then the facts, then the trajectory's own turns
    assert client.calls[0]["messages"][1]["content"].startswith(
        "### user\nfix the bug\n\n### repository facts\nGAP CONTEXT\n\n### assistant\n"
    )


def test_a_failing_gap_skips_its_and_branch_and_runs_its_or_branch(monkeypatch):
    client = ScriptedClient(["<returncode>1</returncode>\n<output>\nTraceback: boom\n</output>"])
    repo = Repo(Grounding("WHOLE COMMAND CONTEXT", None, None, "", None, PARTS))
    assert _simulate(CHAIN, client, repo, monkeypatch) == (
        "<returncode>0</returncode>\n<output>\nSIZE = 1\nTraceback: boom\nfallback ran\n</output>"
    )


def test_a_chain_with_a_canned_stage_is_answered_by_grounding_not_the_refusal(monkeypatch):
    def no_llm(**_kwargs):
        raise AssertionError("the simulator was called")

    repo = Repo(Grounding(None, "grounded", 0, ""))
    client = SimpleNamespace(complete=no_llm)
    assert "grounded" in _simulate("grep -n x f && pytest -q", client, repo, monkeypatch)
    # a lone canned command never needs the repository
    assert PYTEST_MISSING in _simulate("pytest -q", client, repo, monkeypatch)
    assert repo.calls == 1
