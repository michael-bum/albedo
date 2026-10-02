from __future__ import annotations

import asyncio
from pathlib import Path

from albedo_config import JudgeSettings, RepoContextSettings
from albedo_eval_service.judge_api import ObservationSimulationService, SimulateObservationRequest
from albedo_eval_service.repo_context_client import Grounding, stitch_parts
from albedo_eval_service.shared.observation_format import PYTEST_MISSING, command_chain, flat_chain
from repo_context_service.core import RepoContextService
from repo_context_service.git_sim import GitMeta
from repo_context_service.overlay import build_overlay


class Repo:
    """Grounding that answers every command exactly, and counts how often it is asked."""

    def __init__(self):
        self.asked = 0

    async def context_for(self, sample_id, assistant_output, messages=None):
        self.asked += 1
        return Grounding(None, "grounded", 0, "")


def _run(command: str, repo: Repo) -> str:
    service = ObservationSimulationService(
        JudgeSettings(evaluator_model="z-ai/glm-5.2"), None, repo
    )
    return asyncio.run(
        service.simulate(
            SimulateObservationRequest(
                eval_run_id="run",
                sample_id="mini-coder/x:0:0",
                prompt="task",
                messages=[{"role": "user", "content": "task"}],
                assistant_output=f"```bash\n{command}\n```",
            )
        )
    )


def test_each_stage_keeps_the_separator_that_joins_it_to_the_one_before():
    assert command_chain("a && b; c || d\ne &&\n  f") == [
        ("", "a"), ("&&", "b"), (";", "c"), ("||", "d"), ("\n", "e"), ("&&", "f"),
    ]  # fmt: skip


def test_a_compound_command_is_never_split():
    assert flat_chain("for f in a b; do pytest $f; done") is None
    assert flat_chain("grep -n x f && pytest -q") == [("", "grep -n x f"), ("&&", "pytest -q")]


def test_a_chain_with_a_canned_stage_is_answered_by_grounding_not_the_refusal():
    repo = Repo()
    assert _run("grep -n x f && pytest -q", repo) == (
        "<returncode>0</returncode>\n<output>\ngrounded\n</output>"
    )
    # a lone canned command never needs the repository
    assert PYTEST_MISSING in _run("pytest -q", repo)
    assert repo.asked == 1


def test_a_gap_run_gated_by_and_ends_where_a_semicolon_stage_starts(tmp_path):
    """`false && cat a.txt && false; python3 ...`: the `;` stage runs whatever came before it, so
    it must not be folded into the `&&`-gated gap before it and skipped with it."""
    files = {"a.txt": "one\n"}
    service = RepoContextService(RepoContextSettings(_env_file=None, cache_dir=str(tmp_path)))
    overlay = build_overlay([], sorted(files), files.get)
    command = "false && cat a.txt && false; python3 -c 'print(2)'"
    parts = service._parts(
        Path("/nonexistent"), command, overlay, "returncode", GitMeta(), "", set()
    )
    assert [(p["kind"], p["separator"]) for p in parts] == [
        ("gap", ""), ("exact", "&&"), ("gap", "&&"), ("gap", ";"),
    ]  # fmt: skip

    async def answer(gap):
        return ("2", 0) if gap["command"].startswith("python3") else ("", 1)

    assert asyncio.run(stitch_parts(parts, answer)) == ("2", 0)
