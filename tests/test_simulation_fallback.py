from __future__ import annotations

import asyncio
from uuid import uuid4

from albedo_config import JudgeSettings
from albedo_eval_service.judge_api import ObservationSimulationService, SimulateObservationRequest
from albedo_eval_service.judge_llm_client import JudgeRawResponse
from albedo_eval_service.repo_context_client import Grounding
from albedo_eval_service.simulator.prompt_simulator import MUST_PRINT_RETRY


class StubClient:
    """Always returns unusable content, recording the provider ladder."""

    def __init__(self):
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return JudgeRawResponse(model=kwargs["model"], provider="stub", raw="")


def _request() -> SimulateObservationRequest:
    return SimulateObservationRequest(
        eval_run_id=str(uuid4()),
        sample_id="mini-coder/data/train-00000.parquet:0:1",
        prompt="fix the bug",
        assistant_output="I will look around.\n```bash\ncat src/main.py\n```",
        messages=[{"role": "user", "content": "fix the bug"}],
    )


def test_simulation_ladder_tries_each_provider_before_expensive_fallback():
    settings = JudgeSettings(
        engy_api_key="",
        simulation_model="deepseek/deepseek-v4-flash-0731",
        simulation_providers="deepseek,cloudflare",
    )
    client = StubClient()
    service = ObservationSimulationService(settings, client, None)

    observation = asyncio.run(service.simulate(_request()))

    models = [c["model"] for c in client.calls]
    orders = [(c["provider"] or {}).get("order") for c in client.calls]
    assert models[:3] == [
        "deepseek/deepseek-v4-flash-0731",  # rung 1: deepseek provider first
        "deepseek/deepseek-v4-flash-0731",  # rung 2: cloudflare provider first
        settings.evaluator_model,  # rung 3: expensive fallback model last
    ]
    assert orders[0][0] == "deepseek"
    assert orders[1][0] == "cloudflare"
    assert orders[2] is None
    # `cat` must print, so a silent ladder earns one last directed ask before we give up
    assert len(client.calls) == 4
    assert models[3] == "deepseek/deepseek-v4-flash-0731"
    assert MUST_PRINT_RETRY in client.calls[3]["messages"][-1]["content"]
    # everything failed -> deterministic empty observation, never an exception
    assert "returncode" in observation


def test_simulation_single_model_uses_evaluator_provider():
    settings = JudgeSettings(engy_api_key="", simulation_model="")
    client = StubClient()
    service = ObservationSimulationService(settings, client, None)

    asyncio.run(service.simulate(_request()))

    # one ladder rung, then the must-print retry for a read that came back silent
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == settings.evaluator_model
    assert (client.calls[0]["provider"] or {}).get("order") == [
        p.strip() for p in settings.evaluator_providers.split(",")
    ]


class ScriptedClient:
    """Answers each simulation with the next scripted observation, recording what was asked."""

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return JudgeRawResponse(model=kwargs["model"], provider="stub", raw=self.answers.pop(0))


class PartsContext:
    """A repo-context client whose answer splits the command into parts."""

    def __init__(self, parts: list[dict]):
        self.parts = parts

    async def context_for(self, sample_id, assistant_output, messages=None):
        return Grounding("WHOLE COMMAND CONTEXT", None, None, "", None, self.parts)


PARTS = [
    {"kind": "exact", "separator": "", "output": "SIZE = 1\n", "returncode": 0},
    {"kind": "gap", "separator": "&&", "command": "python run.py", "context": "GAP CONTEXT"},
    {"kind": "exact", "separator": "&&", "output": "# demo\n", "returncode": 0},
    {"kind": "exact", "separator": "||", "output": "fallback ran\n", "returncode": 0},
]


def _chain_request() -> SimulateObservationRequest:
    return SimulateObservationRequest(
        eval_run_id=str(uuid4()),
        sample_id="mini-coder/data/train-00000.parquet:0:1",
        prompt="fix the bug",
        assistant_output=(
            "```bash\ncat a.py && python run.py && cat README.md || echo fallback ran\n```"
        ),
        messages=[{"role": "user", "content": "fix the bug"}],
    )


def test_a_gap_is_simulated_alone_and_the_exact_parts_are_set_around_it():
    settings = JudgeSettings(engy_api_key="", simulation_model="stub-model")
    client = ScriptedClient(["<returncode>0</returncode>\n<output>\nran ok\n</output>"])
    service = ObservationSimulationService(settings, client, PartsContext(PARTS))

    observation = asyncio.run(service.simulate(_chain_request()))

    assert (
        observation == "<returncode>0</returncode>\n<output>\nSIZE = 1\nran ok\n# demo\n</output>"
    )
    # one call, for the gap alone, with the gap's own context rather than the whole command's
    assert len(client.calls) == 1
    asked = "\n".join(m["content"] for m in client.calls[0]["messages"])
    assert "GAP CONTEXT" in asked and "WHOLE COMMAND CONTEXT" not in asked


def test_a_gap_that_fails_skips_what_its_and_guards_and_runs_what_its_or_guards():
    settings = JudgeSettings(engy_api_key="", simulation_model="stub-model")
    client = ScriptedClient(["<returncode>1</returncode>\n<output>\nTraceback: boom\n</output>"])
    service = ObservationSimulationService(settings, client, PartsContext(PARTS))

    observation = asyncio.run(service.simulate(_chain_request()))

    # `cat README.md` does not run after the failure, `echo fallback ran` does: the chain ends
    # with its status, and the command is not simulated a second time whole
    assert observation == (
        "<returncode>0</returncode>\n<output>\nSIZE = 1\nTraceback: boom\nfallback ran\n</output>"
    )
    assert len(client.calls) == 1


def test_a_gap_whose_status_is_unknown_where_it_decides_what_runs_sends_the_command_whole():
    # a swe-agent observation carries no exit status: what runs after `&&` cannot be settled
    settings = JudgeSettings(engy_api_key="", simulation_model="stub-model")
    whole = "OBSERVATION:\nSIZE = 1\nran\n# demo"
    client = ScriptedClient(["OBSERVATION:\nran", whole])
    service = ObservationSimulationService(settings, client, PartsContext(PARTS))
    request = _chain_request().model_copy(
        update={
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "```bash\nls\n```"},
                {"role": "user", "content": "OBSERVATION:\na.py"},
            ]
        }
    )

    observation = asyncio.run(service.simulate(request))

    assert observation == whole
    assert len(client.calls) == 2
    asked = "\n".join(m["content"] for m in client.calls[1]["messages"])
    assert "WHOLE COMMAND CONTEXT" in asked
