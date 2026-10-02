from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from albedo_config import RemoteSettings, ScoreBridgeClientSettings
from albedo_eval_service.remote.dataset import EvalSample
from albedo_eval_service.scoring import score_bridge_client
from albedo_eval_service.scoring.score_bridge import ScoreBridgeHub, ScoreBridgeUnavailable
from albedo_eval_service.scoring.score_bridge_client import run_bridge
from albedo_eval_service.scoring.scoring_client import build_scorer
from albedo_eval_service.shared.models import (
    Challenger,
    DatasetConfig,
    EvalRequest,
    PreviousKing,
    ScoringConfig,
)


def test_build_scorer_supports_websocket_backend():
    scorer = build_scorer(RemoteSettings(scoring_backend="websocket", scoring_timeout_seconds=1))

    assert scorer.__class__.__name__ == "WebSocketScoringClient"


def test_score_bridge_hub_reports_unavailable_without_client():
    hub = ScoreBridgeHub()

    with pytest.raises(ScoreBridgeUnavailable):
        hub.request({"hello": "world"}, timeout_seconds=0.01)


def test_score_bridge_client_reconnects_after_disconnect(monkeypatch):
    attempts = []

    async def fake_run_once(settings, *, headers):
        attempts.append(headers)
        raise RuntimeError("socket dropped")

    async def fake_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr("albedo_eval_service.scoring.score_bridge_client._run_once", fake_run_once)
    monkeypatch.setattr("albedo_eval_service.scoring.score_bridge_client.asyncio.sleep", fake_sleep)

    settings = ScoreBridgeClientSettings(
        remote_auth_token="remote-token", reconnect_min_seconds=0.01
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_bridge(settings))

    assert attempts == [{"Authorization": "Bearer remote-token"}]


def _eval_request() -> EvalRequest:
    return EvalRequest(
        eval_run_id=uuid4(),
        submission_id=uuid4(),
        challenger=Challenger(model_uri="challenger", model_hash="sha256:chal"),
        previous_king=PreviousKing(model_uri="king", model_hash="sha256:king", king_version=1),
        dataset=DatasetConfig(
            version="AlienKevin/SWE-ZERO-12M-trajectories",
            manifest_uri="hf://dataset",
            manifest_hash="sha256:manifest",
            sample_count=1,
            sample_seed="seed",
            sampling_algo="explicit",
        ),
        scoring=ScoringConfig(judge_config_hash="sha256:judge"),
        artifact_prefix="local://run",
    )


def test_websocket_scorer_starts_category_prep_over_bridge(monkeypatch):
    calls = []

    def fake_request(payload, *, timeout_seconds, endpoint="/score-batch"):
        calls.append({"payload": payload, "timeout_seconds": timeout_seconds, "endpoint": endpoint})
        return {"category_prep_id": "prep-123"}

    monkeypatch.setattr(
        "albedo_eval_service.scoring.scoring_client.score_bridge_hub.request", fake_request
    )
    scorer = build_scorer(RemoteSettings(scoring_backend="websocket", scoring_timeout_seconds=7))

    prep_id = scorer.start_category_prep(
        request=_eval_request(),
        samples=[EvalSample(sample_id="data/train-00000.parquet:0:0", prompt="Prompt")],
    )

    assert prep_id == "prep-123"
    assert calls[0]["endpoint"] == "/category-prep"
    assert calls[0]["timeout_seconds"] == 7
    assert calls[0]["payload"]["samples"] == [
        {
            "sample_id": "data/train-00000.parquet:0:0",
            "prompt": "Prompt",
            "messages": None,
            "assistant_turns": 12,
            "submit_marker": "",
            "submit_command": "",
        }
    ]


def test_websocket_scorer_simulates_observation_over_bridge(monkeypatch):
    calls = []

    def fake_request(payload, *, timeout_seconds, endpoint="/score-batch"):
        calls.append({"payload": payload, "timeout_seconds": timeout_seconds, "endpoint": endpoint})
        return {"observation": "Observation: ok"}

    monkeypatch.setattr(
        "albedo_eval_service.scoring.scoring_client.score_bridge_hub.request", fake_request
    )
    scorer = build_scorer(RemoteSettings(scoring_backend="websocket", scoring_timeout_seconds=7))
    sample = EvalSample(
        sample_id="data/train-00000.parquet:0:0",
        prompt="Prompt",
        messages=[{"role": "user", "content": "Prompt"}],
    )

    observation = scorer.simulate_observation(
        request=_eval_request(), sample=sample, assistant_output="```bash\nls\n```"
    )

    assert observation == "Observation: ok"
    assert calls[0]["endpoint"] == "/simulate-observation"
    assert calls[0]["payload"]["assistant_output"] == "```bash\nls\n```"


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    @staticmethod
    def json() -> dict:
        return {"scoring_records": [], "summary": {"state": "ok"}}


class _DroppingJudge:
    """A judge client whose first `failures` posts die on the connection, as a pooled keep-alive
    socket the judge-api already closed does."""

    def __init__(self, failures: int):
        self.failures = failures
        self.posts = 0

    async def post(self, endpoint, json):
        self.posts += 1
        if self.posts <= self.failures:
            raise httpx.ReadError("")
        return _Response()


class _Socket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _bridge_message() -> dict:
    return {"type": "score_request", "request_id": "r1", "endpoint": "/score-batch", "payload": {}}


def test_bridge_client_retries_a_dropped_judge_connection_once(monkeypatch):
    monkeypatch.setattr(score_bridge_client.asyncio, "sleep", _no_sleep)
    judge, socket = _DroppingJudge(failures=1), _Socket()
    asyncio.run(
        score_bridge_client._handle_score_request(
            ScoreBridgeClientSettings(_env_file=None), socket, judge, _bridge_message()
        )
    )
    assert judge.posts == 2
    assert socket.sent == [{"type": "score_response", "request_id": "r1", "body": _Response.json()}]


def test_bridge_client_reports_a_connection_that_keeps_dropping(monkeypatch):
    monkeypatch.setattr(score_bridge_client.asyncio, "sleep", _no_sleep)
    judge, socket = _DroppingJudge(failures=5), _Socket()
    asyncio.run(
        score_bridge_client._handle_score_request(
            ScoreBridgeClientSettings(_env_file=None), socket, judge, _bridge_message()
        )
    )
    assert judge.posts == 2
    assert socket.sent[0]["error"].startswith("ReadError")


async def _no_sleep(_seconds: float) -> None:
    return None
