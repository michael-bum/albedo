"""Jev in the question stage: J1 groups within a milestone, J1.a drops cross-milestone duplicates,
J2 scores reference runs and hands back None for a run it cannot take."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from albedo_eval_service.evaluator.reference import jev_align, jev_questions
from albedo_eval_service.evaluator.reference.jev_align import JevUnavailable
from albedo_eval_service.evaluator.reference.jev_questions import (
    cross_milestone_duplicates,
    question_clusters,
    reference_scores,
)

MILESTONES = [{"id": "m1", "statement": "one"}, {"id": "m2", "statement": "two"}]


def fake_jev(monkeypatch, weights: dict[str, float], fail: set[str] = frozenset()):
    """Jev answering each pair with its weight; records every request as (state, question ids)."""
    sent: list[tuple[dict, list[str]]] = []

    async def answer(http, state, questions):
        if state.get("trajectory") in fail:
            raise JevUnavailable('HTTP 400: {"detail":{"error_type":"max_tokens_exceeded"}}')
        sent.append((state, list(questions)))
        return {
            "answers": {q: {"noul": weights.get(q, 0.0)} for q in questions},
            "usage": {"input_tokens": 10},
        }

    monkeypatch.setattr(jev_align, "ask", answer)
    monkeypatch.setattr(jev_questions, "ask", answer)
    return sent


def test_question_dedup_pairs_only_within_a_milestone_over_two_orders(monkeypatch):
    lists = [
        [{"milestone": "m1", "text": "a", "unearned": "x"}, {"milestone": "m2", "text": "b"}],
        [{"milestone": "m1", "text": "a'"}, {"milestone": "m2", "text": "c"}],
    ]
    sent = fake_jev(monkeypatch, {"A1|B1": 0.9, "A2|B2": 0.4})

    clusters = asyncio.run(question_clusters("task", MILESTONES, lists, api_key="k"))

    assert sorted(sorted(c) for c in clusters) == [[(0, 0), (1, 0)], [(0, 1)], [(1, 1)]]
    assert len(sent) == 2, "orders 1-2 only, no second round for undecided pairs"
    assert {q for _, qs in sent for q in qs} == {"A1|B1", "A2|B2"}
    state = sent[0][0]
    assert state["milestones"] == {"m1": "one", "m2": "two"}
    assert state["A1"] == {"milestone": "m1", "question": "a", "not_earned_by": "x"}


def test_question_dedup_splits_large_requests(monkeypatch):
    monkeypatch.setattr(jev_align, "CHUNK", 2)
    lists = [[{"milestone": "m1", "text": t} for t in "abc"], [{"milestone": "m1", "text": "d"}]]
    sent = fake_jev(monkeypatch, {})

    asyncio.run(question_clusters("task", MILESTONES, lists, api_key="k"))

    assert [len(qs) for _, qs in sent] == [2, 2, 2, 2, 2, 2], "6 pairs in 3 requests per order"


def test_cross_milestone_duplicate_drops_the_later_question(monkeypatch):
    questions = [
        {"id": "q_01", "milestone": "m1", "text": "a", "example_bad": "x"},
        {"id": "q_02", "milestone": "m2", "text": "a again"},
        {"id": "q_03", "milestone": "m2", "text": "other"},
    ]
    sent = fake_jev(monkeypatch, {"q_01|q_02": 0.8, "q_01|q_03": 0.3})

    dropped = asyncio.run(cross_milestone_duplicates("task", MILESTONES, questions, api_key="k"))

    assert dropped == {"q_02"}
    assert "q_02|q_03" not in {q for _, qs in sent for q in qs}, "same milestone is J1's job"
    assert len(sent) == 6, "orders 1-2, then orders 3-6 for the undecided q_01|q_03"


def test_cross_milestone_duplicate_skips_a_pair_touching_a_dropped_question(monkeypatch):
    questions = [
        {"id": "q_01", "milestone": "m1", "text": "a"},
        {"id": "q_02", "milestone": "m2", "text": "a"},
        {"id": "q_03", "milestone": "m3", "text": "a"},
    ]
    fake_jev(monkeypatch, {"q_01|q_02": 0.95, "q_01|q_03": 0.9, "q_02|q_03": 0.9})

    dropped = asyncio.run(cross_milestone_duplicates("task", MILESTONES, questions, api_key="k"))

    assert dropped == {"q_02", "q_03"}


def test_reference_scores_returns_none_for_a_run_jev_cannot_take(monkeypatch):
    questions = [{"id": "q_01", "text": "a"}, {"id": "q_02", "text": "b"}]
    sent = fake_jev(monkeypatch, {"q_01": 0.9, "q_02": 0.1}, fail={"too long"})

    scores = asyncio.run(reference_scores(["run 1", "too long", "run 3"], questions, api_key="k"))

    assert scores == [{"q_01": 0.9, "q_02": 0.1}, None, {"q_01": 0.9, "q_02": 0.1}]
    assert sent[0][0] == {"trajectory": "run 1"}


def test_every_question_stage_step_needs_a_key():
    with pytest.raises(JevUnavailable):
        asyncio.run(reference_scores(["doc"], [{"id": "q_01", "text": "a"}], api_key=""))


def test_word_for_word_questions_under_a_milestone_are_one_group_before_jev_is_asked(monkeypatch):
    lists = [
        [{"milestone": "m1", "text": "a"}, {"milestone": "m1", "text": "b"}],
        [{"milestone": "m1", "text": "A "}, {"milestone": "m1", "text": "c"}],
    ]
    sent = fake_jev(monkeypatch, {})

    clusters = asyncio.run(question_clusters("task", MILESTONES, lists, api_key="k"))

    assert sorted(sorted(c) for c in clusters) == [[(0, 0), (1, 0)], [(0, 1)], [(1, 1)]]
    assert {q for _, qs in sent for q in qs} == {"A1|A2", "A1|B2", "A2|B2"}
    assert "B1" not in sent[0][0], "a repeat is shown once, as its group's first member"


def test_reference_scores_hands_a_run_with_an_answer_missing_to_the_fallback(monkeypatch):
    """A question Jev did not answer is not one the run failed to earn: the run goes to GLM."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"q_01": {"noul": 0.9}}})

    monkeypatch.setattr(
        jev_questions,
        "jev_http",
        lambda api_key: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    questions = [{"id": "q_01", "text": "a"}, {"id": "q_02", "text": "b"}]

    assert asyncio.run(reference_scores(["run 1"], questions, api_key="k")) == [None]
