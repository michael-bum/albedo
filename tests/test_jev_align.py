"""Jev pair weights become fact clusters; without a key the caller falls back to exact groups."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from albedo_eval_service.evaluator.reference import jev_align
from albedo_eval_service.evaluator.reference.jev_align import (
    JevUnavailable,
    ask,
    average_linkage,
    jev_clusters,
    pair_weights,
)


def _m(category: str, statement: str) -> dict:
    return {"id": "m", "category": category, "statement": statement, "evidence": []}


def fake_jev(monkeypatch, weights: dict[str, float]) -> list[tuple[dict, list[str]]]:
    """Jev answering each pair with its weight; records every request as (state, question ids)."""
    sent: list[tuple[dict, list[str]]] = []

    async def answer(http, state, questions):
        sent.append((state, list(questions)))
        return {"answers": {q: {"noul": weights.get(q, 0.0)} for q in questions}}

    monkeypatch.setattr(jev_align, "ask", answer)
    return sent


def test_average_linkage_joins_while_the_mean_cross_weight_reaches_the_cut():
    singles = [["A1"], ["B1"], ["C1"]]
    pair = {("A1", "B1"): 0.9, ("A1", "C1"): 0.5, ("B1", "C1"): 0.62}
    weight = {**pair, **{(b, a): w for (a, b), w in pair.items()}}

    assert average_linkage(singles, weight) == [["A1", "B1", "C1"]]
    weight[("A1", "C1")] = weight[("C1", "A1")] = 0.4
    assert average_linkage(singles, weight) == [["A1", "B1"], ["C1"]]


def test_a_missing_key_is_unavailable_not_a_crash():
    with pytest.raises(JevUnavailable):
        asyncio.run(jev_clusters("task", [[{"statement": "a"}], [{"statement": "b"}]], api_key=""))


def test_word_for_word_milestones_stay_one_fact_whatever_their_categories(monkeypatch):
    """One reading calls it an action and another a verification. Exact matching joined them, so
    Jev is asked about the group once, through its first member, and cannot split it."""
    readings = [
        [_m("action", "Run the tests after the fix"), _m("explore", "Read x.py")],
        [_m("verification", "Run the tests after the fix"), _m("explore", "Open x.py")],
    ]
    sent = fake_jev(monkeypatch, {"A2|B2": 0.9})

    clusters = asyncio.run(jev_clusters("task", readings, api_key="k"))

    assert sorted(sorted(c) for c in clusters) == [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]
    assert {q for _, qs in sent for q in qs} == {"A1|A2", "A1|B2", "A2|B2"}
    assert "B1" not in sent[0][0], "a repeat is shown once, as its group's first member"


def test_milestone_alignment_splits_large_requests(monkeypatch):
    monkeypatch.setattr(jev_align, "CHUNK", 2)
    readings = [[_m("explore", t) for t in "ab"], [_m("explore", t) for t in "cd"]]
    sent = fake_jev(monkeypatch, {})

    asyncio.run(jev_clusters("task", readings, api_key="k"))

    assert [len(qs) for _, qs in sent] == [2, 2, 2, 2, 2, 2], "6 pairs in 3 requests per order"


def test_an_answer_jev_did_not_give_is_unavailable_not_a_zero():
    async def reply(**response) -> dict:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, **response))
        async with httpx.AsyncClient(transport=transport) as http:
            return await ask(http, {}, {"A1|B1": {}})

    for response in (
        {"json": {"answers": {}}},
        {"json": {"answers": {"A1|B1": {"noul": None}}}},
        {"json": [{"noul": 0.7}]},
        {"text": "<html>bad gateway</html>"},
    ):
        with pytest.raises(JevUnavailable):
            asyncio.run(reply(**response))
    answered = asyncio.run(reply(json={"answers": {"A1|B1": {"noul": 0.7}}}))
    assert answered["answers"]["A1|B1"]["noul"] == 0.7


def test_a_failed_request_cancels_the_ones_still_in_flight(monkeypatch):
    monkeypatch.setattr(jev_align, "CHUNK", 1)
    finished: list[bytes] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if b"A1|B1" in request.content:
            return httpx.Response(400, text="bad request")
        await asyncio.sleep(0.2)
        finished.append(request.content)
        return httpx.Response(200, json={"answers": {}})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(JevUnavailable):
                await pair_weights(
                    http, lambda order: {}, ["A1|B1", "A2|B2", "A3|B3"], lambda pair: {}
                )
        await asyncio.sleep(0.3)

    asyncio.run(run())
    assert finished == [], "nothing is still being paid for after the batch has failed"
