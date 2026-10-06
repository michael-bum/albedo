"""Milestone alignment by Jev (TypeSafe System One), docs/SCORING.md.

Every pair of exact groups is asked whether both establish the same fact. Jev's answer moves with
the order it reads the state in far more than between identical repeats, so a pair's weight is its
mean Noul over several orders of the state: orders 1-2 for every pair, orders 3-6 for the pairs
still undecided. Groups are joined by average linkage at JOIN_AT. An answer Jev did not give is
JevUnavailable, never a 0: counted as one it would split a pair, or prune a question, on no verdict
at all.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger

from .vector_merge import exact_groups, member_id

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"
SAME_CRITERIA = {
    "true": "They establish the same thing, even if worded differently, at a different length, "
    "reached by a different route, or with one naming a detail the other omits.",
    "false": "They establish different things: for example where a defect is versus how it "
    "arises, what a fix changes versus how the fix is checked, or two different code sites.",
}
NEVER_SAME = {frozenset({"action", "verification"}), frozenset({"action", "claims"})}
FIRST_ORDERS, ORDERS = 2, 6
UNDECIDED = (0.2, 0.85)
JOIN_AT = 0.55
RETRY_STATUSES = {429, 500, 502, 503, 529}
CHUNK = 300  # pairs per request; every request repeats the whole state
TIMEOUT_SECONDS = 15.0
RETRIES = 2
BACKOFF = 1.5


class JevUnavailable(RuntimeError):
    pass


def milestone_card(milestone: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": milestone.get("category"),
        "statement": milestone.get("statement"),
        "evidence": [str(e.get("span", ""))[:300] for e in (milestone.get("evidence") or [])[:3]],
    }


def pair_question(pair: str) -> dict[str, Any]:
    a, b = pair.split("|")
    return {
        "type": "noul",
        "instructions": f"Do `{a}` and `{b}` establish the same fact about the code or the task?",
        "criteria": SAME_CRITERIA,
    }


def shuffled(cards: dict[str, Any], order: int) -> dict[str, Any]:
    ids = list(cards)
    random.Random(order).shuffle(ids)
    return {mid: cards[mid] for mid in ids}


def average_linkage(
    groups: list[list[str]], weight: dict[tuple[str, str], float]
) -> list[list[str]]:
    """`groups` joined while the best mean cross weight between two of them reaches JOIN_AT."""
    groups = [list(group) for group in groups]
    while len(groups) > 1:
        best, pair = JOIN_AT, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                cross = [weight[(a, b)] for a in groups[i] for b in groups[j]]
                mean = sum(cross) / len(cross)
                if mean >= best:
                    best, pair = mean, (i, j)
        if pair is None:
            break
        i, j = pair
        groups[i] += groups.pop(j)
    return groups


def join_groups(start: list[list[str]], means: dict[str, float]) -> list[list[str]]:
    """`start` joined by average linkage, each member weighing what its group's first member does.

    `means` is keyed `a|b` by first members; a pair it does not have weighs 0.
    """
    lead = {member: group[0] for group in start for member in group}
    weight = {
        (a, b): means.get(f"{lead[a]}|{lead[b]}", means.get(f"{lead[b]}|{lead[a]}", 0.0))
        for a in lead
        for b in lead
    }
    return average_linkage(start, weight)


def _checked(payload: Any, questions: dict[str, Any]) -> dict[str, Any]:
    """The payload, once every question asked has a numeric Noul in it; ValueError otherwise."""
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("no answers")
    for key in questions:
        answer = answers.get(key)
        noul = answer.get("noul") if isinstance(answer, dict) else None
        if isinstance(noul, bool) or not isinstance(noul, (int, float)):
            raise ValueError(f"no Noul for {key}")
    return payload


async def ask(
    http: httpx.AsyncClient, state: dict[str, Any], questions: dict[str, Any]
) -> dict[str, Any]:
    body = {"model": JEV_MODEL, "state": state, "questions": questions}
    error = ""
    for attempt in range(RETRIES + 1):
        try:
            response = await http.post(JEV_URL, json=body)
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                try:
                    return _checked(response.json(), questions)
                except ValueError as exc:
                    error = f"unreadable answer: {exc}"
                    break
            error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRY_STATUSES:
                break
        if attempt < RETRIES:
            await asyncio.sleep(BACKOFF * 2**attempt)
    raise JevUnavailable(error)


def jev_http(api_key: str) -> httpx.AsyncClient:
    if not api_key:
        raise JevUnavailable("ALBEDO_JUDGE_JEV_API_KEY is not set")
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10.0),
    )


async def pair_weights(
    http: httpx.AsyncClient,
    state: Callable[[int], dict[str, Any]],
    pairs: list[str],
    question: Callable[[str], dict[str, Any]],
    *,
    second_round: bool = True,
) -> tuple[dict[str, float], int, int]:
    """Each pair's mean Noul over orders 1-2 of the state, then over orders 3-6 for the pairs still
    undecided (unless `second_round` is off), CHUNK pairs a request. Returns the weights, the input
    tokens spent and how many pairs went to the second round."""
    votes: dict[str, list[float]] = {pair: [] for pair in pairs}
    tokens = 0

    async def vote(orders: range, asked: list[str]) -> None:
        nonlocal tokens
        requests = [
            asyncio.create_task(
                ask(http, state(order), {p: question(p) for p in asked[i : i + CHUNK]})
            )
            for order in orders
            for i in range(0, len(asked), CHUNK)
        ]
        try:
            payloads = await asyncio.gather(*requests)
        finally:
            # one failed request fails them all, so the rest would only be paid for and dropped
            for request in requests:
                request.cancel()
            await asyncio.gather(*requests, return_exceptions=True)
        for payload in payloads:
            tokens += int((payload.get("usage") or {}).get("input_tokens") or 0)
            for pair, answer in (payload.get("answers") or {}).items():
                if pair in votes:
                    votes[pair].append(float(answer["noul"]))

    def mean(pair: str) -> float:
        return sum(votes[pair]) / len(votes[pair]) if votes[pair] else 0.0

    await vote(range(1, FIRST_ORDERS + 1), pairs)
    undecided = (
        [p for p in pairs if UNDECIDED[0] <= mean(p) <= UNDECIDED[1]] if second_round else []
    )
    await vote(range(FIRST_ORDERS + 1, ORDERS + 1), undecided)
    return {pair: mean(pair) for pair in pairs}, tokens, len(undecided)


async def jev_clusters(
    problem: str, readings: list[list[dict[str, Any]]], *, api_key: str
) -> list[list[tuple[int, int]]]:
    """The readings' milestones clustered by fact, as (reading, index) pairs covering every one."""
    refs = {member_id(r, i): (r, i) for r, ms in enumerate(readings) for i in range(len(ms))}

    def milestone(mid: str) -> dict[str, Any]:
        reading, index = refs[mid]
        return readings[reading][index]

    exact = [[member_id(r, i) for r, i in group] for group in exact_groups(readings)]
    cards = {group[0]: milestone_card(milestone(group[0])) for group in exact}
    categories = {group[0]: {milestone(m).get("category") for m in group} for group in exact}
    pairs = [
        f"{a}|{b}"
        for a, b in itertools.combinations(cards, 2)
        if any(frozenset({x, y}) not in NEVER_SAME for x in categories[a] for y in categories[b])
    ]
    started = time.monotonic()

    def state(order: int) -> dict[str, Any]:
        return {"task": problem[:1500], **shuffled(cards, order)}

    async with jev_http(api_key) as http:
        means, tokens, undecided = await pair_weights(http, state, pairs, pair_question)

    groups = join_groups(exact, means)
    logger.info(
        "milestone_alignment_jev milestones={} exact_groups={} pairs={} undecided={} tokens={} "
        "seconds={:.1f}",
        len(refs),
        len(exact),
        len(pairs),
        undecided,
        tokens,
        time.monotonic() - started,
    )
    return [[refs[mid] for mid in group] for group in groups]
