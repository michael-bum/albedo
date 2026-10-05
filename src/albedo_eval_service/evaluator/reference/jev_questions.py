"""Jev (TypeSafe System One) in the question stage, docs/SCORING.md.

J1 `question_clusters` groups the pool questions under each milestone by what they test. J1.a
`cross_milestone_duplicates` drops a finished question that tests what one under an earlier
milestone tests. J2 `reference_scores` asks, per reference run, whether the run earned each
question, and hands back None for a run Jev did not answer so the caller can judge it another way.
"""

from __future__ import annotations

import asyncio
import collections
import functools
import itertools
import time
from typing import Any

from loguru import logger

from .jev_align import JOIN_AT, JevUnavailable, ask, jev_http, join_groups, pair_weights, shuffled
from .vector_merge import exact_key, member_id

SAME_TEST_CRITERIA = {
    "true": "They test the same thing: a judge reading any candidate trajectory would give the "
    "same answer to both, even if they are worded differently, at a different length, or one "
    "names a detail the other omits.",
    "false": "They test different things: some candidate could earn one and not the other - for "
    "example one asks whether the candidate worked in an area and the other whether it "
    "established a specific fact there, or they ask about different facts, changes, checks or "
    "code sites.",
}
EARNED_CRITERIA = {
    "true": "The candidate's own CANDIDATE OUTPUT blocks, or the observations its own commands "
    "produced, show it: it read, ran or changed that code, drew that conclusion from content "
    "visible in the trajectory, made that change, or ran that check.",
    "false": "Nothing in the candidate's own blocks shows it: it is only named in the task or "
    "context, asserted in prose without the work behind it, or never reached.",
}


def same_test_question(pair: str) -> dict[str, Any]:
    a, b = pair.split("|")
    return {
        "type": "noul",
        "instructions": f"Do `{a}` and `{b}` test the same thing about a candidate?",
        "criteria": SAME_TEST_CRITERIA,
    }


def question_card(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "milestone": question.get("milestone"),
        "question": question.get("text"),
        "not_earned_by": question.get("example_bad") or question.get("unearned") or "",
    }


def _state(
    problem: str, milestones: list[dict[str, Any]], cards: dict[str, Any], order: int
) -> dict[str, Any]:
    statements = {str(m.get("id")): str(m.get("statement") or "") for m in milestones}
    return {"task": problem[:1500], "milestones": statements, **shuffled(cards, order)}


async def question_clusters(
    problem: str,
    milestones: list[dict[str, Any]],
    lists: list[list[dict[str, Any]]],
    *,
    api_key: str,
) -> list[list[tuple[int, int]]]:
    """The readings' questions grouped by what they test, as (reading, index) pairs covering all."""
    pool = {member_id(r, i): (r, i) for r, qs in enumerate(lists) for i in range(len(qs))}

    def question(mid: str) -> dict[str, Any]:
        reading, index = pool[mid]
        return lists[reading][index]

    exact: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
    for mid in pool:
        texts = exact[str(question(mid).get("milestone"))]
        texts.setdefault(exact_key(question(mid)), []).append(mid)
    starts = [list(texts.values()) for texts in exact.values()]
    cards = {group[0]: question_card(question(group[0])) for start in starts for group in start}
    pairs = [f"{a[0]}|{b[0]}" for start in starts for a, b in itertools.combinations(start, 2)]
    started = time.monotonic()
    async with jev_http(api_key) as http:
        means, tokens, _ = await pair_weights(
            http,
            functools.partial(_state, problem, milestones, cards),
            pairs,
            same_test_question,
            second_round=False,
        )
    groups = [group for start in starts for group in join_groups(start, means)]
    logger.info(
        "question_alignment_jev questions={} pairs={} groups={} tokens={} seconds={:.1f}",
        len(pool),
        len(pairs),
        len(groups),
        tokens,
        time.monotonic() - started,
    )
    return [[pool[mid] for mid in group] for group in groups]


async def cross_milestone_duplicates(
    problem: str,
    milestones: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    api_key: str,
) -> set[str]:
    """The ids of the questions that test what a question under an earlier milestone tests.

    `questions` are in milestone order, so the first of a pair is the one kept. Pairs are taken
    heaviest first, and a pair touching a question already dropped is skipped.
    """
    pairs = [
        f"{a['id']}|{b['id']}"
        for a, b in itertools.combinations(questions, 2)
        if a.get("milestone") != b.get("milestone")
    ]
    if not pairs:
        return set()
    cards = {str(q["id"]): question_card(q) for q in questions}
    started = time.monotonic()
    async with jev_http(api_key) as http:
        means, tokens, undecided = await pair_weights(
            http, functools.partial(_state, problem, milestones, cards), pairs, same_test_question
        )
    dropped: set[str] = set()
    for pair, weight in sorted(means.items(), key=lambda kv: -kv[1]):
        if weight < JOIN_AT:
            break
        keep, drop = pair.split("|")
        if keep not in dropped and drop not in dropped:
            dropped.add(drop)
    logger.info(
        "cross_milestone_dedup_jev questions={} pairs={} undecided={} dropped={} tokens={} "
        "seconds={:.1f}",
        len(questions),
        len(pairs),
        undecided,
        len(dropped),
        tokens,
        time.monotonic() - started,
    )
    return dropped


async def reference_scores(
    documents: list[str], questions: list[dict[str, Any]], *, api_key: str
) -> list[dict[str, float] | None]:
    """Per reference document, question id -> Jev's earned score; None where Jev did not answer."""
    asked = {
        str(q["id"]): {"type": "noul", "instructions": q["text"], "criteria": EARNED_CRITERIA}
        for q in questions
    }

    async def one(http, run: int, document: str) -> dict[str, float] | None:
        try:
            payload = await ask(http, {"trajectory": document}, asked)
        except JevUnavailable as exc:
            logger.warning("reference_prune_jev_failed run={} error={}", run, str(exc)[:200])
            return None
        return {qid: float(a["noul"]) for qid, a in (payload.get("answers") or {}).items()}

    async with jev_http(api_key) as http:
        return list(
            await asyncio.gather(*[one(http, run, d) for run, d in enumerate(documents, start=1)])
        )
