from __future__ import annotations

import asyncio

from albedo_eval_service.shared.observation_memo import ObservationMemo


def _produce_counter(answers: list[str]):
    calls = {"n": 0}

    async def produce() -> str:
        calls["n"] += 1
        return answers[min(calls["n"] - 1, len(answers) - 1)]

    return produce, calls


def _observe(memo, key, produce, store=None) -> str:
    return asyncio.run(memo.observe(key, produce, store=store))


def test_a_usable_answer_is_served_verbatim_on_repeat():
    memo = ObservationMemo()
    produce, calls = _produce_counter(["file contents", "different"])
    first = _observe(memo, "k", produce, store=lambda obs: True)
    second = _observe(memo, "k", produce, store=lambda obs: True)
    assert first == second == "file contents"
    assert calls["n"] == 1


def test_an_answer_the_predicate_rejects_is_served_once_and_asked_again():
    memo = ObservationMemo()
    leak = "Let me check the file.\n```bash\ncat a.py\n```"
    produce, calls = _produce_counter([leak, leak, "real content"])
    usable = lambda obs: "```" not in obs  # noqa: E731
    assert _observe(memo, "k", produce, store=usable) == leak
    assert _observe(memo, "k", produce, store=usable) == leak
    assert _observe(memo, "k", produce, store=usable) == "real content"
    assert _observe(memo, "k", produce, store=usable) == "real content"
    assert calls["n"] == 3


def test_without_a_predicate_the_memo_behaves_as_before():
    memo = ObservationMemo()
    produce, calls = _produce_counter(["anything"])
    for _ in range(4):
        assert _observe(memo, "k", produce) == "anything"
    assert calls["n"] == 1
