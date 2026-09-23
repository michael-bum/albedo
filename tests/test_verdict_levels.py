from __future__ import annotations

import json
import math

from albedo_eval_service.shared.verdict_levels import (
    ANCHORS,
    LETTERS,
    PHI,
    PRUNE_EARNED_MIN,
    SCORING_MODE,
    TOP_LOGPROBS,
    read_verdict_logprobs,
)


def _raw(verdicts: dict[str, str]) -> str:
    return json.dumps(
        {"answers": [{"asked": qid, "reason": "e", "verdict": v} for qid, v in verdicts.items()]}
    )


def logprob_entries(raw: str, top: dict[str, dict[str, float]] | None = None) -> list[dict]:
    """One token per character; a verdict letter's top_logprobs come from `top[letter]`."""
    top = top or {}
    entries = []
    for index, char in enumerate(raw):
        candidates = None
        if char in PHI and raw[index - 1] == '"' and raw[index + 1] == '"':
            spread = top.get(char) or {char: 0.0}
            candidates = [{"token": t, "logprob": lp} for t, lp in spread.items()]
        entries.append({"token": char, "logprob": 0.0, "top_logprobs": candidates or []})
    return entries


def test_ladder_is_linear_between_the_anchors():
    assert len(LETTERS) == 20 and TOP_LOGPROBS == 20
    assert SCORING_MODE == "graded_20"
    assert PRUNE_EARNED_MIN == 0.5
    for letter, value in ANCHORS.items():
        assert PHI[letter] == value
    assert PHI["B"] == 0.0375 and PHI["D"] == 0.1125
    assert PHI["K"] == 0.42 and PHI["N"] == 0.63
    assert PHI["P"] == 0.76 and PHI["S"] == 0.94
    values = [PHI[letter] for letter in LETTERS]
    assert values == sorted(values)


def test_reading_takes_the_expectation_over_the_live_letters():
    raw = _raw({"q_01": "T", "q_02": "E"})
    entries = logprob_entries(
        raw,
        top={"T": {"T": math.log(0.5), "O": math.log(0.5)}, "E": {"E": 0.0, "x": -1.0}},
    )
    reading = read_verdict_logprobs(raw, entries)
    assert reading.ok
    assert reading.scores == {"q_01": 0.85, "q_02": 0.15}
    assert reading.distributions == {"q_01": {"T": 0.5, "O": 0.5}, "q_02": {"E": 1.0}}


def test_reading_strips_quotes_and_ignores_the_sentinel_and_non_letters():
    raw = _raw({"q_01": "J"})
    entries = logprob_entries(
        raw,
        top={
            "J": {
                '"J"': math.log(0.75),
                " O": math.log(0.25),
                "yes": -0.1,
                "K": -3.4028234663852886e38,
            }
        },
    )
    reading = read_verdict_logprobs(raw, entries)
    assert reading.ok
    assert reading.scores == {"q_01": round(0.75 * 0.35 + 0.25 * 0.70, 6)}
    assert set(reading.distributions["q_01"]) == {"J", "O"}


def test_reading_rejects_misaligned_logprobs():
    raw = _raw({"q_01": "T"})
    shifted = logprob_entries(raw)
    shifted.insert(0, {"token": "", "logprob": 0.0, "top_logprobs": []})
    shifted[1:] = shifted[2:] + [{"token": " ", "logprob": 0.0, "top_logprobs": []}]
    reading = read_verdict_logprobs(raw, shifted)
    assert not reading.ok
    assert "logprob token" in reading.error


def test_reading_rejects_missing_or_short_logprobs():
    raw = _raw({"q_01": "T", "q_02": "E"})
    assert read_verdict_logprobs(raw, None).error == "no logprobs in response"
    assert read_verdict_logprobs(raw, []).error == "no logprobs in response"
    short = logprob_entries(raw)[:-3]
    assert "do not reproduce the content" in read_verdict_logprobs(raw, short).error
    no_top = logprob_entries(raw)
    for entry in no_top:
        entry["top_logprobs"] = []
    assert "missing from top_logprobs" in read_verdict_logprobs(raw, no_top).error


def test_reading_ignores_a_terminator_token_past_the_content():
    """Ambient appends `<|endoftext|>` after the JSON; it carries no verdict and must not matter."""
    raw = _raw({"q_01": "T"})
    entries = logprob_entries(raw) + [
        {"token": "<|endoftext|>", "logprob": -0.01, "top_logprobs": []}
    ]
    reading = read_verdict_logprobs(raw, entries)
    assert reading.ok
    assert reading.scores == {"q_01": 1.0}


def test_reading_needs_one_verdict_per_answer():
    raw = _raw({"q_01": "T"}) + ' "verdict":"A"'
    reading = read_verdict_logprobs(raw, logprob_entries(raw))
    assert "verdict count mismatch" in reading.error
