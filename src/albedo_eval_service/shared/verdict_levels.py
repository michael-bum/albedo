from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .json_extract import extract_json

LETTERS = "ABCDEFGHIJKLMNOPQRST"
ANCHORS: dict[str, float] = {"A": 0.0, "E": 0.15, "J": 0.35, "O": 0.70, "T": 1.0}
SCORING_MODE = "graded_20"
PRUNE_EARNED_MIN = 0.5
NO_CREDIT = "A"
UNSETTLED = "J"
FULL_CREDIT = "T"
TOP_LOGPROBS = len(LETTERS)
DISTRIBUTION_MIN_P = 0.001
_LOGPROB_FLOOR = -1e30


def _interpolate() -> dict[str, float]:
    anchors = sorted((LETTERS.index(letter), value) for letter, value in ANCHORS.items())
    phi: dict[str, float] = {}
    for (i0, v0), (i1, v1) in zip(anchors, anchors[1:]):
        for i in range(i0, i1 + 1):
            phi[LETTERS[i]] = round(v0 + (v1 - v0) * (i - i0) / (i1 - i0), 6)
    return phi


PHI: dict[str, float] = _interpolate()

_VERDICT_VALUE_RE = re.compile(r'"verdict"\s*:\s*"\s*([A-Ta-t])\s*"')


@dataclass(frozen=True)
class LogprobReading:
    scores: dict[str, float] = field(default_factory=dict)
    distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _letter(token: Any) -> str | None:
    text = str(token or "").strip().strip('"').strip()
    return text.upper() if len(text) == 1 and text.upper() in PHI else None


def _ordered_question_ids(raw: str) -> list[str]:
    obj = extract_json(raw, prefer_keys=("answers",))
    items = obj.get("answers") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return []
    return [
        str(item.get("asked") or item.get("id") or "").strip()
        for item in items
        if isinstance(item, dict)
    ]


def read_verdict_logprobs(raw: str, entries: list[dict[str, Any]] | None) -> LogprobReading:
    if not entries:
        return LogprobReading(error="no logprobs in response")
    question_ids = _ordered_question_ids(raw)
    matches = list(_VERDICT_VALUE_RE.finditer(raw or ""))
    if not question_ids or len(matches) != len(question_ids):
        return LogprobReading(
            error=f"verdict count mismatch: {len(matches)} verdicts for {len(question_ids)} answers"
        )
    spans: list[tuple[int, int, dict[str, Any]]] = []
    tokens: list[str] = []
    position = 0
    for entry in entries:
        token = str(entry.get("token") or "")
        spans.append((position, position + len(token), entry))
        tokens.append(token)
        position += len(token)
    # Verdicts are located in the token text, not the content: a provider's stream can drop a
    # few characters elsewhere (seen on Alibaba), which would shift every content offset. The
    # stream must still carry the same verdicts in the same order as the content. A terminator
    # past the end of the content (Ambient sends `<|endoftext|>`) carries no verdict.
    spanned_matches = list(_VERDICT_VALUE_RE.finditer("".join(tokens)))
    written_letters = [m.group(1).upper() for m in matches]
    streamed_letters = [m.group(1).upper() for m in spanned_matches]
    if streamed_letters != written_letters:
        return LogprobReading(
            error=f"logprob tokens carry verdicts {''.join(streamed_letters) or 'none'}"
            f" but the content has {''.join(written_letters)}"
        )
    scores: dict[str, float] = {}
    distributions: dict[str, dict[str, float]] = {}
    for match, qid in zip(spanned_matches, question_ids):
        offset = match.start(1)
        entry = next((e for start, end, e in spans if start <= offset < end), None)
        if entry is None:
            return LogprobReading(error=f"no logprob token at verdict of {qid}")
        sampled = _letter(entry.get("token"))
        written = match.group(1).upper()
        if sampled != written:
            token = str(entry.get("token"))
            return LogprobReading(
                error=f"logprob token {token!r} is not the verdict {written} of {qid}"
            )
        live: dict[str, float] = {}
        for candidate in entry.get("top_logprobs") or []:
            letter = _letter(candidate.get("token"))
            logprob = candidate.get("logprob")
            if letter is None or logprob is None or float(logprob) <= _LOGPROB_FLOOR:
                continue
            live[letter] = max(float(logprob), live.get(letter, -math.inf))
        if written not in live:
            return LogprobReading(error=f"verdict {written} of {qid} missing from top_logprobs")
        mass = sum(math.exp(v) for v in live.values())
        probabilities = {letter: math.exp(v) / mass for letter, v in live.items()}
        scores[qid] = round(sum(p * PHI[letter] for letter, p in probabilities.items()), 6)
        distributions[qid] = {
            letter: round(p, 6)
            for letter, p in sorted(probabilities.items(), key=lambda kv: -kv[1])
            if p >= DISTRIBUTION_MIN_P
        }
    return LogprobReading(scores=scores, distributions=distributions)
