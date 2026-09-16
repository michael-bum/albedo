from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from albedo_eval_service.shared.json_extract import extract_json
from albedo_eval_service.shared.loop_check import (
    DUP_CMD_THRESHOLD,
    MAX_RUN_THRESHOLD,
    commands_of,
)
from sanity_service.judge_panel import make_client, query_panel
from sanity_service.rubricisity import (
    TAIL_JUDGE_QUESTIONS,
    TAIL_JUDGE_SYSTEM,
    TAIL_JUDGE_USER,
)

TAIL_CUTOFF = 16
TAIL_JUDGE_FAIL_ZEROS = 3
TAIL_JUDGE_MIN_FAILED_SAMPLES = 2


Paired = tuple[str, str | None, str | None]  # assistant turn, its observation, requester message


def loop_stats(assistant_turns: list[str], observations: list[str | None] | None = None) -> dict:
    cmds = commands_of(assistant_turns, observations)
    max_run = run = 1
    for prev, cur in zip(cmds, cmds[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    return {
        "n_cmds": len(cmds),
        "dup_cmd_ratio": 1 - len(set(cmds)) / len(cmds) if cmds else 0.0,
        "max_cmd_run": max_run if cmds else 0,
    }


def looping_reason(assistant_turns: list[str], observations: list[str | None] | None = None) -> str:
    stats = loop_stats(assistant_turns, observations)
    if stats["dup_cmd_ratio"] >= DUP_CMD_THRESHOLD:
        return f"looping: duplicate command ratio {stats['dup_cmd_ratio']:.2f}"
    if stats["max_cmd_run"] >= MAX_RUN_THRESHOLD:
        return f"looping: same command repeated {stats['max_cmd_run']}x consecutively"
    return ""


_TASK_CHAR_CAP = 4000
_TURN_CHAR_CAP = 2500
_RESULT_CHAR_CAP = 1500


@dataclass
class TailVerdict:
    sample_id: str
    checked: bool
    passed: bool
    reason: str = ""
    answers: dict[str, int] | None = None


def paired_turns(turns: list[dict]) -> list[Paired]:
    """Each scored assistant turn with the observation it got and any requester message injected
    before the next assistant turn — the judge's questions presuppose both."""
    paired: list[list] = []
    current: list | None = None
    for turn in turns:
        content = str(turn.get("content") or "")
        if turn.get("role") == "assistant":
            current = [content, None, None] if turn.get("score_target") else None
            if current:
                paired.append(current)
        elif current:
            slot = 1 if turn.get("environment_observation") else 2
            current[slot] = content if current[slot] is None else f"{current[slot]}\n\n{content}"
    return [tuple(p) for p in paired]


def _tail_user(task: str, paired: list[Paired], *, submit_clause: str = "") -> str:
    blocks = []
    for n, (turn, result, request) in enumerate(paired[TAIL_CUTOFF:], start=TAIL_CUTOFF + 1):
        block = f"LATE TURN {n}:\n{turn[:_TURN_CHAR_CAP]}"
        if result is not None:
            block += f"\nRESULT {n} (environment output):\n{result[:_RESULT_CHAR_CAP] or '(empty)'}"
        if request is not None:
            block += f"\nREQUESTER {n} (from the task requester):\n{request[:_TURN_CHAR_CAP]}"
        blocks.append(block)
    questions = "\n".join(f"{qid}: {text}" for qid, text in TAIL_JUDGE_QUESTIONS)
    return TAIL_JUDGE_USER.format(
        task=(task or "")[:_TASK_CHAR_CAP],
        start=TAIL_CUTOFF + 1,
        total=len(paired),
        tail="\n\n".join(blocks),
        questions=questions,
        submit=submit_clause or "(not stated)",
    )


def _parse_tail_answers(raw: str) -> dict[str, int] | None:
    obj = extract_json(raw, prefer_keys=("answers",))
    items = obj.get("answers") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return None
    answers: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict) and str(item.get("id", "")) in dict(TAIL_JUDGE_QUESTIONS):
            try:
                answers[str(item["id"])] = 1 if int(item.get("answer", 0)) == 1 else 0
            except (TypeError, ValueError):
                continue
    return answers if len(answers) == len(TAIL_JUDGE_QUESTIONS) else None


async def judge_tail(
    client, task: str, paired: list[Paired], *, sample_id: str, submit_clause: str = ""
) -> TailVerdict:
    results = await query_panel(
        client,
        TAIL_JUDGE_SYSTEM.format(cutoff=TAIL_CUTOFF),
        _tail_user(task, paired, submit_clause=submit_clause),
        temperature=0.0,
    )
    usable = next((r for r in results if not r.error and r.raw.strip()), None)
    if usable is None:
        return TailVerdict(sample_id, checked=True, passed=True, reason="tail judge unavailable")
    answers = _parse_tail_answers(usable.raw)
    if answers is None:
        return TailVerdict(sample_id, checked=True, passed=True, reason="tail judge unparsable")
    zeros = [qid for qid, bit in answers.items() if bit == 0]
    if len(zeros) >= TAIL_JUDGE_FAIL_ZEROS:
        return TailVerdict(
            sample_id,
            checked=True,
            passed=False,
            reason=(
                f"degenerate tail past turn {TAIL_CUTOFF}: the late turns stopped doing "
                "purposeful work (repeated or aimless actions, filler or broken text, "
                "reasoning that does not drive the commands); every turn this deep must "
                "still be a concrete, new step that follows from the observations"
            ),
            answers=answers,
        )
    return TailVerdict(sample_id, checked=True, passed=True, answers=answers)


async def run_tail_check(states, *, client=None) -> list[TailVerdict]:
    verdicts: list[TailVerdict] = []
    needs_judge = []
    for state in states:
        if state.error or state.heuristic_reason:
            continue
        paired = paired_turns(state.turns)
        scored = [turn for turn, _, _ in paired]
        reason = looping_reason(scored, [result for _, result, _ in paired])
        if reason:
            state.heuristic_reason = f"tail_check: {reason}"
            verdicts.append(TailVerdict(state.sample_id, checked=True, passed=False, reason=reason))
            logger.warning("[tail-check] {} {}", state.sample_id, reason)
            continue
        if len(scored) <= TAIL_CUTOFF:
            verdicts.append(
                TailVerdict(
                    state.sample_id,
                    checked=False,
                    passed=True,
                    reason=f"finished in {len(scored)} turns",
                )
            )
            continue
        needs_judge.append((state, paired))

    if not needs_judge:
        return verdicts

    own_client = client is None
    if own_client:
        client = make_client()
    judge_failed: list = []
    try:
        for state, paired in needs_judge:
            verdict = await judge_tail(
                client,
                state.prompt,
                paired,
                sample_id=state.sample_id,
                submit_clause=state.submit_clause,
            )
            verdicts.append(verdict)
            if not verdict.passed:
                judge_failed.append((state, verdict))
                logger.warning("[tail-check] {} {}", state.sample_id, verdict.reason)
    finally:
        if own_client:
            await client.aclose()
    if len(judge_failed) >= TAIL_JUDGE_MIN_FAILED_SAMPLES:
        for state, verdict in judge_failed:
            state.heuristic_reason = f"tail_check: {verdict.reason}"
    elif judge_failed:
        logger.info(
            "[tail-check] {} judge-failed tail(s) below the {}-sample threshold, not failing",
            len(judge_failed),
            TAIL_JUDGE_MIN_FAILED_SAMPLES,
        )
    return verdicts
