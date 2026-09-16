from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .observation_format import (
    action_blocks,
    echoed_command,
    leaked_turn,
    prints_nothing_on_success,
    silent_observation,
)
from .submit_protocol import asked_submit, first_bash_command

DUP_CMD_THRESHOLD = 0.61
MAX_RUN_THRESHOLD = 5

MAX_LISTED_COMMANDS = 5
MAX_COMMAND_CHARS = 120

_CANDIDATE_RE = re.compile(
    r"^CANDIDATE OUTPUT(?: \d+)?:\n------\n(.*?)\n------"
    r"(?=\n+(?:CANDIDATE OUTPUT|ENVIRONMENT OBSERVATION|CONTEXT )[^\n]*:\n------|\Z)",
    re.MULTILINE | re.DOTALL,
)
_BLOCK_RE = re.compile(
    r"^(CANDIDATE OUTPUT(?: \d+)?|ENVIRONMENT OBSERVATION[^\n]*|CONTEXT [^\n]*):"
    r"\n------\n(.*?)\n------"
    r"(?=\n+(?:CANDIDATE OUTPUT|ENVIRONMENT OBSERVATION|CONTEXT )[^\n]*:\n------|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class LoopingCommand:
    command: str
    count: int
    longest_run: int


@dataclass(frozen=True)
class LoopVerdict:
    looped: bool
    reasons: tuple[str, ...]
    commands: tuple[LoopingCommand, ...]
    n_cmds: int
    dup_cmd_ratio: float
    max_cmd_run: int


def candidate_turns(document: str) -> list[str]:
    turns = [match.group(1).rstrip() for match in _CANDIDATE_RE.finditer(document or "")]
    if turns:
        return turns
    return [document] if document else []


def candidate_turns_with_observations(document: str) -> tuple[list[str], list[str | None]]:
    """The scored turns and, index-aligned, the observation each received (None when none did)."""
    turns: list[str] = []
    observations: list[str | None] = []
    for match in _BLOCK_RE.finditer(document or ""):
        label, body = match.group(1), match.group(2).rstrip()
        if label.startswith("CANDIDATE OUTPUT"):
            turns.append(body)
            observations.append(None)
        elif label.startswith("ENVIRONMENT OBSERVATION") and turns and observations[-1] is None:
            observations[-1] = body
    if turns:
        return turns, observations
    return ([document], [None]) if document else ([], [])


def unanswered(command: str, observation: str | None) -> bool:
    """Nothing to act on: an echo, a leaked turn, or silence for a command that must print."""
    if observation is None:
        return False
    if echoed_command(command, observation) or leaked_turn(observation):
        return True
    return silent_observation(observation) and not prints_nothing_on_success(command)


def commands_of(turns: list[str], observations: list[str | None] | None = None) -> list[str]:
    """With `observations` (index-aligned), an exact re-issue of the previous command after an
    unanswered observation is not counted: re-asking a shell that said nothing is rational, and
    dropping only duplicates keeps both statistics monotone."""
    cmds: list[str] = []
    previous: tuple[list[str], str, str | None] = ([], "", None)
    for index, turn in enumerate(turns):
        blocks = [c for c in action_blocks(turn) if not asked_submit(c)]
        seen = observations and index < len(observations)
        observation = observations[index] if seen else None
        if not (blocks and blocks == previous[0] and unanswered(previous[1], previous[2])):
            cmds += blocks
        previous = (blocks, first_bash_command(turn), observation)
    return cmds


def loop_stats(turns: list[str], observations: list[str | None] | None = None) -> dict:
    cmds = commands_of(turns, observations)
    max_run = run = 1
    for prev, cur in zip(cmds, cmds[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    return {
        "n_cmds": len(cmds),
        "dup_cmd_ratio": 1 - len(set(cmds)) / len(cmds) if cmds else 0.0,
        "max_cmd_run": max_run if cmds else 0,
    }


def _longest_runs(cmds: list[str]) -> dict[str, int]:
    longest: dict[str, int] = {}
    run = 0
    for index, cmd in enumerate(cmds):
        run = run + 1 if index and cmd == cmds[index - 1] else 1
        if run > longest.get(cmd, 0):
            longest[cmd] = run
    return longest


def loop_verdict(turns: list[str], observations: list[str | None] | None = None) -> LoopVerdict:
    cmds = commands_of(turns, observations)
    stats = loop_stats(turns, observations)
    counts = Counter(cmds)
    longest = _longest_runs(cmds)

    reasons: list[str] = []
    if stats["dup_cmd_ratio"] >= DUP_CMD_THRESHOLD:
        reasons.append(f"duplicate command ratio {stats['dup_cmd_ratio']:.2f}")
    if stats["max_cmd_run"] >= MAX_RUN_THRESHOLD:
        reasons.append(f"same command repeated {stats['max_cmd_run']}x consecutively")

    looping = [
        LoopingCommand(command=cmd, count=count, longest_run=longest.get(cmd, 1))
        for cmd, count in counts.items()
        if count >= 2 or longest.get(cmd, 1) >= MAX_RUN_THRESHOLD
    ]
    looping.sort(key=lambda item: (-item.longest_run, -item.count, item.command))

    return LoopVerdict(
        looped=bool(reasons),
        reasons=tuple(reasons),
        commands=tuple(looping) if reasons else (),
        **stats,
    )


def loop_verdict_for_document(document: str) -> LoopVerdict:
    turns, observations = candidate_turns_with_observations(document)
    return loop_verdict(turns, observations)


def _render_command(entry: LoopingCommand) -> str:
    command = entry.command
    if len(command) > MAX_COMMAND_CHARS:
        command = command[: MAX_COMMAND_CHARS - 1] + "…"
    if entry.longest_run >= 2:
        return f"`{command}` {entry.count}x ({entry.longest_run} consecutive)"
    return f"`{command}` {entry.count}x"


def loop_explanation(verdict: LoopVerdict) -> str:
    listed = verdict.commands[:MAX_LISTED_COMMANDS]
    parts = [
        "Trajectory is looped, so every question is scored 0 without judging.",
        "; ".join(verdict.reasons) + ".",
    ]
    if listed:
        rendered = "; ".join(_render_command(entry) for entry in listed)
        hidden = len(verdict.commands) - len(listed)
        suffix = f"; and {hidden} more" if hidden > 0 else ""
        parts.append(f"Looping commands: {rendered}{suffix}.")
    return " ".join(parts)
