from __future__ import annotations

import re

from ..command_search import ParseFailure, parse_pipe_stage
from ..shell import Stage, Unsupported, parse_command
from .models import GitPlan
from .templates import GLOBAL_BOOL_FLAGS, GLOBAL_VALUE_FLAGS

STDERR_ONLY_OUTPUT = {"checkout"}


def parse_git(cmd: str) -> GitPlan | ParseFailure:
    """A git command and the pipe stages after it. Its errors may be merged into the output
    (`2>&1`) or dropped (`2>/dev/null`), and its output written to one file."""
    parsed = parse_command(cmd or "")
    if isinstance(parsed, Unsupported) or len(parsed.stages) != 1:
        return ParseFailure("unsupported_shell", "control operators")
    head, *rest = parsed.stages[0].pipeline
    words = [word.text for word in head.words]
    if not words or words[0] != "git":
        return ParseFailure("not_git", words[0] if words else "")
    if head.expands or any(command.expands or command.redirects for command in rest):
        return ParseFailure("unsupported_shell", "expansion or redirect in the pipe")
    dropped_stderr, redirect = False, None
    for target in head.redirects:
        if target.fd == 2 and target.target.text == "/dev/null":
            dropped_stderr = True
        elif target.fd == 2 and target.mode == "duplicate":
            continue
        elif target.fd == 1 and target.mode == "write" and redirect is None and not rest:
            redirect = target.target.text
        else:
            return ParseFailure("unsupported_shell", "output redirected to a file")
    index = 1
    while index < len(words) and words[index].startswith("-"):
        if words[index] in GLOBAL_VALUE_FLAGS:
            index += 2
        elif words[index] in GLOBAL_BOOL_FLAGS:
            index += 1
        else:
            return ParseFailure("unknown_flag", words[index])
    if index >= len(words):
        return ParseFailure("unsupported_form", "git without subcommand")
    pipeline = []
    for command in rest:
        stage = parse_pipe_stage([word.text for word in command.words])
        if isinstance(stage, ParseFailure):
            return stage
        pipeline.append(stage)
    return GitPlan(
        sub=words[index],
        args=words[index + 1 :],
        pipeline=pipeline,
        raw=cmd,
        dropped_stderr=dropped_stderr,
        redirect=redirect,
    )


def git_stages(command: str) -> list[Stage]:
    """The stages of a command that run git, or none when the shell parser refuses it."""
    parsed = parse_command(command or "")
    if isinstance(parsed, Unsupported):
        return []
    return [stage for stage in parsed.stages if stage.command.name == "git"]


GIT_HEAD = re.compile(r"(?:^|[;&|(]\s*|&&\s*)(?:\w+=\S+\s+)*(?:sudo\s+|env\s+)?git\s")


def is_git_command(command: str) -> bool:
    return bool(GIT_HEAD.search(command or ""))


_GIT_SUB = re.compile(
    r"\bgit\s+(?:(?:-C|-c|--git-dir|--work-tree)\s+\S+\s+|--no-pager\s+)*([a-z][a-z-]*)"
)


SHORT_STATUS = re.compile(r"\bgit\s+status\b[^|&]*\s(?:-s\b|--short\b|--porcelain\b)")
STASH_RESTORE = re.compile(r"\bgit\s+stash\s+(?:pop|apply)\b")


def subcommand_of(stage: str) -> str:
    match = _GIT_SUB.search(stage or "")
    return match.group(1) if match else ""
