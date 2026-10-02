"""A model of git in the sandbox checkout: its answers to git commands (`run_git`, `parse_git`)
and what git commands a transcript ran do to the index and the working tree (`apply_git_stage`,
`learn_git`)."""

from __future__ import annotations

from .chain import git_evidence
from .diffs import blob_hash
from .execute import run_git
from .models import GitMeta, GitResult, GitState
from .notes import explain_git, ledger_block
from .parse import is_git_command, parse_git
from .patches import apply_hunks, unified_diff
from .session import apply_git_stage, learn_git
from .templates import DEFAULT_ABBREV, DEFAULT_BRANCH

__all__ = [
    "DEFAULT_ABBREV",
    "DEFAULT_BRANCH",
    "GitMeta",
    "GitResult",
    "GitState",
    "apply_git_stage",
    "apply_hunks",
    "blob_hash",
    "explain_git",
    "git_evidence",
    "is_git_command",
    "learn_git",
    "ledger_block",
    "parse_git",
    "run_git",
    "unified_diff",
]
