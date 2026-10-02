from __future__ import annotations

from ..command_search import ParseFailure
from .models import GitMeta, GitPlan, GitResult, GitState
from .parse import STDERR_ONLY_OUTPUT
from .render import HANDLERS
from .templates import DEFAULT_ABBREV, HISTORY_ONLY
from .views import Views

_SUMMARY_FLAGS = {"--stat", "--name-only", "--name-status", "--numstat", "--shortstat", "--quiet"}


def _prints_short_hashes(plan: GitPlan) -> bool:
    """Whether a command's output holds abbreviated hashes (a oneline log, a diff's `index`)."""
    if plan.sub in ("log", "stash"):
        return True
    if plan.sub in ("diff", "show"):
        return not plan.flags & _SUMMARY_FLAGS
    if plan.sub == "reset":
        return "--hard" in plan.flags
    return plan.sub == "rev-parse" and any(flag.startswith("--short") for flag in plan.flags)


def run_git(
    plan: GitPlan, overlay, read_base, listing: list[str], meta: GitMeta | None = None
) -> GitResult | ParseFailure:
    meta = meta or GitMeta()
    handler = HANDLERS.get(plan.sub)
    if handler is None:
        return ParseFailure("unsupported_form", f"git {plan.sub}")
    state = getattr(overlay, "git", None) or GitState()
    if plan.sub in HISTORY_ONLY:
        if state.history_dirty:
            return ParseFailure("unsupported_form", "commit history diverged")
    elif state.unknown:
        return ParseFailure("unsupported_form", "git state diverged")
    if state.modes and plan.sub in ("diff", "stash", "commit"):
        return ParseFailure("unsupported_form", "a tracked file's mode changed")
    abbrev = state.abbrev or meta.abbrev
    if abbrev is None and _prints_short_hashes(plan):
        return ParseFailure("unsupported_form", "the length of a short hash is not known")
    views = Views(overlay, state, read_base, listing, abbrev or DEFAULT_ABBREV)
    result = handler(plan, views, meta)
    if isinstance(result, GitResult) and plan.dropped_stderr and plan.sub in STDERR_ONLY_OUTPUT:
        return GitResult(output="", returncode=result.returncode, empty=True)
    if isinstance(result, GitResult) and plan.redirect:
        return GitResult(output="", returncode=result.returncode, empty=True)
    return result
