from __future__ import annotations

from dataclasses import replace

from ..command_search import ParseFailure
from .execute import run_git
from .models import GitMeta, GitPlan
from .parse import GIT_HEAD, git_stages, parse_git
from .templates import DEFAULT_ABBREV, GIT_EVIDENCE_HEADER

_RELAXABLE = {"--all", "--graph", "--decorate", "--no-merges", "-p", "--patch", "-i", "--follow"}
_RELAXABLE_PREFIX = ("--grep", "--since", "--until", "--author", "--committer", "-S", "-G")


def _drop_filters(args: list[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        index += 1
        if token in _RELAXABLE:
            continue
        if token.startswith(_RELAXABLE_PREFIX):
            if "=" not in token and index < len(args) and not args[index].startswith("-"):
                index += 1
            continue
        out.append(token)
    return out


def _relaxed(plan: GitPlan) -> GitPlan:
    return GitPlan(sub=plan.sub, args=_drop_filters(plan.args), raw=plan.raw)


def git_evidence(
    command: str, overlay, read_base, listing: list[str], meta: GitMeta | None = None
) -> str:
    if not GIT_HEAD.search(command or ""):
        return ""
    meta = meta or GitMeta()
    unsure_abbrev = meta.abbrev is None and not getattr(
        getattr(overlay, "git", None), "abbrev", None
    )
    if unsure_abbrev:
        # shown at git's shortest length; which length this repository prints is not known
        meta = replace(meta, abbrev=DEFAULT_ABBREV)
    fragments: list[str] = []
    for stage in git_stages(command):
        label = stage.text
        plan = parse_git(stage.text)
        partial = False
        if isinstance(plan, ParseFailure):
            if len(stage.pipeline) == 1:
                continue
            plan = parse_git(stage.command.text)
            if isinstance(plan, ParseFailure):
                continue
            partial = True
        result = run_git(plan, overlay, read_base, listing, meta)
        if isinstance(result, ParseFailure) or not result.exact:
            relaxed = _relaxed(plan)
            if relaxed.args == plan.args and not plan.pipeline:
                continue
            relaxed.evidence = True
            result = run_git(relaxed, overlay, read_base, listing, meta)
            partial = True
        if partial:
            label = (
                f"{label}   (computed WITHOUT this command's pipes and filters "
                "— apply them yourself)"
            )
        if isinstance(result, ParseFailure) or not result.exact or not result.output:
            continue
        if unsure_abbrev:
            label += (
                f"   (short hashes shown with {DEFAULT_ABBREV} characters: git prints more in a "
                "larger repository, so keep the prefixes and choose their length)"
            )
        fragments.append(f"$ {label}\n{result.output}")
    if not fragments:
        return ""
    return "\n" + GIT_EVIDENCE_HEADER + "\n" + "\n\n".join(fragments) + "\n"
