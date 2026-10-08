from __future__ import annotations

import re

from albedo_eval_service.shared.observation_format import observed_returncode

from ..command_search import ParseFailure, repo_path
from .execute import run_git
from .models import GitMeta, GitPlan, GitResult, GitState, StashEntry
from .parse import GIT_HEAD, parse_git, subcommand_of
from .patches import learn_from_observed_diff
from .templates import DEFAULT_ABBREV, HARNESS_SUBJECT, HISTORY_CHANGING, OPAQUE, READ_ONLY
from .views import Views, in_scope, moved_files, normalize_paths, removed_files

_BRANCH_LINE = re.compile(r"^On branch (\S+)$", re.M)
_DETACHED_LINE = re.compile(r"^Not currently on any branch\.$", re.M)
_DETACHED_AT_LINE = re.compile(r"^HEAD detached at ([0-9a-f]{4,40})$", re.M)
_INDEX_LINE = re.compile(r"^index ([0-9a-f]{4,40})\.\.[0-9a-f]{4,40}", re.M)
_WIP_LINE = re.compile(
    r"^Saved working directory and index state WIP on ([^:]+): ([0-9a-f]{4,40}) (.*)$", re.M
)
_HEAD_NOW_LINE = re.compile(r"^HEAD is now at ([0-9a-f]{4,40}) (.*)$", re.M)
_ONELINE_ENTRY = re.compile(r"^([0-9a-f]{7,12}) \S", re.M)
_HARNESS_HEAD = re.compile(r"^([0-9a-f]{7,12}) SWE-bench$", re.M)
_HEAD_ENTRY = re.compile(r"([0-9a-f]{7,12}) (?:\(HEAD[^)]*\) )?(.+)")
_WHOLE_LOG = re.compile(r"\bgit log(?: (?:--oneline|--no-decorate|-\d+|-n \d+))+\s*$")
_SHA = re.compile(r"[0-9a-f]{7,40}")
_BRANCH_RENAME = re.compile(r"\s-[mM]\b|--move")
_DIFF_COMMAND = re.compile(r"\bgit\s+(?:--no-pager\s+)?diff\b")


def learn_git_facts(state: GitState, observation: str, command: str = "") -> None:
    text = observation or ""
    if match := _HARNESS_HEAD.search(text):
        state.head_short = match.group(1)
        state.head_subject = HARNESS_SUBJECT
        state.abbrev = len(match.group(1))
    if match := _BRANCH_LINE.search(text):
        state.branch = match.group(1)
        state.detached, state.detached_at = False, None
    elif _DETACHED_LINE.search(text):
        state.detached, state.detached_at = True, None
    elif match := _DETACHED_AT_LINE.search(text):
        state.detached, state.detached_at = True, match.group(1)
        state.abbrev = len(match.group(1))
    if match := _WIP_LINE.search(text):
        branch, short, subject = match.groups()
        state.abbrev = len(short)
        state.head_short = short
        state.head_subject = subject
        if branch == "(no branch)":
            state.detached = True
        else:
            state.branch = branch
    elif match := _HEAD_NOW_LINE.search(text):
        state.abbrev = len(match.group(1))
        state.head_short = match.group(1)
        state.head_subject = match.group(2)
    elif match := _INDEX_LINE.search(text):
        state.abbrev = len(match.group(1))
    elif "--oneline" in (command or "") and (match := _ONELINE_ENTRY.search(text)):
        state.abbrev = len(match.group(1))
        first = text[match.start() :].split("\n", 1)[0]
        if _WHOLE_LOG.search(command) and (head := _HEAD_ENTRY.fullmatch(first)):
            # a log of the whole history starts at the commit HEAD is at
            state.head_short, state.head_subject = head.groups()


def _scope_paths(plan: GitPlan, views: Views) -> list[str]:
    tokens = [p for p in plan.paths if p not in ("HEAD", "--")]
    if not tokens or any(t in (".", "./", "-A", "*") for t in tokens):
        return []
    return normalize_paths(tokens)


def _stage(views: Views, paths: list[str], only_tracked: bool) -> list[str] | None:
    """`git add` of these paths: each change goes to the index, a deletion included; an
    untracked file only when .gitignore does not exclude it. None when what to stage is not
    known."""
    staged: list[str] = []
    for path in paths:
        tracked = views.tracked(path)
        if only_tracked and not tracked:
            continue
        present = views.present(path)
        if present is None:
            return None
        if not present:
            if views.in_index(path):
                views.state.index.pop(path, None)
                if views.overlay.in_base(path):
                    views.state.staged_deleted.add(path)
                staged.append(path)
            continue
        if not views.in_index(path):
            ignored = views.ignored(path)
            if ignored is None:
                return None
            if ignored and not views.overlay.in_base(path):
                continue
        views.state.index[path] = views.work(path)
        views.state.staged_deleted.discard(path)
        staged.append(path)
    return staged


def _restore(views: Views, paths: list[str], source: str) -> list[str]:
    for path in paths:
        from_head = source == "head" or path not in views.state.index
        text = views.head(path) if from_head else views.state.index[path]
        if from_head and text is None:
            views.overlay.drop(path)
        else:
            views.overlay.put(path, text)
    return list(paths)


def _apply_add(plan: GitPlan, views: Views) -> dict | None:
    if {"-n", "--dry-run"} & plan.flags:
        return {}
    if plan.flags - {"-A", "--all", "-u", "--update", "-f", "--force", "-v", "--verbose", "--"}:
        return None
    scope = _scope_paths(plan, views)
    only_tracked = bool({"-u", "--update"} & plan.flags)
    if views.unsure and not only_tracked:
        return None
    targets = [p for p in views.touched() if in_scope(p, scope)]
    staged = _stage(views, targets, only_tracked)
    return None if staged is None else {"staged": staged}


def _apply_checkout(plan: GitPlan, views: Views) -> dict | None:
    created = {"-b", "-B", "-c", "-C"} & set(plan.values)
    if created:
        return {"branch": plan.values[created.pop()]}
    operands = [p for p in plan.paths if p != "--"]
    source = "index"
    if operands and operands[0] == "HEAD":
        source = "head"
        operands = operands[1:]
    if not operands:
        return None
    if "--" not in plan.flags and source == "index":
        normalized = normalize_paths(operands)
        if not all(p in views.listing_set or p in views.overlay.created for p in normalized):
            return None
    scope = normalize_paths([p for p in operands if p not in (".", "./")])
    targets = [p for p in views.touched() if in_scope(p, scope) and views.tracked(p)]
    restored = _restore(views, targets, source)
    if source == "head":
        for path in targets:
            views.state.index.pop(path, None)
    return {"restored": restored, "from": source}


def _apply_restore(plan: GitPlan, views: Views) -> dict | None:
    source = plan.values.get("--source", "")
    if source and source != "HEAD":
        return None
    scope = _scope_paths(plan, views)
    targets = [p for p in views.touched() if in_scope(p, scope) and views.tracked(p)]
    effect: dict = {}
    if "--staged" in plan.flags:
        for path in targets:
            views.state.index.pop(path, None)
            views.state.staged_deleted.discard(path)
        effect["unstaged"] = list(targets)
    if "--staged" not in plan.flags or {"-W", "--worktree"} & plan.flags:
        effect["restored"] = _restore(views, targets, "head" if source == "HEAD" else "index")
    return effect


def _apply_reset(plan: GitPlan, views: Views) -> dict | None:
    revs = [p for p in plan.paths if p in ("HEAD", "HEAD~1", "ORIG_HEAD") or _SHA.fullmatch(p)]
    if any(rev != "HEAD" for rev in revs):
        return None
    if {"--merge", "--keep"} & plan.flags:
        return None
    scope = _scope_paths(plan, views)
    targets = [p for p in views.touched() if in_scope(p, scope)]
    if "--soft" in plan.flags:
        return {"soft": True}
    tracked = [p for p in targets if views.tracked(p)]
    unstaged = [p for p in targets if p in views.state.index or p in views.state.staged_deleted]
    for path in unstaged:
        views.state.index.pop(path, None)
        views.state.staged_deleted.discard(path)
    if "--hard" in plan.flags:
        _restore(views, tracked, "head")
        return {"reset": "hard", "restored": tracked}
    return {"unstaged": unstaged}


def _apply_stash(plan: GitPlan, views: Views) -> dict | None:
    action = plan.paths[0] if plan.paths else "push"
    state = views.state
    if action in ("list", "show"):
        return {"stash_depth": len(state.stash)}
    if action == "clear":
        state.stash.clear()
        return {"stash": "cleared"}
    if action in ("push", "save"):
        untracked = bool({"-u", "--include-untracked", "-a", "--all"} & plan.flags)
        ignored_too = bool({"-a", "--all"} & plan.flags)
        if views.unsure:
            return None
        entry = StashEntry()
        for path in views.touched():
            present = views.present(path)
            if present is None:
                return None
            if not views.tracked(path):
                ignored = views.ignored(path) if present else True
                if ignored is None and untracked:
                    return None
                if present and untracked and (ignored_too or not ignored):
                    entry.created[path] = views.work(path)
            elif not views.overlay.in_base(path):
                entry.staged_new[path] = views.work(path) if present else None
            elif not present:
                entry.deleted.add(path)
            elif views.work(path) != views.head(path) or path in views.state.index:
                entry.content[path] = views.work(path)
        targets = entry.paths()
        if not targets:
            return {"stashed": []}
        state.stash.append(entry)
        _restore(views, [p for p in targets if p in entry.content or p in entry.deleted], "head")
        for path in [*entry.staged_new, *entry.created]:
            views.overlay.drop(path)
        state.index.clear()
        state.staged_deleted.clear()
        return {"stashed": targets}
    if action in ("pop", "apply"):
        if not state.stash:
            return None
        entry = state.stash[-1] if action == "apply" else state.stash.pop()
        for path, text in [*entry.content.items(), *entry.created.items()]:
            views.overlay.put(path, text)
        for path, text in entry.staged_new.items():
            views.overlay.put(path, text)
            state.index[path] = text
        for path in entry.deleted:
            views.overlay.drop(path)
        return {"unstashed": entry.paths()}
    if action == "drop":
        if state.stash:
            state.stash.pop()
        return {"stash": "dropped"}
    return None


def _apply_mv(plan: GitPlan, views: Views) -> dict | None:
    moves = moved_files(plan, views)
    if not moves:
        return None if moves is None else {}
    state = views.state
    for source, target in moves:
        staged = views.staged(source)
        views.overlay.put(target, views.work(source))
        views.overlay.drop(source)
        state.index.pop(source, None)
        if views.overlay.in_base(source):
            state.staged_deleted.add(source)
        state.index[target] = staged
        state.staged_deleted.discard(target)
    return {"moved": [f"{source} -> {target}" for source, target in moves]}


def _apply_rm(plan: GitPlan, views: Views) -> dict | None:
    files = removed_files(plan, views)
    if files is None:
        return None
    for path in files:
        views.state.index.pop(path, None)
        if views.overlay.in_base(path):
            views.state.staged_deleted.add(path)
        if "--cached" not in plan.flags:
            views.overlay.drop(path)
    if "--cached" not in plan.flags:
        _drop_emptied(views.overlay, files)
    return {"removed": files}


def _drop_emptied(overlay, removed: list[str]) -> None:
    """git removes a directory its last file left (`overlay.drop` keeps it, as `rm` does)."""
    for path in removed:
        directory = path.rpartition("/")[0]
        while directory and not overlay.entries(directory) and not overlay.doubt(directory):
            overlay.dirs.discard(directory)
            directory = directory.rpartition("/")[0]


_MUTATORS = {
    "mv": _apply_mv,
    "rm": _apply_rm,
    "add": _apply_add,
    "checkout": _apply_checkout,
    "switch": _apply_checkout,
    "restore": _apply_restore,
    "reset": _apply_reset,
    "stash": _apply_stash,
}


# subcommands this does not model that change working-tree files, not only history
_CHANGES_FILES = {"am", "merge", "rebase", "revert", "cherry-pick", "pull"}


def learn_git(overlay, command: str, observation: str) -> None:
    """Adopt what a git command's observation shows about the repository: the branch, the short
    sha length and head commit, and the text of files an observed `git diff` shows."""
    if not GIT_HEAD.search(command or ""):
        return
    learn_git_facts(overlay.git, observation, command)
    if observed_returncode(observation) in (0, None) and _DIFF_COMMAND.search(command):
        learn_from_observed_diff(overlay, overlay.git, command, observation, overlay.read_base)


def apply_git_stage(
    overlay, stage: str, certain: bool = True, status: int | None = None, turn: int = 0
) -> None:
    """Apply one git stage of a command to the git model and the working tree: `certain` False
    when it may not have run, `status` its exit status when known. What this cannot follow
    leaves the git state unknown, and, for a command that rewrites files, the files too."""
    state = overlay.git
    plan = parse_git(stage)
    sub = plan.sub if isinstance(plan, GitPlan) else subcommand_of(stage)
    if isinstance(plan, GitPlan) and plan.redirect:
        _write_redirect(overlay, plan, certain)
    if sub in READ_ONLY:
        if sub == "branch" and _BRANCH_RENAME.search(stage):
            state.poison("branch renamed")
        return
    if sub == "add" and status not in (0, None):
        return  # a failing `git add` stages nothing
    if isinstance(plan, ParseFailure) or not certain or (status is not None and status != 0):
        reason = plan.reason if isinstance(plan, ParseFailure) else "uncertain_or_failed"
        state.poison(f"{reason}:{sub or '?'}")
        if sub not in ("add", "commit", "branch", "tag", "config"):
            overlay.unsure.add("")
        return
    if plan.sub in OPAQUE:
        state.poison(f"opaque:{plan.sub}", history=plan.sub in HISTORY_CHANGING)
        if plan.sub in _CHANGES_FILES:
            overlay.unsure.add("")
        return
    mutator = _MUTATORS.get(plan.sub)
    if mutator is None:
        return
    views = Views(overlay, state, overlay.read_base, overlay.listing(), DEFAULT_ABBREV)
    if state.unknown:
        _apply_unknown(plan, views)
        return
    effect = mutator(plan, views)
    if effect is None:
        state.poison(f"unmodelled:{plan.sub}", history=plan.sub in ("reset", "checkout", "switch"))
        overlay.unsure.add("")
        return
    if effect:
        state.record(turn, stage.strip(), effect)
        if "branch" in effect:
            # a branch made and switched to: HEAD is on it from here on
            state.branch, state.detached, state.detached_at = effect["branch"], False, None


def _apply_unknown(plan: GitPlan, views: Views) -> None:
    """A command run while the index is unknown: whether a file was staged cannot be told, so
    what it restores from HEAD is restored, and everything else it may have changed is left
    unknown rather than untouched."""
    overlay, flags = views.overlay, plan.flags
    touched = views.touched()
    if plan.sub == "stash":
        action = plan.paths[0] if plan.paths else "push"
        if action in ("pop", "apply") or (action in ("push", "save") and len(plan.paths) > 1):
            overlay.unsure.add("")
        elif action in ("push", "save"):
            _restore_unknown(views, touched, from_head=True)
    elif plan.sub == "reset":
        if "--hard" not in flags:
            return
        if any(
            p != "HEAD" and (p in ("HEAD~1", "ORIG_HEAD") or _SHA.fullmatch(p)) for p in plan.paths
        ):
            overlay.unsure.add("")
            return
        scope = _scope_paths(plan, views)
        _restore_unknown(views, [p for p in touched if in_scope(p, scope)], from_head=True)
    elif plan.sub in ("checkout", "switch"):
        if {"-b", "-B", "-c", "-C"} & set(plan.values):
            return
        operands = [p for p in plan.paths if p != "--"]
        from_head = operands[:1] == ["HEAD"]
        operands = operands[1:] if from_head else operands
        if not operands:
            overlay.unsure.add("")  # a branch switch rewrites files this cannot name
            return
        scope = normalize_paths([p for p in operands if p not in (".", "./")])
        _restore_unknown(views, [p for p in touched if in_scope(p, scope)], from_head)
    elif plan.sub == "restore":
        if "--staged" in flags and not {"-W", "--worktree"} & flags:
            return
        source = plan.values.get("--source", "")
        if source and source != "HEAD":
            overlay.unsure.add("")
            return
        scope = _scope_paths(plan, views)
        _restore_unknown(views, [p for p in touched if in_scope(p, scope)], source == "HEAD")
    elif plan.sub in ("rm", "mv"):
        if plan.sub == "rm" and "--cached" in flags:
            return
        paths = [repo_path(t, overlay.cwd, overlay.root) for t in plan.paths if t != "--"]
        if None in paths or (plan.sub == "mv" and len(paths) != 2):
            overlay.unsure.add("")
            return
        if plan.sub == "mv" and overlay.kind(paths[1]) == "directory":
            paths[1] = f"{paths[1]}/{paths[0].rsplit('/', 1)[-1]}"
            overlay.put(paths[1], None, certain=False)
        elif plan.sub == "mv":
            overlay.put(paths[1], None, certain=False)
        for path in paths[:1] if plan.sub == "mv" else paths:
            overlay.drop(path, certain=False)


def _restore_unknown(views: Views, paths: list[str], from_head: bool) -> None:
    """Restore `paths` while the index is unknown: a tracked file comes back from HEAD when the
    command restores HEAD and HEAD is still the checkout's commit, and otherwise its text is
    unknown; a new file may have been staged, so what became of it is unknown."""
    overlay = views.overlay
    for path in paths:
        if overlay.in_base(path) and from_head and not views.state.history_dirty:
            _restore(views, [path], "head")
        elif overlay.in_base(path) or not from_head:
            overlay.put(path, None, certain=False)
        else:
            overlay.drop(path, certain=False)


def _write_redirect(overlay, plan: GitPlan, certain: bool) -> None:
    """The text of the file a git command's output is redirected to, when git's output is
    known. The shell makes the file before git runs, so git sees it (a `git status` lists it)."""
    target = repo_path(plan.redirect, overlay.cwd, overlay.root)
    if target is None or not certain or overlay.git.unknown:
        return
    capture = GitPlan(sub=plan.sub, args=plan.args, pipeline=plan.pipeline, raw=plan.raw)
    captured = run_git(capture, overlay, overlay.read_base, overlay.listing(), GitMeta())
    if isinstance(captured, GitResult) and captured.exact:
        overlay.put(target, captured.output + "\n" if captured.output else "")
