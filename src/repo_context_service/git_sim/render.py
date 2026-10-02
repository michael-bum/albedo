from __future__ import annotations

import re

from ..command_search import ParseFailure, apply_pipeline, repo_path
from .diffs import diff_head, diff_pairs
from .models import GitMeta, GitPlan, GitResult
from .patches import commit_header, filter_diff, parse_patch, reabbrev, retarget_funcnames
from .templates import (
    BRANCH_HEADER,
    CLEAN_TRAILER,
    DEFAULT_BRANCH,
    DETACHED_HEADER,
    ENTRY_LABEL_WIDTH,
    EVIDENCE_LOG_LIMIT,
    HARD_RESET_LINE,
    HARNESS_SUBJECT,
    RESET_HEADER,
    STAGED_HEADER,
    STAGED_HINT,
    STASH_EMPTY_LINE,
    STASH_MISSING_LINE,
    STASH_SAVED_LINE,
    UNSTAGED_HEADER,
    UNSTAGED_HINTS,
    UNSTAGED_TRAILER,
    UNTRACKED_HEADER,
    UNTRACKED_HINT,
    UNTRACKED_ONLY_TRAILER,
    UPDATED_PATHS_LINE,
)
from .views import (
    Views,
    differs_from,
    dirty_paths,
    in_scope,
    moved_files,
    normalize_paths,
    removed_files,
    resolve_head,
)


def _entry(label: str, path: str) -> str:
    return "\t" + (label + ":").ljust(ENTRY_LABEL_WIDTH) + path


def status_sets(
    views: Views, scope: list[str] | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str], bool]:
    """What `git status` reports: staged and unstaged changes as (label, path), the untracked
    entries (a directory the index holds nothing under shows as `dir/`, ignored files not at
    all), and whether any of it is unknown."""
    staged: list[tuple[str, str]] = []
    unstaged: list[tuple[str, str]] = []
    untracked_paths: list[str] = []
    unknown = bool(views.unsure or views.state.modes)
    for path in views.touched():
        if not in_scope(path, scope or []):
            continue
        present = views.present(path)
        if present is None:
            unknown = True
            continue
        if path in views.state.staged_deleted:
            staged.append(("deleted", path))
        elif path in views.state.index:
            text = views.state.index[path]
            if not views.overlay.in_base(path):
                staged.append(("new file", path))
            elif text is None or text != views.head(path):
                staged.append(("modified", path))
                unknown = unknown or text is None
        if not views.in_index(path):
            if present:
                untracked_paths.append(path)
            continue
        current, reference = views.work(path), views.staged(path)
        if not present:
            unstaged.append(("deleted", path))
        elif current is None or reference is None or current != reference:
            unstaged.append(("modified", path))
            unknown = unknown or current is None
    untracked, doubtful = _untracked_entries(views, untracked_paths)
    staged = _renames(views, staged)
    labels = {label for label, _ in staged}
    unknown = unknown or {"deleted", "new file"} <= labels
    return staged, unstaged, untracked, unknown or doubtful


def _renames(views: Views, staged: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """A staged deletion and a staged new file with the same text are one rename. (One with
    similar text is too, which `status_sets` leaves unknown.)"""
    gone = {views.head(path): path for label, path in staged if label == "deleted"}
    out, used = [], set()
    for label, path in staged:
        source = gone.get(views.state.index.get(path)) if label == "new file" else None
        if source is not None and source not in used:
            used.add(source)
            out.append(("renamed", f"{source} -> {path}"))
        else:
            out.append((label, path))
    return [entry for entry in out if entry[0] != "deleted" or entry[1] not in used]


def _untracked_entries(views: Views, paths: list[str]) -> tuple[list[str], bool]:
    shown: set[str] = set()
    doubtful = False
    for path in paths:
        ignored = views.ignored(path)
        doubtful = doubtful or ignored is None
        if ignored is not False:
            continue
        parts = path.split("/")
        entry = next(
            (
                "/".join(parts[:depth]) + "/"
                for depth in range(1, len(parts))
                if not views.holds_tracked("/".join(parts[:depth]))
            ),
            path,
        )
        shown.add(entry)
    return sorted(shown), doubtful


def head_line(views: Views, meta: GitMeta) -> str:
    if views.state.detached:
        return DETACHED_HEADER
    if views.state.branch:
        return BRANCH_HEADER.format(branch=views.state.branch)
    if meta.detached:
        return DETACHED_HEADER
    return BRANCH_HEADER.format(branch=meta.branch or DEFAULT_BRANCH)


def _render_status_long(views: Views, meta: GitMeta, scope: list[str]) -> list[str]:
    staged, unstaged, untracked, _ = status_sets(views, scope)
    sections: list[list[str]] = []
    if staged:
        sections.append(
            [STAGED_HEADER, STAGED_HINT] + [_entry(label, path) for label, path in staged]
        )
    if unstaged:
        hints = list(UNSTAGED_HINTS)
        if any(label == "deleted" for label, _ in unstaged):
            hints[0] = hints[0].replace("git add ", "git add/rm ")
        sections.append(
            [UNSTAGED_HEADER, *hints] + [_entry(label, path) for label, path in unstaged]
        )
    if untracked:
        sections.append([UNTRACKED_HEADER, UNTRACKED_HINT] + ["\t" + path for path in untracked])
    head = [head_line(views, meta)]
    if meta.tracking and head[0].startswith("On branch"):
        head += [*meta.tracking.split("\n"), ""]
    if not sections:
        return [*head, CLEAN_TRAILER]
    lines = head
    for position, section in enumerate(sections):
        if position:
            lines.append("")
        lines.extend(section)
    lines.append("")
    if not staged:
        lines.append(UNSTAGED_TRAILER if unstaged else UNTRACKED_ONLY_TRAILER)
    return lines


def _render_status_short(views: Views, scope: list[str]) -> list[str]:
    staged, unstaged, untracked, _ = status_sets(views, scope)
    # a rename is keyed by its target, which an unstaged change to it names (`RM a -> b`)
    staged_map = {path.split(" -> ")[-1]: (label, path) for label, path in staged}
    unstaged_map = {path: label for label, path in unstaged}
    rows: list[tuple[str, str]] = []
    codes = {"new file": "A", "modified": "M", "deleted": "D", "renamed": "R"}
    for key in sorted(set(staged_map) | set(unstaged_map)):
        label, shown = staged_map.get(key, (None, key))
        left = codes[label] if label else " "
        right = codes[unstaged_map[key]] if key in unstaged_map else " "
        rows.append((left + right, shown))
    rows += [("??", path) for path in untracked]
    return [f"{code} {path}" for code, path in rows]


def _run_diff(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    allowed = {"--cached", "--staged", "--name-only", "--quiet", "--exit-code", "--no-color", "--"}
    if plan.flags - allowed or plan.values:
        return ParseFailure("unsupported_form", "diff flag")
    if any(".." in token for token in plan.paths):
        return ParseFailure("unsupported_form", "diff range")
    scope = normalize_paths([p for p in plan.paths if p != "HEAD"])
    if any(p not in views.listing_set for p in scope):
        return ParseFailure("unsupported_form", "diff pathspec")
    if "HEAD" in plan.paths:
        pairs, unknown = diff_head(views, scope)
    else:
        pairs, unknown = diff_pairs(views, bool({"--cached", "--staged"} & plan.flags), scope)
    if "--name-only" in plan.flags:
        lines = [path for path, _ in pairs]
    else:
        lines = [line for _, block in pairs for line in block]
    if not lines and scope:
        return ParseFailure("unsupported_form", "named path shows no change we can verify")
    returncode = 1 if {"--quiet", "--exit-code"} & plan.flags and pairs else 0
    if returncode and plan.pipeline:
        return ParseFailure("unsupported_form", "exit status of a piped diff")
    lines = [] if "--quiet" in plan.flags else lines
    lines = apply_pipeline(lines, plan.pipeline)
    return GitResult("\n".join(lines), returncode=returncode, empty=not lines, exact=not unknown)


def _run_status(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if plan.flags - {"-s", "--short", "--porcelain", "--long", "--"} or plan.values:
        return ParseFailure("unsupported_form", "status flag")
    scope = normalize_paths(plan.paths)
    if not any(in_scope(path, scope) for path in views.touched()) and not views.state.ledger:
        return ParseFailure("unsupported_form", "no observed change in status scope")
    if status_sets(views, scope)[3]:
        return ParseFailure("unsupported_form", "file with unknown content in status scope")
    if {"-s", "--short", "--porcelain"} & plan.flags:
        lines = _render_status_short(views, scope)
    else:
        lines = _render_status_long(views, meta, scope)
    lines = apply_pipeline(lines, plan.pipeline)
    return GitResult(output="\n".join(lines), empty=not lines)


def _run_add(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    """`git add` prints nothing, or fails on a pathspec that names nothing; adding an ignored
    file by name is refused with a hint this does not reproduce."""
    if plan.flags - {"-A", "--all", "-u", "--update", "-f", "--force", "--"} or plan.values:
        return ParseFailure("unsupported_form", "add flag")
    for token in plan.paths:
        if token in (".", "./") or token.startswith(":"):
            continue
        path = repo_path(token, views.overlay.cwd, views.overlay.root)
        if path is None or any(char in token for char in "*?["):
            return ParseFailure("unsupported_form", "add pathspec")
        kind = views.overlay.kind(path)
        if kind == "unknown":
            return ParseFailure("unsupported_form", "add of a path that may not exist")
        if kind is None and not views.tracked(path) and not views.holds_tracked(path):
            return GitResult(f"fatal: pathspec '{token}' did not match any files", returncode=128)
        if kind is not None and not views.tracked(path) and views.ignored(path) is not False:
            if not {"-f", "--force"} & plan.flags:
                return ParseFailure("unsupported_form", "add of an ignored path")
    return GitResult(output="", empty=True)


def _run_ls_files(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if plan.flags - {"--"}:
        return ParseFailure("unsupported_form", "ls-files flag")
    scope = normalize_paths(plan.paths)
    paths = [p for p in views.listing if in_scope(p, scope)]
    extra = sorted(p for p in views.state.index if p not in views.listing_set)
    lines = sorted(set(paths) | {p for p in extra if in_scope(p, scope)})
    lines = apply_pipeline(lines, plan.pipeline)
    return GitResult(output="\n".join(lines), empty=not lines)


def _run_rev_parse(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if not meta.sha:
        return ParseFailure("unsupported_form", "no head sha")
    if set(plan.values) - {"--short"} or len(plan.flags | set(plan.values)) > 1:
        return ParseFailure("unsupported_form", "rev-parse options")
    detached = views.state.detached or (meta.detached and not views.state.branch)
    if "--git-dir" in plan.flags:
        value = ".git"
    elif "--is-inside-work-tree" in plan.flags:
        value = "true"
    elif "--abbrev-ref" in plan.flags and plan.paths == ["HEAD"]:
        value = "HEAD" if detached else views.state.branch or meta.branch or DEFAULT_BRANCH
    elif "--short" in plan.values and plan.paths == ["HEAD"] and plan.values["--short"].isdigit():
        value = meta.sha[: max(4, int(plan.values["--short"]))]
    elif "--short" in plan.flags and plan.paths in (["HEAD"], []):
        value = meta.short
    elif plan.flags:
        return ParseFailure("unsupported_form", "toplevel path unknown")
    elif plan.paths in (["HEAD"], []):
        value = meta.sha
    else:
        return ParseFailure("unsupported_form", "rev-parse target")
    return GitResult(output=value, empty=False)


def _run_remote(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if not meta.owner or not meta.repo:
        return ParseFailure("unsupported_form", "remote unknown")
    if plan.paths and plan.paths[0] != "show":
        return ParseFailure("unsupported_form", "remote subcommand")
    url = f"https://github.com/{meta.owner}/{meta.repo}"
    if {"-v", "--verbose"} & plan.flags:
        lines = [f"origin\t{url} (fetch)", f"origin\t{url} (push)"]
    else:
        lines = ["origin"]
    return GitResult(output="\n".join(lines), empty=False)


def _names_head(rev: str, views: Views, meta: GitMeta) -> bool:
    """Whether a rev is the checked-out commit: HEAD, or a prefix of its sha when known."""
    if rev in ("HEAD", "@"):
        return True
    if len(rev) < 4 or not re.fullmatch(r"[0-9a-f]+", rev):
        return False
    shas = [views.state.head_short or ""]
    if meta.sha and (meta.detached or views.state.detached or meta.root_header):
        shas.append(meta.sha)
    return any(sha and (sha.startswith(rev) or rev.startswith(sha)) for sha in shas)


def _show_blob(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    """`git show <rev>:<path>`: the file's text as committed at the checked-out commit, which is
    the checkout as the session found it."""
    if plan.flags - {"--no-color"} or len(plan.paths) != 1:
        return ParseFailure("unsupported_form", "show <rev>:<path> with flags")
    rev, _, raw = plan.paths[0].partition(":")
    path = repo_path(raw)
    if not path or not _names_head(rev, views, meta):
        return ParseFailure("unsupported_form", "show <rev>:<path> of another commit")
    if not views.overlay.in_base(path):
        if views.holds_tracked(path) or plan.pipeline:
            return ParseFailure("unsupported_form", "show <rev>:<dir> or a failing pipe")
        where = "exists on disk, but not in" if views.present(path) else "does not exist in"
        error = "" if plan.dropped_stderr else f"fatal: path '{raw}' {where} '{rev}'"
        return GitResult(output=error, returncode=128, empty=not error)
    text = views.head(path)
    if text is None or "\x00" in text:
        return ParseFailure("unsupported_form", "show <rev>:<path> unreadable")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = apply_pipeline(lines, plan.pipeline)
    return GitResult(output="\n".join(lines), empty=not lines)


def _run_show(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if len(plan.paths) == 1 and ":" in plan.paths[0]:
        return _show_blob(plan, views, meta)
    if plan.flags - {"--stat", "--", "--no-color"}:
        return ParseFailure("unsupported_form", "show flag")
    wanted: list[str] = []
    if not plan.paths:
        if not (meta.detached and meta.sha):
            return ParseFailure("unsupported_form", "show HEAD is the harness commit")
        rev = meta.sha
    elif len(plan.paths) == 1:
        rev = plan.paths[0]
    elif "--" in plan.flags:
        rev = plan.paths[0]
        wanted = normalize_paths(plan.paths[1:])
        if not all(path in views.listing_set for path in wanted):
            return ParseFailure("unsupported_form", "show pathspec not tracked")
    else:
        return ParseFailure("unsupported_form", "show needs one rev")
    if ":" in rev:
        return ParseFailure("unsupported_form", "show <rev>:<path>")
    if not re.fullmatch(r"[0-9a-f]{6,40}", rev):
        return ParseFailure("unsupported_form", "show non-sha rev")
    if meta.squashed:
        return ParseFailure("unsupported_form", "the history is one local commit")
    if not callable(meta.commit_patch):
        return ParseFailure("unsupported_form", "no patch source")
    text = meta.commit_patch(rev)
    if not text:
        return ParseFailure("unsupported_form", "patch unavailable")
    patch = parse_patch(text)
    if patch is None:
        return ParseFailure("unsupported_form", "patch unparsed")
    lines = commit_header(patch)
    if patch["sha"] == meta.sha:
        decoration = _head_decoration(plan, views, meta)
        if isinstance(decoration, ParseFailure):
            return decoration
        lines[0] += decoration
    if "--stat" in plan.flags:
        if wanted:
            return ParseFailure("unsupported_form", "show --stat with pathspec")
        lines += ["", *patch["stat"]]
    else:
        diff = filter_diff(patch["diff"], wanted) if wanted else patch["diff"]
        if wanted and not diff:
            return ParseFailure("unsupported_form", "commit does not touch the pathspec")
        lines += ["", *retarget_funcnames(reabbrev(diff, views.abbrev), views)]
    lines = apply_pipeline(lines, plan.pipeline)
    return GitResult(output="\n".join(lines), empty=not lines)


_LOG_COUNT = re.compile(r"^-(\d+)$")
_LOG_ALLOWED = {"--oneline", "--no-decorate", "--no-merges", "--no-color", "--"}


def _run_log(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if "--oneline" not in plan.args:
        return ParseFailure("unsupported_form", "log format")
    limit = 0
    paths: list[str] = []
    after_dashes = False
    index = 0
    while index < len(plan.args):
        token = plan.args[index]
        index += 1
        if after_dashes:
            paths.append(token)
            continue
        if token == "--":
            after_dashes = True
            continue
        if match := _LOG_COUNT.match(token):
            limit = int(match.group(1))
            continue
        if token in ("-n", "--max-count") or token.startswith("--max-count="):
            value = token.partition("=")[2] or (plan.args[index] if index < len(plan.args) else "")
            index += 0 if "=" in token else 1
            if not value.isdigit():
                return ParseFailure("unsupported_form", "log count")
            limit = int(value)
            continue
        if token in _LOG_ALLOWED:
            continue
        if token.startswith("-"):
            return ParseFailure("unsupported_form", f"log flag {token}")
        paths.append(token)

    if len(paths) > 1:
        return ParseFailure("unsupported_form", "log with several pathspecs")
    scoped = None
    if paths:
        scoped = repo_path(paths[0])
        if not scoped or not (scoped in views.listing_set or views.holds_tracked(scoped)):
            return ParseFailure("unsupported_form", "log pathspec not tracked")
    if not callable(meta.history):
        return ParseFailure("unsupported_form", "no history source")
    payload = meta.history(scoped)
    if not payload or not payload.get("commits"):
        return ParseFailure("unsupported_form", "history unavailable")
    commits = payload["commits"]
    if not limit and not payload.get("complete"):
        if not plan.evidence:
            return ParseFailure("unsupported_form", "history incomplete for an unlimited log")
        limit = EVIDENCE_LOG_LIMIT

    lines = [f"{entry['sha'][: views.abbrev]} {entry['subject']}" for entry in commits]
    heads_at_checkout = bool(meta.sha) and commits[0]["sha"] == meta.sha
    head = 0 if heads_at_checkout else None
    if meta.squashed:
        # the one local commit is all the history there is, and only an observation names it
        if not views.state.head_short:
            return ParseFailure("unsupported_form", "the local commit's hash is unknown")
        lines, head = [f"{views.state.head_short} {views.state.head_subject}"], 0
    elif scoped is None and not (views.state.detached or meta.detached or heads_at_checkout):
        # a commit sits on top of the history the API knows, and only an observation can name it
        if not views.state.head_short:
            return ParseFailure("unsupported_form", "harness commit sha unknown")
        lines.insert(0, f"{views.state.head_short} {views.state.head_subject or HARNESS_SUBJECT}")
        head = 0
    if limit:
        lines = lines[:limit]
    if head is not None and head < len(lines):
        decoration = _head_decoration(plan, views, meta)
        if isinstance(decoration, ParseFailure):
            return decoration
        sha, _, subject = lines[head].partition(" ")
        lines[head] = f"{sha}{decoration} {subject}"
    lines = apply_pipeline(lines, plan.pipeline)
    return GitResult(output="\n".join(lines), empty=not lines)


def _head_decoration(plan: GitPlan, views: Views, meta: GitMeta) -> str | ParseFailure:
    """What git prints beside the commit HEAD is at: ` (HEAD)` at a detached HEAD when it writes
    to a terminal. On a branch it would also name the branch and its upstream, which are not
    known here."""
    terminal = not plan.pipeline and plan.redirect is None and "--no-decorate" not in plan.args
    if not terminal or meta.decorate is False:
        return ""
    if meta.decorate is None:
        return ParseFailure("unsupported_form", "whether git decorates refs is not known")
    if not (views.state.detached or (meta.detached and not views.state.branch)):
        return ParseFailure("unsupported_form", "the refs beside a branch's commit")
    return " (HEAD)"


def _run_checkout(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if plan.flags - {"--", "-f", "--force", "-q", "--quiet"}:
        return ParseFailure("unsupported_form", "checkout flag")
    operands = list(plan.paths)
    if not operands:
        return ParseFailure("unsupported_form", "checkout without pathspec")
    if "--" in plan.flags:
        return GitResult(output="", empty=True)
    normalized = normalize_paths(operands)
    if not all(p in views.listing_set for p in normalized):
        return ParseFailure("unsupported_form", "checkout target not a tracked path")
    count = sum(1 for path in normalized if differs_from(views, path, "index"))
    return GitResult(
        output=UPDATED_PATHS_LINE.format(n=count, s="" if count == 1 else "s"), empty=False
    )


def _run_restore(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if plan.flags - {"--", "--staged", "-S", "--worktree", "-W"}:
        return ParseFailure("unsupported_form", "restore flag")
    if not plan.paths:
        return ParseFailure("unsupported_form", "restore without pathspec")
    return GitResult(output="", empty=True)


def _run_reset(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if plan.flags - {"--", "--hard", "--soft", "--mixed", "-q", "--quiet"}:
        return ParseFailure("unsupported_form", "reset flag")
    revs = [p for p in plan.paths if p == "HEAD"]
    operands = [p for p in plan.paths if p != "HEAD"]
    if len(revs) + len(operands) != len(plan.paths):
        return ParseFailure("unsupported_form", "reset rev")
    if {"--soft", "-q", "--quiet"} & plan.flags:
        return GitResult(output="", empty=True)
    if "--hard" in plan.flags:
        if operands:
            return ParseFailure("unsupported_form", "reset --hard with paths")
        resolve_head(views, meta)
        short, subject = views.state.head_short, views.state.head_subject
        if not short or subject is None:
            return ParseFailure("unsupported_form", "head subject unknown")
        return GitResult(output=HARD_RESET_LINE.format(short=short, subject=subject), empty=False)
    scope = normalize_paths(operands)
    residual = dirty_paths(views, scope, "head")
    if not residual:
        return GitResult(output="", empty=True)
    rows = [f"{'M' if views.present(path) else 'D'}\t{path}" for path in residual]
    return GitResult(output="\n".join([RESET_HEADER, *rows]), empty=False)


def _run_stash(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    action = plan.paths[0] if plan.paths else "push"
    state = views.state
    if action in ("pop", "apply") and not state.stash:
        return GitResult(output=STASH_MISSING_LINE, returncode=1, empty=False)
    if action not in ("push", "save"):
        return ParseFailure("unsupported_form", f"git stash {action}")
    untracked = bool({"-u", "--include-untracked", "-a", "--all"} & plan.flags)
    if views.unsure or (untracked and status_sets(views)[3]):
        return ParseFailure("unsupported_form", "stash of files not known")
    changed = dirty_paths(views, [], "head") or state.index or state.staged_deleted
    if not changed and not (untracked and status_sets(views)[2]):
        return GitResult(output=STASH_EMPTY_LINE, empty=False)
    resolve_head(views, meta)
    short, subject = state.head_short, state.head_subject
    if not short or subject is None:
        return ParseFailure("unsupported_form", "head subject unknown")
    branch = "(no branch)" if state.detached else (state.branch or meta.branch or DEFAULT_BRANCH)
    return GitResult(
        output=STASH_SAVED_LINE.format(branch=branch, short=short, subject=subject), empty=False
    )


def _run_mv(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    if moved_files(plan, views) is None:
        return ParseFailure("unsupported_form", "git mv")
    return GitResult(output="", empty=True)


def _run_rm(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    files = removed_files(plan, views)
    if files is None:
        return ParseFailure("unsupported_form", "git rm")
    quiet = {"-q", "--quiet"} & plan.flags
    return GitResult(output="" if quiet else "\n".join(f"rm '{path}'" for path in files))


def _run_branch(plan: GitPlan, views: Views, meta: GitMeta) -> GitResult | ParseFailure:
    """`git branch` at a detached HEAD of a checkout with no other branch: `* (no branch)`. The
    branches of a checkout on one, remote ones and the details `-v` adds are not known here."""
    if plan.flags - {"-a", "--all", "--list"} or plan.values or plan.paths:
        return ParseFailure("unsupported_form", "branch form")
    if views.state.branch or not (views.state.detached or meta.detached):
        return ParseFailure("unsupported_form", "the branches of a checkout on one")
    lines = apply_pipeline(["* (no branch)"], plan.pipeline)
    return GitResult(output="\n".join(lines), empty=not lines)


HANDLERS = {
    "mv": _run_mv,
    "rm": _run_rm,
    "status": _run_status,
    "diff": _run_diff,
    "add": _run_add,
    "ls-files": _run_ls_files,
    "rev-parse": _run_rev_parse,
    "remote": _run_remote,
    "checkout": _run_checkout,
    "log": _run_log,
    "show": _run_show,
    "restore": _run_restore,
    "reset": _run_reset,
    "stash": _run_stash,
    "branch": _run_branch,
}
