from __future__ import annotations

import bisect
import fnmatch

from ..command_search import repo_path, shell_glob
from .models import GitMeta, GitPlan, GitState


def normalize_paths(tokens: list[str]) -> list[str]:
    return [repo_path(token) or "" for token in tokens]


def in_scope(path: str, scope: list[str]) -> bool:
    return not scope or any(
        entry in ("", ".", "./") or path == entry or path.startswith(entry.rstrip("/") + "/")
        for entry in scope
    )


class Views:
    """The three versions git compares - HEAD, the index, the working tree - of each path."""

    def __init__(self, overlay, state: GitState, read_base, listing: list[str], abbrev: int):
        self.overlay = overlay
        self.state = state
        self.read_base = read_base
        self.listing = listing
        self.listing_set = set(listing)
        self.abbrev = abbrev

    def head(self, path: str) -> str | None:
        return self.read_base(path)

    def present(self, path: str) -> bool | None:
        """Whether the working tree holds the file; None when that is not known."""
        kind = self.overlay.kind(path)
        return None if kind == "unknown" else kind == "file"

    def work(self, path: str) -> str | None:
        """The working-tree text of a file present, None when it is unknown or absent."""
        return self.overlay.text(path)

    def staged(self, path: str) -> str | None:
        if path in self.state.staged_deleted:
            return None
        if path in self.state.index:
            return self.state.index[path]
        return self.read_base(path)

    def in_index(self, path: str) -> bool:
        return path not in self.state.staged_deleted and (
            path in self.state.index or self.overlay.in_base(path)
        )

    @property
    def unsure(self) -> set[str]:
        """The checkout's directories whose entries are unknown (git sees nothing outside it)."""
        return {path for path in self.overlay.unsure if path[:1] != "/"}

    def touched(self) -> list[str]:
        overlay = self.overlay
        paths = set(overlay.content) | overlay.dirty | set(self.state.index) | overlay.maybe
        paths |= self.state.staged_deleted | overlay.created | overlay.deleted
        return sorted(p for p in paths if p and not p.startswith("/") and ".." not in p)

    def tracked(self, path: str) -> bool:
        return self.overlay.in_base(path) or path in self.state.index

    def holds_tracked(self, directory: str) -> bool:
        """Whether the index holds a file under a directory."""
        inside = directory + "/"
        base = self.overlay.base
        for path in base[bisect.bisect_left(base, inside) :]:
            if not path.startswith(inside):
                break
            if path not in self.state.staged_deleted:
                return True
        return any(p.startswith(inside) for p in self.state.index)

    def ignored(self, path: str) -> bool | None:
        """Whether .gitignore rules exclude an untracked path; None when a .gitignore that
        decides it has unknown text."""
        parts = path.split("/")
        for depth in range(1, len(parts) + 1):
            decided = self._excluded("/".join(parts[:depth]), depth < len(parts))
            if decided is not False:
                return decided
        return False

    def _excluded(self, path: str, is_dir: bool) -> bool | None:
        """The last .gitignore rule matching a path decides it, deeper files after shallower."""
        decision = False
        parts = path.split("/")
        for depth in range(len(parts)):
            base = "/".join(parts[:depth])
            rules = f"{base}/.gitignore" if base else ".gitignore"
            kind = self.overlay.kind(rules)
            text = self.overlay.text(rules)
            if kind == "unknown" or (kind == "file" and text is None):
                return None
            for line in (text or "").split("\n"):
                matched = _rule_matches(line, "/".join(parts[depth:]), is_dir)
                if matched is not None:
                    decision = matched
        return decision


def _rule_matches(line: str, path: str, is_dir: bool) -> bool | None:
    """How one .gitignore line decides a path relative to the file's directory: True to
    exclude, False to re-include (`!`), None when it does not match."""
    rule = line.rstrip()
    if not rule or rule.startswith("#"):
        return None
    negate = rule.startswith("!")
    rule = rule[1:] if negate else rule
    if rule.endswith("/"):
        if not is_dir:
            return None
        rule = rule.rstrip("/")
    if "/" in rule:
        rule = rule.lstrip("/")
        candidates = [rule, rule[3:]] if rule.startswith("**/") else [rule]
        hit = any(shell_glob(path, candidate) for candidate in candidates)
    else:
        hit = fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], rule)
    return (not negate) if hit else None


def differs_from(views: Views, path: str, against: str) -> bool:
    """Whether the working copy of a path differs from HEAD or from the index.

    A working copy we cannot read counts as differing: the caller can only prove a path is
    clean by comparing the two texts, never by failing to find one.
    """
    reference = views.head(path) if against == "head" else views.staged(path)
    if not views.present(path):
        return reference is not None or views.present(path) is None
    current = views.work(path)
    return current is None or reference is None or current != reference


def dirty_paths(views: Views, scope: list[str], against: str) -> list[str]:
    return [
        path
        for path in views.touched()
        if in_scope(path, scope) and views.tracked(path) and differs_from(views, path, against)
    ]


def resolve_head(views: Views, meta: GitMeta) -> None:
    """Name the head commit from the API's history when an observation has not yet."""
    state = views.state
    if state.head_short and state.head_subject is not None:
        return
    if not (meta.detached and meta.sha and state.abbrev and callable(meta.history)):
        return
    commits = (meta.history(None) or {}).get("commits") or []
    if commits and commits[0]["sha"] == meta.sha:
        state.head_short = meta.sha[: state.abbrev]
        state.head_subject = commits[0]["subject"]


def moved_files(plan: GitPlan, views: Views) -> list[tuple[str, str]] | None:
    """The (source, destination) of each file `git mv SRC DST` moves: empty for a move git
    refuses, None for one this does not follow."""
    overlay = views.overlay
    if plan.flags - {"-f", "--force", "--"} or len(plan.paths) != 2:
        return None
    source, target = (repo_path(token, overlay.cwd, overlay.root) for token in plan.paths)
    kind = overlay.kind(source) if source else None
    if not source or target is None or kind not in ("file", "directory"):
        return None
    if overlay.kind(target) == "directory":
        target = f"{target}/{source.rsplit('/', 1)[-1]}" if target else source.rsplit("/", 1)[-1]
    elif overlay.kind(target) is not None:
        return None
    parent = overlay.kind(target.rpartition("/")[0])
    if parent == "unknown":
        return None
    if parent != "directory":
        return []  # git refuses: the destination directory does not exist
    files = [source] if kind == "file" else overlay.files_under(source)
    if not files or any(not views.in_index(path) for path in files):
        return None
    return [(path, target + path[len(source) :]) for path in files]


def removed_files(plan: GitPlan, views: Views) -> list[str] | None:
    """The files `git rm` removes from the index (and, without --cached, the working tree);
    None for a removal git refuses or this does not follow."""
    overlay = views.overlay
    allowed = {"-r", "--cached", "-f", "--force", "-q", "--quiet", "--"}
    if plan.flags - allowed or not plan.paths:
        return None
    files: list[str] = []
    for token in plan.paths:
        path = repo_path(token, overlay.cwd, overlay.root)
        if path is None or overlay.kind(path) == "unknown":
            return None
        if views.in_index(path):
            files.append(path)
        elif "-r" in plan.flags and views.holds_tracked(path):
            files += [p for p in [*overlay.files_under(path), *overlay.deleted] if
                      p.startswith(path + "/") and views.in_index(p)]  # fmt: skip
        else:
            return None
    if not {"-f", "--force", "--cached"} & plan.flags and any(
        views.work(path) != views.staged(path) or path in views.state.index for path in files
    ):
        return None
    return sorted(set(files))
