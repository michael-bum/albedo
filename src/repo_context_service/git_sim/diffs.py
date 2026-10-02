from __future__ import annotations

import hashlib

from .templates import (
    DEFAULT_MODE,
    FUNCNAME,
    FUNCNAME_MAX_CHARS,
    NO_NEWLINE_MARKER,
    NULL_BLOB,
)
from .views import Views, in_scope


def blob_hash(text: str) -> str:
    data = text.encode("utf-8", "surrogateescape")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _text_lines(text: str) -> tuple[list[str], bool]:
    if text == "":
        return [], False
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
        return parts, False
    return parts, True


def _funcname(lines: list[str], start: int) -> str:
    for index in range(min(start, len(lines)) - 1, -1, -1):
        line = lines[index]
        if FUNCNAME.match(line):
            return " " + line.strip()[:FUNCNAME_MAX_CHARS]
    return ""


def _hunks(
    old: list[str], new: list[str], old_open: bool, new_open: bool, context: int = 3
) -> list[str]:
    from difflib import SequenceMatcher

    out: list[str] = []
    # a last line without its line break differs from the same line with one
    keyed_old = old[:-1] + [old[-1] + "\0"] if old and old_open else old
    keyed_new = new[:-1] + [new[-1] + "\0"] if new and new_open else new
    matcher = SequenceMatcher(None, keyed_old, keyed_new, autojunk=False)
    for group in matcher.get_grouped_opcodes(context):
        first, last = group[0], group[-1]
        old_start, old_len = first[1], last[2] - first[1]
        new_start, new_len = first[3], last[4] - first[3]
        out.append(
            f"@@ -{_span(old_start, old_len)} +{_span(new_start, new_len)} @@"
            f"{_funcname(old, old_start)}"
        )
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for offset in range(i1, i2):
                    out.append(" " + old[offset])
                    if old_open and new_open and offset == len(old) - 1:
                        out.append(NO_NEWLINE_MARKER)
                continue
            if tag in ("replace", "delete"):
                for offset in range(i1, i2):
                    out.append("-" + old[offset])
                    if old_open and offset == len(old) - 1:
                        out.append(NO_NEWLINE_MARKER)
            if tag in ("replace", "insert"):
                for offset in range(j1, j2):
                    out.append("+" + new[offset])
                    if new_open and offset == len(new) - 1:
                        out.append(NO_NEWLINE_MARKER)
    return out


def _span(start: int, length: int) -> str:
    """A hunk header range: git leaves the length out when it is 1."""
    return str(start + 1) if length == 1 else f"{start + (1 if length else 0)},{length}"


def _diff_file(path: str, old: str | None, new: str | None, abbrev: int) -> list[str]:
    if old == new:
        return []
    header = [f"diff --git a/{path} b/{path}"]
    old_lines, old_open = _text_lines(old or "")
    new_lines, new_open = _text_lines(new or "")
    old_hash = NULL_BLOB if old is None else blob_hash(old)
    new_hash = NULL_BLOB if new is None else blob_hash(new)
    index = f"index {old_hash[:abbrev]}..{new_hash[:abbrev]}"
    if old is None:
        header += [f"new file mode {DEFAULT_MODE}", index]
        left, right = "/dev/null", f"b/{path}"
    elif new is None:
        header += [f"deleted file mode {DEFAULT_MODE}", index]
        left, right = f"a/{path}", "/dev/null"
    else:
        header.append(f"{index} {DEFAULT_MODE}")
        left, right = f"a/{path}", f"b/{path}"
    body = _hunks(old_lines, new_lines, old_open, new_open)
    return header + [f"--- {left}", f"+++ {right}", *body] if body else header


def diff_pairs(
    views: Views, cached: bool, scope: list[str]
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """`git diff` (index to working tree) or `git diff --cached` (HEAD to index): each changed
    path's diff, and the paths whose change is unknown."""
    pairs: list[tuple[str, list[str]]] = []
    unknown: list[str] = sorted(views.unsure)
    for path in views.touched():
        if not in_scope(path, scope):
            continue
        if cached:
            if path not in views.state.index and path not in views.state.staged_deleted:
                continue
            old = views.head(path) if views.overlay.in_base(path) else None
            new = None if path in views.state.staged_deleted else views.state.index[path]
            if new is None and path not in views.state.staged_deleted:
                unknown.append(path)
                continue
        else:
            if not views.in_index(path):
                continue
            old, new = views.staged(path), _working(views, path, unknown)
            if new is _UNKNOWN:
                continue
        if old != new and (lines := _diff_file(path, old, new, views.abbrev)):
            pairs.append((path, lines))
    return _renamed(pairs, views, unknown) if cached else pairs, unknown


def _renamed(pairs: list[tuple[str, list[str]]], views: Views, unknown: list[str]):
    """git shows a deleted file and an added one with the same text as one rename. One with
    similar text is a rename too, which this does not measure: that pair is left unknown."""
    added = {
        path: views.state.index.get(path)
        for path, lines in pairs
        if len(lines) > 1 and lines[1].startswith("new file")
    }
    out, renamed = [], set()
    for path, lines in pairs:
        if not lines[1].startswith("deleted file"):
            continue
        text = views.head(path)
        target = next((new for new, body in added.items() if body == text), None)
        if target is not None and target not in renamed:
            renamed |= {path, target}
            header = [f"diff --git a/{path} b/{target}", "similarity index 100%"]
            out.append((target, [*header, f"rename from {path}", f"rename to {target}"]))
        elif added:
            unknown.append(path)
    return sorted([(p, lines) for p, lines in pairs if p not in renamed] + out)


def diff_head(views: Views, scope: list[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """`git diff HEAD`: HEAD to working tree, for every path the index holds."""
    pairs: list[tuple[str, list[str]]] = []
    unknown: list[str] = sorted(views.unsure)
    for path in views.touched():
        if not in_scope(path, scope) or not views.in_index(path):
            continue
        old = views.head(path) if views.overlay.in_base(path) else None
        new = _working(views, path, unknown)
        if (
            new is not _UNKNOWN
            and old != new
            and (block := _diff_file(path, old, new, views.abbrev))
        ):
            pairs.append((path, block))
    return pairs, unknown


_UNKNOWN = "\0unknown"


def _working(views: Views, path: str, unknown: list[str]) -> str | None:
    """The working-tree text of a path, None when it is absent; `_UNKNOWN`, recorded in
    `unknown`, when that is not known."""
    present = views.present(path)
    text = views.work(path) if present else None
    if present is None or (present and text is None):
        unknown.append(path)
        return _UNKNOWN
    return text
