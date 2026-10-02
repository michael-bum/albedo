from __future__ import annotations

import re

from .models import GitState
from .templates import AUTHOR_LINE, COMMIT_LINE, DATE_LINE, FUNCNAME, FUNCNAME_MAX_CHARS
from .views import Views

_PATCH_FROM = re.compile(r"^From ([0-9a-f]{40}) ", re.M)
_PATCH_AUTHOR = re.compile(r"^From: (.+(?:\n[ \t].+)*)$", re.M)
_PATCH_DATE = re.compile(r"^Date: (.+)$", re.M)
_PATCH_INDEX = re.compile(r"^index ([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})(.*)$", re.M)


def _git_date(raw: str) -> str | None:
    from email.utils import parsedate_to_datetime

    try:
        moment = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        return None
    return f"{moment:%a %b} {moment.day} {moment:%H:%M:%S %Y} {moment:%z}"


def parse_patch(text: str) -> dict | None:
    from email.header import decode_header, make_header

    sha = _PATCH_FROM.search(text or "")
    author = _PATCH_AUTHOR.search(text or "")
    date = _PATCH_DATE.search(text or "")
    if not (sha and author and date):
        return None
    try:
        author_text = str(make_header(decode_header(author.group(1).strip())))
    except (UnicodeDecodeError, ValueError):
        author_text = author.group(1).strip()
    stamp = _git_date(date.group(1))
    if stamp is None:
        return None
    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if line.startswith("Subject: ")), None)
    if start is None:
        return None
    subject_parts = [lines[start][len("Subject: ") :]]
    index = start + 1
    while index < len(lines) and lines[index].startswith(" "):
        subject_parts.append(lines[index][1:])
        index += 1
    subject = " ".join(part.strip() for part in subject_parts).strip()
    for prefix in ("[PATCH] ", "[PATCH"):
        if subject.startswith(prefix):
            subject = subject.split("] ", 1)[-1] if "] " in subject else subject[len(prefix) :]
            break
    body: list[str] = []
    while index < len(lines) and lines[index].strip() != "---":
        body.append(lines[index])
        index += 1
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    diff_at = next(
        (i for i, line in enumerate(lines) if line.startswith("diff --git ")), len(lines)
    )
    stat = [line for line in lines[index + 1 : diff_at] if line.strip()]
    diff = lines[diff_at:]
    while diff and not diff[-1].strip():
        diff.pop()
    return {
        "sha": sha.group(1),
        "author": author_text,
        "date": stamp,
        "subject": subject,
        "body": body,
        "stat": stat,
        "diff": diff,
    }


def commit_header(patch: dict) -> list[str]:
    lines = [
        COMMIT_LINE.format(sha=patch["sha"]),
        AUTHOR_LINE.format(author=patch["author"]),
        DATE_LINE.format(date=patch["date"]),
        "",
        "    " + patch["subject"],
    ]
    if patch["body"]:
        lines += ["    ", *(("    " + line) if line.strip() else "    " for line in patch["body"])]
    return lines


def reabbrev(diff: list[str], abbrev: int) -> list[str]:
    def shorten(match):
        old, new, tail = match.group(1), match.group(2), match.group(3)
        if len(old) < abbrev or len(new) < abbrev:
            return match.group(0)
        return f"index {old[:abbrev]}..{new[:abbrev]}{tail}"

    return [_PATCH_INDEX.sub(shorten, line) for line in diff]


_HUNK_HEADER = re.compile(r"^(@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@)(?: (.*))?$")
_NEW_SIDE = re.compile(r"^\+\+\+ b/(.*)$")


def _funcname_above(file_lines: list[str], anchor: str, near: int) -> str | None:
    positions = [i for i, line in enumerate(file_lines) if line == anchor]
    if not positions:
        return None
    start = min(positions, key=lambda i: abs(i - near))
    for above in range(start - 1, -1, -1):
        if FUNCNAME.match(file_lines[above]):
            return " " + file_lines[above].rstrip()[:FUNCNAME_MAX_CHARS]
    return ""


def retarget_funcnames(diff: list[str], views: Views) -> list[str]:
    out = list(diff)
    file_lines: list[str] | None = None
    for position, line in enumerate(diff):
        if match := _NEW_SIDE.match(line):
            path = match.group(1)
            text = views.read_base(path) if path in views.listing_set else None
            file_lines = text.split("\n") if text else None
            continue
        header = _HUNK_HEADER.match(line)
        if not header or file_lines is None:
            continue
        anchor = next(
            (body[1:] for body in diff[position + 1 : position + 6] if body[:1] in (" ", "-")),
            None,
        )
        if anchor is None:
            continue
        replacement = _funcname_above(file_lines, anchor, int(header.group(2)) - 1)
        if replacement is not None:
            out[position] = header.group(1) + replacement
    return out


_DIFF_HEADER = re.compile(r"^diff --git a/(.*) b/(.*)$")


def filter_diff(diff: list[str], wanted: list[str]) -> list[str]:
    keep = set(wanted)
    out: list[str] = []
    including = False
    for line in diff:
        if match := _DIFF_HEADER.match(line):
            including = match.group(2) in keep or match.group(1) in keep
        if including:
            out.append(line)
    return out


_OUTPUT_BODY = re.compile(r"<output>\n?(.*?)\n?</output>", re.S)
_DIFF_OLD = re.compile(r"^--- (\S+)")
_DIFF_NEW = re.compile(r"^\+\+\+ (\S+)")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def unified_diff(text: str) -> list[tuple[str, str, list[tuple[int, list[str]]], bool]] | None:
    """The files of a unified diff, each as (old path, new path, hunks, whether a line is
    marked as lacking its final newline), a hunk as (old start line, body); None when the text
    is not a diff this reads."""
    files = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        old = _DIFF_OLD.match(lines[index])
        new = _DIFF_NEW.match(lines[index + 1]) if old and index + 1 < len(lines) else None
        if not new:
            index += 1
            continue
        hunks, marked = [], False
        index += 2
        while index < len(lines) and (header := _HUNK.match(lines[index])):
            old_left = int(header.group(2) or 1)
            new_left = int(header.group(4) or 1)
            body, index = [], index + 1
            while (old_left > 0 or new_left > 0) and index < len(lines):
                line = lines[index]
                tag = line[:1] or " "
                if tag == "\\":
                    marked = True
                elif tag in " -+":
                    body.append(tag + line[1:])
                    old_left -= tag in " -"
                    new_left -= tag in " +"
                else:
                    return None
                index += 1
            if index < len(lines) and lines[index].startswith("\\"):
                marked, index = True, index + 1
            hunks.append((int(header.group(1)), body))
        files.append((old.group(1), new.group(1), hunks, marked))
    return files or None


def observed_patches(observation: str) -> dict[str, list[tuple[int, list[str]]]]:
    """The hunks of each file a diff shown in an observation changes in place: not a file it
    adds or removes, nor one whose last line lacks its line break (the hunks do not carry it)."""
    body = observation or ""
    if match := _OUTPUT_BODY.search(body):
        body = match.group(1)
    return {
        new.removeprefix("b/"): hunks
        for old, new, hunks, marked in unified_diff(body) or []
        if "/dev/null" not in (old, new) and not marked and hunks
    }


def apply_hunks(text: str, hunks: list[tuple[int, list[str]]]) -> str | None:
    """`text` with the hunks applied at the lines they name, or None when one does not match
    there. A hunk that only adds lines starts after the line its start names."""
    lines = text.split("\n")
    trailing = bool(lines) and lines[-1] == ""
    if trailing:
        lines.pop()
    out: list[str] = []
    cursor = 0
    for start, body in hunks:
        index = start - 1 if any(entry[:1] in " -" for entry in body) else start
        if index < cursor or index > len(lines):
            return None
        out.extend(lines[cursor:index])
        position = index
        for entry in body:
            tag, content = entry[:1], entry[1:]
            if tag in " -":
                if position >= len(lines) or lines[position] != content:
                    return None
                position += 1
                if tag == " ":
                    out.append(content)
            else:
                out.append(content)
        cursor = position
    out.extend(lines[cursor:])
    return "\n".join(out) + "\n" if out else ""


def learn_from_observed_diff(overlay, state: GitState, command: str, observation: str, read_base):
    text = command or ""
    if "--stat" in text or "--name-only" in text or "--numstat" in text:
        return
    staged = "--cached" in text or "--staged" in text
    if not staged and state.index:
        return
    for path, hunks in observed_patches(observation).items():
        base = read_base(path)
        if base is None:
            continue
        patched = apply_hunks(base, hunks)
        if patched is None:
            continue
        if staged:
            state.index[path] = patched
        else:
            overlay.know(path, patched)
