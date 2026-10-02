"""Exact answers for the read-only commands agents run against a checkout.

`parse_search(command)` reads one pipeline - a read (`cat`, `nl`, `head`, `tail`, `sed -n`, an
`awk` line range, `wc`), a search (`grep`, `find`, `ls`), `echo`, `printf` or `pwd`, followed by
pipe stages that filter its output (`head`, `tail`, `sort`, `uniq`, `wc`, `grep`, `cat -n`,
`sed -n`) - into a `SearchPlan`, or a `ParseFailure` for anything it does not reproduce.
`run_search(plan, ...)` runs it against the files of a listing and gives the terminal's text and
the exit status.

A command writes its error messages to the terminal, not to the pipe, unless it redirects them
(`2>&1`); `2>/dev/null` drops them. A directory walk (`grep -r`, `find`) visits entries in an order
only the filesystem knows: its lines come out sorted here, and a pipe stage whose output depends on
their order (`head`, `tail`, ...) is not reproduced. The checkout's `.git` directory exists with
entries that are not known: a listing of the root shows it, and a walk that would enter it is only
reproduced when its filters exclude every file git keeps there.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .shell import SimpleCommand, Unsupported, parse_command


@dataclass
class ParseFailure:
    reason: str
    detail: str = ""


@dataclass
class Stage:
    """A pipe stage: `name` says what it does to its input lines, `args` how."""

    name: str
    args: list[str] = field(default_factory=list)
    # the grep a "filter" stage runs
    options: GrepOptions | None = None


@dataclass
class GrepOptions:
    """How a grep matches and what it prints; the files it searches are its operands."""

    patterns: list[str] = field(default_factory=list)
    recursive: bool = False
    ignore_case: bool = False
    word: bool = False
    whole_line: bool = False
    fixed: bool = False
    extended: bool = False
    invert: bool = False
    line_numbers: bool = False
    files_only: bool = False
    files_without: bool = False
    count_only: bool = False
    quiet: bool = False
    no_messages: bool = False
    skip_binary: bool = False
    binary_text: bool = False
    with_filename: bool | None = None
    after: int = 0
    before: int = 0
    max_count: int = 0
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    exclude_dir: list[str] = field(default_factory=list)
    # grep run by find -exec or xargs: once per file, or once for all
    per_file: bool = False


@dataclass
class SearchPlan:
    tool: str = ""
    targets: list[str] = field(default_factory=list)
    # targets that came from an unquoted glob, which the shell expands against the checkout
    globbed: set[str] = field(default_factory=set)
    # find
    find_tests: list[tuple[str, str, bool]] = field(default_factory=list)
    # the tests are alternatives (`-o`) rather than all required
    any_test: bool = False
    pruned: list[tuple[str, str]] = field(default_factory=list)
    type_filter: str = ""
    max_depth: int | None = None
    min_depth: int = 0
    # the grep this command is (tool "grep"), or runs on what find prints (-exec, xargs)
    grep: GrepOptions | None = None
    exec_batched: bool = False
    xargs_skip_empty: bool = False
    # ls
    show_hidden: bool = False
    show_dots: bool = False
    long: bool = False
    one_per_line: bool = False
    # reads
    read_mode: str = ""
    read_range: tuple[int, int] | None = None
    counts: str = ""
    number_lines: bool = False
    literal: str | None = None
    print_cwd: bool = False
    pipeline: list[Stage] = field(default_factory=list)
    # where messages go: dropped (2>/dev/null) or into the pipe (2>&1); output dropped (>/dev/null)
    errors_dropped: bool = False
    errors_piped: bool = False
    output_dropped: bool = False


@dataclass
class SearchResult:
    output: str
    missing: list[str] = field(default_factory=list)
    empty: bool = False
    # None when not derivable: a wrong status would tell the assistant its command failed
    returncode: int | None = None
    # the output does not end with a line break: its last line is a file's unterminated last line
    open_end: bool = False

    @property
    def raw(self) -> str:
        """The output exactly as the terminal shows it, final line break included."""
        return "" if self.empty else self.output + ("" if self.open_end else "\n")


class _Decline(Exception):
    """Raised inside a run when the answer depends on something not known here."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass
class _Run:
    """What a command wrote, in order: each line with whether it is an error message."""

    lines: list[tuple[str, bool]] = field(default_factory=list)
    returncode: int = 0
    open_end: bool = False
    # lines that come from a directory walk, in an order only the filesystem decides
    unordered: bool = False
    missing: list[str] = field(default_factory=list)

    def out(self, line: str) -> None:
        self.lines.append((line, False))

    def error(self, line: str, returncode: int) -> None:
        self.lines.append((line, True))
        self.returncode = returncode


_MAX_FILES = 5000
_MAX_LINES = 5000
# the most text a command may read or make here; one past it is left to the simulator
MAX_TEXT_CHARS = 8 * 1024 * 1024
# more words than any real use of a brace list makes
_MAX_BRACE_WORDS = 256

_POSIX_CLASS = {
    "alpha": "a-zA-Z",
    "digit": "0-9",
    "alnum": "a-zA-Z0-9",
    "upper": "A-Z",
    "lower": "a-z",
    "space": " \\t\\n\\r\\f\\v",
    "blank": " \\t",
    "punct": "!-/:-@\\[-`{-~",
    "xdigit": "0-9A-Fa-f",
    "cntrl": "\\x00-\\x1f\\x7f",
    "print": " -~",
    "graph": "!-~",
}
_SWAPPED = "|(){}+?"
# escapes GNU grep and sed give a meaning; any other escaped letter stands for itself
_GNU_ESCAPES = set("wWsSbB<>`'") | set("0123456789") | set(".*[]^$\\/+?(){}|")


def ere_to_python(pattern: str, sed: bool = False) -> str | None:
    """A POSIX extended regular expression, with GNU's extensions, as a Python one; None when
    it names a character class that does not exist. `sed` reads `\t` and `\n` as GNU sed does,
    a tab and a line break (grep reads them as the letters)."""
    return _translate(pattern, basic=False, sed=sed)


def bre_to_python(pattern: str, sed: bool = False) -> str | None:
    r"""A POSIX basic regular expression (grep, sed without -E) as a Python one. There `|`, `(`,
    `)`, `{`, `}`, `+` and `?` are literal and their backslashed forms are the operators, the
    other way round from Python; `^` and `$` anchor only at the ends of the expression or of a
    `\(...\)` group or `\|` alternative."""
    return _translate(pattern, basic=True, sed=sed)


def _translate(pattern: str, basic: bool, sed: bool) -> str | None:
    out: list[str] = []
    in_brackets = False
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if in_brackets:
            if pattern.startswith("[:", index):
                close = pattern.find(":]", index)
                members = _POSIX_CLASS.get(pattern[index + 2 : close]) if close != -1 else None
                if members is None:
                    return None
                out.append(members)
                index = close + 2
                continue
            # inside brackets a backslash is a member, not an escape
            out.append("\\\\" if char == "\\" else char)
            opening = pattern[index - 1] == "[" or pattern[index - 2 : index] == "[^"
            in_brackets = not (char == "]" and not opening)
        elif char == "[":
            out.append(char)
            in_brackets = True
        elif char == "\\" and index + 1 < len(pattern):
            following = pattern[index + 1]
            if following == "<":
                out.append(r"\b(?=\w)")
            elif following == ">":
                out.append(r"\b(?<=\w)")
            elif sed and following in "tn":
                out.append(char + following)
            elif following not in _GNU_ESCAPES:
                out.append(re.escape(following))
            elif basic and following in _SWAPPED:
                out.append(following)
            else:
                out.append(char + following)
            index += 2
            continue
        elif (
            basic
            and char == "^"
            and not (index == 0 or pattern[index - 2 : index] in ("\\(", "\\|"))
        ):
            out.append("\\^")
        elif (
            basic
            and char == "$"
            and not (index == len(pattern) - 1 or pattern[index + 1 : index + 3] in ("\\)", "\\|"))
        ):
            out.append("\\$")
        else:
            out.append("\\" + char if basic and char in _SWAPPED else char)
        index += 1
    return "".join(out)


# where a sandbox holds the checkout: /testbed, or /workspace/<owner>__<repo>__<version>. Narrower
# than a bare /workspace/<name>, so a path an observation invented at the workspace root cannot
# pass for the checkout
CHECKOUT_ROOT = re.compile(
    r"/testbed(?![A-Za-z0-9_.+-])|/workspace/[A-Za-z0-9_.+-]+__[A-Za-z0-9_.+-]+(?![A-Za-z0-9_.+-])"
)


def repo_path(target: str, cwd: str | None = "", root: str = "") -> str | None:
    """The checkout path ("" for its root) a command names from directory `cwd`, or None when
    it lies outside the checkout or `cwd` is not known.

    `cwd` is a checkout path, or an absolute path for a directory outside it. An absolute
    target is inside the checkout only under `root` (the directory the session's transcript
    shows the checkout at) or a sandbox checkout root; anything else - /tmp, a build tree
    elsewhere on the machine - is outside, whatever the files it names are called.
    """
    if not target.startswith("/"):
        if cwd is None:
            return None
        if cwd.startswith("/"):
            target = f"{cwd.rstrip('/')}/{target}"
    if target.startswith("/"):
        target = posixpath.normpath(target)
        prefix = root if root and (target == root or target.startswith(root + "/")) else None
        if prefix is None and (match := CHECKOUT_ROOT.match(target)):
            prefix = match.group(0)
        if prefix is None:
            return None
        target, cwd = target[len(prefix) :], ""
    parts = cwd.split("/") if cwd else []
    for segment in target.split("/"):
        if segment == "..":
            if not parts:
                return None
            parts.pop()
        elif segment not in ("", "."):
            parts.append(segment)
    return "/".join(parts)


# the one directory outside the checkout whose entries are all the session's own
TMP = "/tmp"


def session_path(target: str, cwd: str | None = "", root: str = "") -> str | None:
    """The path a session's overlay keys a file by: its checkout path (see `repo_path`), or
    for a file outside the checkout its absolute path; None when `cwd` is not known."""
    inside = repo_path(target, cwd, root)
    if inside is not None:
        return inside
    if target.startswith("/"):
        return "/" + posixpath.normpath(target).lstrip("/")
    if cwd and cwd.startswith("/"):
        return "/" + posixpath.normpath(f"{cwd}/{target}").lstrip("/")
    return None


def expand_braces(word: str) -> list[str] | None:
    """The words bash makes of one with a brace list (`a{b,c}`) or sequence (`f{1..3}`); None
    when that is more than `_MAX_BRACE_WORDS` words."""
    match = re.search(r"\{([^{}]*)\}", word)
    if match is None:
        return [word]
    body, head, tail = match.group(1), word[: match.start()], word[match.end() :]
    sequence = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", body)
    if sequence:
        first, last = int(sequence.group(1)), int(sequence.group(2))
        if abs(last - first) >= _MAX_BRACE_WORDS:
            return None
        step = 1 if last >= first else -1
        items = [str(n) for n in range(first, last + step, step)]
    elif "," in body:
        items = body.split(",")
    else:
        return [word]
    out: list[str] = []
    for item in items:
        expanded = expand_braces(head + item + tail)
        if expanded is None or len(out) + len(expanded) > _MAX_BRACE_WORDS:
            return None
        out += expanded
    return out


@lru_cache(maxsize=512)
def _glob_re(pattern: str) -> re.Pattern:
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            close = pattern.find("]", i)
            if close == -1:
                out.append(re.escape(c))
            else:
                out.append(pattern[i : close + 1])
                i = close + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def shell_glob(path: str, pattern: str) -> bool:
    return bool(_glob_re(pattern).match(path))


_SAFE_NAME = re.compile(r"[A-Za-z0-9%+,\-./:=@^_]+")


def _quotef(name: str) -> str:
    """A file name as coreutils prints it in a message: bare when it needs no quoting."""
    return name if _SAFE_NAME.fullmatch(name) else _quote(name)


def _quote(name: str) -> str:
    if "'" in name:
        raise _Decline("file name with a quote")
    return f"'{name}'"


# the directories git keeps a checkout's history in, and file names found there
_GIT_INTERNALS = (
    ".git", ".git/HEAD", ".git/config", ".git/description", ".git/index", ".git/packed-refs",
    ".git/ORIG_HEAD", ".git/FETCH_HEAD", ".git/COMMIT_EDITMSG", ".git/hooks",
    ".git/hooks/pre-commit.sample", ".git/info", ".git/info/exclude", ".git/logs", ".git/logs/HEAD",
    ".git/logs/refs/heads/main", ".git/logs/refs/heads/master", ".git/objects",
    ".git/objects/pack", ".git/objects/pack/pack-0a1b.pack", ".git/objects/pack/pack-0a1b.idx",
    ".git/objects/pack/pack-0a1b.rev", ".git/objects/info", ".git/objects/info/packs",
    ".git/objects/0a", ".git/objects/0a/1b2c3d4e", ".git/refs", ".git/refs/heads",
    ".git/refs/heads/main", ".git/refs/heads/master", ".git/refs/tags", ".git/branches",
    ".git/shallow", ".git/modules",
)  # fmt: skip
_GIT_DIRECTORIES = {
    ".git", ".git/hooks", ".git/info", ".git/logs", ".git/objects", ".git/objects/pack",
    ".git/objects/info", ".git/objects/0a", ".git/refs", ".git/refs/heads", ".git/refs/tags",
    ".git/branches", ".git/modules",
}  # fmt: skip


class _Tree:
    """The checkout as a search sees it: the files present and how to read them, the
    directories (those holding files, those the overlay made, and `.git`), and, through the
    overlay, which paths may or may not exist."""

    def __init__(
        self, listing: list[str], read_file, size_file, overlay, root: str = "", terminal=False
    ):
        self.listing = listing
        # whether the output goes to the terminal the session's commands print on
        self.terminal = terminal
        # the checkout's absolute root as the session shows it
        self.root = root
        self.text = read_file
        self.size_file = size_file
        self.overlay = overlay
        self._files: set[str] | None = None
        self._children: dict[str, dict[str, bool]] | None = None

    def kind(self, path: str) -> str | None:
        """ "file", "dir", or None for a path that is absent; a path that may or may not exist,
        or lies in `.git`, declines the run."""
        if path == ".git" or not path:
            return "dir"
        if path.startswith(".git/"):
            raise _Decline("inside .git")
        if self.overlay is not None:
            kind = self.overlay.kind(path)
            if kind == "unknown":
                raise _Decline(f"{path} may or may not exist")
            return {"file": "file", "directory": "dir"}.get(kind)
        if self._files is None:
            self._files = set(self.listing)
        if path in self._files:
            return "file"
        parent, _, name = path.rpartition("/")
        return "dir" if self.children(parent).get(name) else None

    def absent_reason(self, path: str) -> str:
        """The error for an absent path: `Not a directory` when a parent of it is a file."""
        parts = path.split("/")
        if any(self.kind("/".join(parts[:end])) == "file" for end in range(1, len(parts))):
            return "Not a directory"
        return "No such file or directory"

    def children(self, directory: str) -> dict[str, bool]:
        """A directory's entries, each with whether it is a directory."""
        if self.overlay is not None:
            self.check_walk(directory)
            entries = dict(self.overlay.entries(directory))
        else:
            if self._children is None:
                self._children = {}
                for path in self.listing:
                    parts = path.split("/")
                    for depth in range(len(parts)):
                        entry = self._children.setdefault("/".join(parts[:depth]), {})
                        is_dir = depth < len(parts) - 1
                        entry[parts[depth]] = entry.get(parts[depth], False) or is_dir
            entries = dict(self._children.get(directory, {}))
        if not directory:
            entries[".git"] = True
        return entries

    def pycache_in(self, directory: str) -> str | None:
        """A module beside which a `__pycache__` may be (see `Overlay.pycache_in`)."""
        return self.overlay.pycache_in(directory) if self.overlay is not None else None

    def check_walk(self, path: str) -> None:
        if self.overlay is not None and self.overlay.doubt(path):
            raise _Decline(f"entries of {path or '.'} are not known")

    def size(self, path: str) -> int:
        if self.size_file is not None and (size := self.size_file(path)) is not None:
            return size
        text = self.text(path)
        if text is None:
            raise _Decline(f"size of {path}")
        return len(text.encode("utf-8", "replace"))

    def walk(self, start: str, depth: int = 0):
        """(path, is_dir, depth, internal) for `start` and everything under it, depth-first, a
        directory's entries sorted. `.git`'s entries are not known: in their place come the
        files and directories git keeps there (`_GIT_INTERNALS`), marked internal."""
        is_dir = self.kind(start) == "dir"
        yield start, is_dir, depth, False
        if start == ".git":
            for internal in _GIT_INTERNALS[1:]:
                yield internal, internal in _GIT_DIRECTORIES, depth + internal.count("/"), True
            return
        if not is_dir:
            return
        prefix = f"{start}/" if start else ""
        for name, child_is_dir in sorted(self.children(start).items()):
            if child_is_dir:
                yield from self.walk(prefix + name, depth + 1)
            else:
                yield prefix + name, False, depth + 1, False

    def resolve(self, target: str) -> str:
        # outside the checkout only the session's own files are known, which only it records
        resolve = session_path if self.overlay is not None else repo_path
        path = resolve(target, "", self.root)
        if path is None:
            raise _Decline(f"{target} is outside the checkout")
        return path

    def expand(self, targets: list[str], plan: SearchPlan) -> list[str]:
        """The operands after the shell expands the unquoted globs among them; one that
        matches nothing stays as written."""
        out: list[str] = []
        for target in targets:
            if target not in plan.globbed or not any(char in target for char in "*?["):
                out.append(target)
                continue
            if target.startswith("/") or ".." in target.split("/"):
                raise _Decline("glob outside the working directory")
            prefix = "./" if target.startswith("./") else ""
            matches = self._glob(target[len(prefix) :])
            out += [prefix + match for match in matches] or [target]
        return out

    def _glob(self, pattern: str) -> list[str]:
        current = [""]
        segments = [segment for segment in pattern.split("/") if segment]
        for position, segment in enumerate(segments):
            last = position == len(segments) - 1
            following: list[str] = []
            for directory in current:
                joined = f"{directory}/{segment}" if directory else segment
                if not any(char in segment for char in "*?["):
                    if self.kind(joined) is not None:
                        following.append(joined)
                    continue
                for name, is_dir in sorted(self.children(directory).items()):
                    hidden = name.startswith(".") and not segment.startswith(".")
                    if not hidden and fnmatch.fnmatchcase(name, segment) and (last or is_dir):
                        following.append(f"{directory}/{name}" if directory else name)
            current = following
        if pattern.endswith("/"):
            current = [path for path in current if self.kind(path) == "dir"]
        return current


_GENERATED_PATH = re.compile(
    r"(^|/)(node_modules|\.venv|venv|site-packages|dist-packages|build|dist|target|out|"
    r"__pycache__|\.pytest_cache|\.tox|\.eggs|[^/]+\.egg-info|coverage|htmlcov)(/|$)"
    r"|\.(pyc|pyo|so|o|a|class|jar|whl|log)$"
)


def _missing(run: _Run, tree: _Tree, path: str, template: str, returncode: int, **fields) -> None:
    """The message (`template` with `{why}` and `fields`) for an operand that is absent; one a
    build could have made is not claimed."""
    if _GENERATED_PATH.search(path):
        raise _Decline(f"{path} may be a build product")
    run.missing.append(path)
    run.error(template.format(why=tree.absent_reason(path), **fields), returncode)


def parse_search(cmd: str) -> SearchPlan | ParseFailure:
    parsed = parse_command(cmd or "")
    if isinstance(parsed, Unsupported):
        return ParseFailure("unsupported_shell", parsed.reason)
    if len(parsed.stages) != 1:
        return ParseFailure("unsupported_shell", "several stages")
    commands = parsed.stages[0].pipeline
    plan = SearchPlan()
    words_of: list[list[str]] = []
    for position, command in enumerate(commands):
        if command.expands or command.assignments:
            return ParseFailure("unsupported_shell", "expansion")
        failure = _redirects(command, plan, position, len(commands))
        if failure:
            return failure
        words: list[str] = []
        for word in command.words:
            expanded = expand_braces(word.text) if word.braces else [word.text]
            if expanded is None:
                return ParseFailure("unsupported_shell", "brace expansion")
            if word.globs:
                plan.globbed.update(expanded)
            words += expanded
        if command.heredoc is not None:
            if (
                position
                or words[:1] != ["cat"]
                or len(words) > 1
                or command.heredoc.literal is None
            ):
                return ParseFailure("unsupported_form", "heredoc")
            plan.literal = command.heredoc.literal
        words_of.append(words)
    try:
        rest = _parse_head(words_of, plan)
        for words in rest:
            stage = parse_pipe_stage(words)
            if isinstance(stage, ParseFailure):
                return stage
            plan.pipeline.append(stage)
    except _Decline as declined:
        return ParseFailure("unsupported_form", declined.detail)
    return plan


def _redirects(
    command: SimpleCommand, plan: SearchPlan, position: int, count: int
) -> ParseFailure | None:
    for redirect in command.redirects:
        target = redirect.target.text
        if redirect.fd == 2 and redirect.mode in ("write", "append") and target == "/dev/null":
            plan.errors_dropped = True
        elif redirect.fd == 2 and redirect.mode == "duplicate" and target == "1":
            if position != 0:
                return ParseFailure("unsupported_shell", "redirect")
            plan.errors_piped = count > 1
        elif redirect.fd == 1 and redirect.mode == "write" and target == "/dev/null":
            if position != count - 1:
                return ParseFailure("unsupported_shell", "redirect")
            plan.output_dropped = True
        elif redirect.mode == "read" and position == 0 and command.words[:1]:
            return ParseFailure("unsupported_shell", "input redirect")
        else:
            return ParseFailure("unsupported_shell", "output redirected to a file")
    return None


_READ_CMDS = ("cat", "nl", "head", "tail", "sed", "awk", "wc")


def _parse_head(commands: list[list[str]], plan: SearchPlan) -> list[list[str]]:
    """Parse the pipeline's first command into `plan`; the pipe stages left to parse."""
    head, rest = commands[0], commands[1:]
    name = head[0] if head else ""
    plan.tool = name
    if plan.literal is not None:
        return rest
    if name in _READ_CMDS:
        _parse_read(head, plan)
    elif name == "grep":
        plan.grep = GrepOptions()
        plan.targets += _parse_grep(head, plan.grep)
    elif name == "ls":
        _parse_ls(head, plan)
    elif name == "echo":
        _parse_echo(head, plan)
    elif name == "printf":
        plan.literal = printf_output(head[1:])
        if plan.literal is None:
            raise _Decline("printf format")
    elif name == "pwd":
        if len(head) > 1:
            raise _Decline(f"pwd {head[1]}")
        plan.print_cwd = True
    elif name == "find":
        _parse_find(head, plan)
        if rest and rest[0][:1] == ["xargs"]:
            _parse_xargs(rest[0], plan)
            rest = rest[1:]
    else:
        raise _Decline(f"not a search: {name}")
    return rest


def _parse_grep(tokens: list[str], options: GrepOptions) -> list[str]:
    """Read grep's options into `options`; its operands (the files it searches)."""
    patterns: list[str] = []
    operands: list[str] = []
    long_flags = {
        "--recursive": "recursive", "--line-number": "line_numbers",
        "--files-with-matches": "files_only", "--files-without-match": "files_without",
        "--ignore-case": "ignore_case", "--invert-match": "invert", "--count": "count_only",
        "--word-regexp": "word", "--line-regexp": "whole_line", "--quiet": "quiet",
        "--silent": "quiet", "--no-messages": "no_messages", "--extended-regexp": "extended",
        "--fixed-strings": "fixed", "--text": "binary_text",
    }  # fmt: skip
    letters = {"r": "recursive", "R": "recursive", "n": "line_numbers", "l": "files_only",
               "L": "files_without", "i": "ignore_case", "w": "word", "x": "whole_line",
               "c": "count_only", "E": "extended", "F": "fixed", "s": "no_messages",
               "a": "binary_text", "v": "invert", "q": "quiet", "I": "skip_binary"}  # fmt: skip
    index = 1
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token == "--":
            operands += tokens[index:]
            break
        if token.startswith("--"):
            name, equals, value = token.partition("=")
            if name in long_flags and not equals:
                setattr(options, long_flags[name], True)
            elif name in ("--include", "--exclude", "--exclude-dir") and equals:
                getattr(options, name[2:].replace("-", "_")).append(value)
            elif name == "--regexp" and equals:
                patterns.append(value)
            elif name == "--binary-files" and value == "text":
                options.binary_text = True
            elif name not in ("--color", "--colour"):
                raise _Decline(f"grep {token}")
        elif token.startswith("-") and len(token) > 1:
            if token[1:].isdigit():
                options.before = options.after = int(token[1:])
                continue
            chars = token[1:]
            for position, char in enumerate(chars):
                if char in "ABCme":
                    value = chars[position + 1 :]
                    if not value:
                        if index >= len(tokens):
                            raise _Decline(f"-{char} without a value")
                        value, index = tokens[index], index + 1
                    if char == "e":
                        patterns.append(value)
                    elif not value.isdigit():
                        raise _Decline(f"grep -{char} {value}")
                    elif char == "m":
                        options.max_count = int(value)
                    else:
                        if char in "AC":
                            options.after = int(value)
                        if char in "BC":
                            options.before = int(value)
                    break
                if char == "h":
                    options.with_filename = False
                elif char == "H":
                    options.with_filename = True
                elif char in letters:
                    setattr(options, letters[char], True)
                else:
                    raise _Decline(f"grep -{char}")
        else:
            operands.append(token)
    if not patterns:
        if not operands:
            raise _Decline("grep without a pattern")
        patterns.append(operands.pop(0))
    options.patterns = [line for pattern in patterns for line in pattern.split("\n")]
    return operands


# find tests that name what they match against, and whether they ignore case
_FIND_TESTS = {"-name": ("name", False), "-iname": ("name", True), "-path": ("path", False),
               "-ipath": ("path", True), "-wholename": ("path", False)}  # fmt: skip


def _parse_find(tokens: list[str], plan: SearchPlan) -> None:
    """`find START... TESTS`: -name/-iname/-path tests (each possibly negated), `-type f|d`,
    `-maxdepth`, `-mindepth`, `X -prune -o ... -print`, `A -o B` between tests alone, and
    `-exec grep ... {} ;` / `+` or `-exec ls [-l] {} ;`."""
    args = tokens[1:]
    while args and not args[0].startswith(("-", "!", "(")):
        plan.targets.append(args.pop(0))
    if "-exec" in args:
        start = args.index("-exec")
        command, args = args[start + 1 :], args[:start]
        if not command or command[-1] not in (";", "+") or command[-2:-1] != ["{}"]:
            raise _Decline("find -exec form")
        plan.exec_batched = command[-1] == "+"
        words = command[:-2]
        if words[:1] == ["ls"] and all(re.fullmatch(r"-[la1]+", word) for word in words[1:]):
            plan.tool, plan.long = "find_ls", any("l" in word for word in words[1:])
        elif words[:1] == ["grep"]:
            inner = GrepOptions(per_file=not plan.exec_batched)
            if _parse_grep(words, inner) or inner.recursive:
                raise _Decline("find -exec grep with operands")
            plan.grep = inner
        else:
            raise _Decline("find -exec of a command other than grep or ls")
    alternatives = "-o" in [arg for i, arg in enumerate(args) if args[i - 1 : i] != ["-prune"]]
    index, negate = 0, False
    while index < len(args):
        token = args[index]
        value = args[index + 1] if index + 1 < len(args) else None
        if token in ("-not", "!"):
            negate = True
            index += 1
            continue
        if token in _FIND_TESTS and value is not None:
            if args[index + 2 : index + 4] == ["-prune", "-o"]:
                plan.pruned.append((_FIND_TESTS[token][0], value.rstrip("/")))
                index += 4
                continue
            kind, fold = _FIND_TESTS[token]
            plan.find_tests.append((kind + ("_fold" if fold else ""), value, negate))
            index += 2
        elif token == "-o" and alternatives and plan.find_tests and not negate:
            index += 1
        elif token == "-type" and value in ("f", "d") and not negate:
            plan.type_filter = value
            index += 2
        elif token in ("-maxdepth", "-mindepth") and value is not None and value.isdigit():
            if token == "-maxdepth":
                plan.max_depth = int(value)
            else:
                plan.min_depth = int(value)
            index += 2
        elif token == "-print" and not negate:
            index += 1
        else:
            raise _Decline(f"find {token}")
        negate = False
    if not plan.targets:
        raise _Decline("find without a starting point")
    if alternatives:
        # `A -o B -o C` of tests alone: an entry is printed when any of them matches
        if plan.type_filter or any(negated for _, _, negated in plan.find_tests):
            raise _Decline("find -o with other tests")
        plan.any_test = True
    if plan.tool == "find_ls" and plan.type_filter == "d":
        raise _Decline("find -exec ls of directories")


def _parse_xargs(tokens: list[str], plan: SearchPlan) -> None:
    """`find ... | xargs [-r] [-I{} | -n1] grep ...`: grep over the files find printed."""
    args = tokens[1:]
    per_file = False
    while args and args[0].startswith("-"):
        option = args.pop(0)
        if option in ("-r", "--no-run-if-empty"):
            plan.xargs_skip_empty = True
        elif option == "-I" and args and args[0] == "{}":
            args.pop(0)
            per_file = True
        elif option == "-I{}":
            per_file = True
        elif option in ("-n1", "-n") and (option == "-n1" or (args and args.pop(0) == "1")):
            per_file = True
        else:
            raise _Decline(f"xargs {option}")
    if per_file:
        if args[-1:] != ["{}"]:
            raise _Decline("xargs -I without a trailing {}")
        args = args[:-1]
        plan.xargs_skip_empty = True
    if args[:1] != ["grep"]:
        raise _Decline("xargs of a command other than grep")
    inner = GrepOptions(per_file=per_file)
    if _parse_grep(args, inner) or inner.recursive:
        raise _Decline("xargs grep with operands")
    plan.grep = inner
    plan.exec_batched = not per_file
    plan.tool = "xargs"


def _parse_ls(tokens: list[str], plan: SearchPlan) -> None:
    for token in tokens[1:]:
        if token in ("--all", "--almost-all"):
            plan.show_hidden = True
            plan.show_dots = token == "--all"
        elif token.startswith("-") and len(token) > 1:
            for char in token[1:]:
                if char not in "1aAl":
                    raise _Decline(f"ls -{char}")
                plan.long = plan.long or char == "l"
                plan.one_per_line = plan.one_per_line or char == "1"
                plan.show_hidden = plan.show_hidden or char in "aA"
                plan.show_dots = plan.show_dots or char == "a"
        else:
            plan.targets.append(token)
    plan.targets = plan.targets or ["."]


def _parse_echo(tokens: list[str], plan: SearchPlan) -> None:
    args, newline = tokens[1:], "\n"
    while args and args[0] in ("-n", "-E"):
        newline = "" if args.pop(0) == "-n" else newline
    if args[:1] == ["-e"]:
        raise _Decline("echo -e")
    plan.literal = " ".join(args) + newline


def printf_output(args: list[str]) -> str | None:
    """The output of `printf FORMAT [ARGS]` for a format of plain text, `%s`, `%%` and the
    escapes \\n, \\t and \\\\; None for anything else."""
    if args and args[0] == "--":
        args = args[1:]
    if not args or re.search(r"%(?![s%])|\\[^nt\\]", args[0]):
        return None
    escapes = {"n": "\n", "t": "\t", "\\": "\\"}
    form = re.sub(r"\\(.)", lambda match: escapes[match.group(1)], args[0])
    pieces = form.split("%s")
    values, slots = args[1:], len(pieces) - 1
    out, index = [], 0
    while True:
        text = pieces[0]
        for position, piece in enumerate(pieces[1:]):
            value = values[index + position] if index + position < len(values) else ""
            text += value + piece
        out.append(text.replace("%%", "%"))
        index += slots
        if not slots or index >= len(values):
            return "".join(out)


# `awk 'NR>=190 && NR<=200' f.py` and `awk 'NR==5'` are line-range reads spelled another way
_AWK_RANGE = re.compile(r"NR\s*(>=|>)\s*(\d+)\s*&&\s*NR\s*(<=|<)\s*(\d+)")
_AWK_LINE = re.compile(r"NR\s*==\s*(\d+)")
_SED_PRINT = re.compile(r"(\d+|\$)(?:,(\d+|\$))?p")


def _parse_read(tokens: list[str], plan: SearchPlan) -> None:
    name, args = tokens[0], tokens[1:]
    plan.read_mode = name
    if name == "awk":
        if not args or args[0].startswith("-"):
            raise _Decline("awk options")
        program = args[0].strip()
        if match := _AWK_RANGE.fullmatch(program):
            first = int(match.group(2)) + (match.group(1) == ">")
            last = int(match.group(4)) - (match.group(3) == "<")
        elif match := _AWK_LINE.fullmatch(program):
            first = last = int(match.group(1))
        else:
            raise _Decline(f"awk program {program[:24]}")
        plan.read_mode, plan.read_range = "lines", (max(first, 1), last)
        plan.targets += args[1:]
        if not plan.targets:
            raise _Decline("awk reading standard input")
        return
    if name == "sed":
        if args[:1] != ["-n"] or len(args) < 3:
            raise _Decline("sed without -n")
        match = _SED_PRINT.fullmatch(args[1])
        if match is None:
            raise _Decline(f"sed script {args[1]}")
        first = -1 if match.group(1) == "$" else int(match.group(1))
        last = match.group(2)
        last = first if last is None else (-1 if last == "$" else int(last))
        if first == 0:
            raise _Decline("sed line 0")
        plan.read_mode, plan.read_range = "sed", (first, last)
        plan.targets += args[2:]
        return
    if name in ("head", "tail"):
        count, rest = _count_option(args, bare=False)
        for token in rest:
            if token.startswith("-"):
                raise _Decline(f"{name} {token}")
            plan.targets.append(token)
        plan.read_mode, digits = _head_tail(name, count)
        plan.read_range = (int(digits), 0)
    elif name == "wc":
        letters = "".join(arg[1:] for arg in args if arg.startswith("-") and len(arg) > 1)
        if set(letters) - set("lwc"):
            raise _Decline(f"wc -{letters}")
        plan.counts = "".join(c for c in "lwc" if c in letters) or "lwc"
        plan.targets += [arg for arg in args if not arg.startswith("-") or arg == "-"]
    else:
        for token in args:
            if not token.startswith("-") or token == "-":
                plan.targets.append(token)
            elif name == "cat" and set(token[1:]) <= {"n"}:
                plan.number_lines = True
            elif name == "nl" and token == "-ba":
                plan.number_lines = True
            else:
                raise _Decline(f"{name} {token}")
        if name == "nl" and not plan.number_lines:
            raise _Decline("nl without -ba")
    if not plan.targets or "-" in plan.targets:
        raise _Decline(f"{name} reading standard input")


def _count_option(args: list[str], bare: bool) -> tuple[str, list[str]]:
    """head/tail's line count (`-N`, `-n N`, `-nN`; 10 by default) and the other words; with
    `bare`, a number standing alone is a count too."""
    count, rest, index = "10", [], 0
    while index < len(args):
        token = args[index]
        if re.fullmatch(r"-\d+", token):
            count = token[1:]
        elif token == "-n" and index + 1 < len(args):
            count, index = args[index + 1], index + 1
        elif token.startswith("-n") and len(token) > 2:
            count = token[2:]
        elif bare and token.isdigit():
            count = token
        else:
            rest.append(token)
        index += 1
    return count, rest


def _head_tail(name: str, count: str) -> tuple[str, str]:
    """The read mode and the count's digits: `head -n -N` is all but the last N lines
    (head_all_but), `tail -n +N` from line N on (tail_from)."""
    if not re.fullmatch(r"[+-]?\d+", count) or (name == "head" and count[0] == "+"):
        raise _Decline(f"{name} -n {count}")
    sign = count[0] if count[0] in "+-" else ""
    return name + {"-": "_all_but", "+": "_from", "": ""}[sign], count.lstrip("+-")


def parse_pipe_stage(tokens: list[str]) -> Stage | ParseFailure:
    """One pipe stage after the command whose output it filters."""
    name, args = (tokens[0], tokens[1:]) if tokens else ("", [])
    try:
        if name in ("head", "tail"):
            count, rest = _count_option(args, bare=True)
            if rest:
                raise _Decline(" ".join(tokens))
            mode, digits = _head_tail(name, count)
            return Stage(mode, [digits])
        if name == "sort":
            if set(args) - {"-u", "-r", "-n"}:
                raise _Decline(" ".join(tokens))
            return Stage("sort", sorted(set(args)))
        if name == "uniq":
            if set(args) - {"-c"}:
                raise _Decline(" ".join(tokens))
            return Stage("uniq", list(args))
        if name == "wc":
            letters = "".join(arg[1:] for arg in args if arg.startswith("-"))
            if set(letters) - set("lwc") or any(not arg.startswith("-") for arg in args):
                raise _Decline(" ".join(tokens))
            counts = "".join(c for c in "lwc" if c in letters) or "lwc"
            return Stage("wc", [counts])
        if name == "cat":
            if args in ([], ["-"]):
                return Stage("passthrough")
            if args == ["-n"]:
                return Stage("number")
            raise _Decline(" ".join(tokens))
        if name == "sed":
            match = _SED_PRINT.fullmatch(args[1]) if args[:1] == ["-n"] and len(args) == 2 else None
            if match is None or match.group(1) == "0":
                raise _Decline(" ".join(tokens))
            first = match.group(1)
            last = match.group(2) or first
            return Stage("slice", [first, last])
        if name == "grep":
            options = GrepOptions()
            if _parse_grep(tokens, options) or options.recursive:
                raise _Decline("grep reading files in a pipe")
            if options.files_only or options.files_without:
                raise _Decline("grep listing files in a pipe")
            if options.invert and (options.before or options.after):
                raise _Decline("grep -v with context")
            _compile(options)
            return Stage("filter", options=options)
    except _Decline as declined:
        return ParseFailure("unsupported_pipe", declined.detail)
    return ParseFailure("unsupported_pipe", name)


class _Matcher:
    """A search's pattern, matched as a UTF-8 locale reads text (a character at a time) and,
    for a line that is not ASCII, as the C locale does (a byte at a time): a line the two
    readings disagree on declines the search, since the sandbox's locale is not known."""

    def __init__(self, source: str, flags: int, classes: bool):
        self.regex = re.compile(source, flags)
        self.bytewise = re.compile(_byte_view(source), flags | re.ASCII)
        # a named class (`[:alpha:]`) takes in the locale's letters, which only it knows
        self.classes = classes

    def search(self, line: str) -> bool:
        hit = bool(self.regex.search(line))
        if not line.isascii() and (
            self.classes or hit != bool(self.bytewise.search(_byte_view(line)))
        ):
            raise _Decline("a match that depends on the locale")
        return hit


def _byte_view(text: str) -> str:
    """The text with each UTF-8 byte as one character."""
    return text.encode("utf-8", "surrogateescape").decode("latin-1")


def _compile(options: GrepOptions) -> _Matcher:
    sources = []
    for pattern in options.patterns:
        if options.fixed:
            source = re.escape(pattern)
        else:
            source = (ere_to_python if options.extended else bre_to_python)(pattern)
            if source is None:
                raise _Decline("character class")
        sources.append(f"(?:{source})")
    source = "|".join(sources)
    if options.whole_line:
        source = f"^(?:{source})$"
    elif options.word:
        source = rf"(?<!\w)(?:{source})(?!\w)"
    try:
        classes = any("[:" in pattern for pattern in options.patterns) and not options.fixed
        return _Matcher(source, re.IGNORECASE if options.ignore_case else 0, classes)
    except re.error as exc:
        message = next((m for needle, m in _GREP_PATTERN_ERROR if needle in str(exc)), None)
        if message is None:
            raise _Decline(f"pattern {exc}") from exc
        raise _PatternError(f"grep: {message}") from exc


class _PatternError(_Decline):
    """A pattern grep itself rejects, with the message it prints."""


# Python's complaint about a pattern, and GNU grep's for the same mistake
_GREP_PATTERN_ERROR = (
    ("missing ), unterminated subpattern", r"Unmatched ( or \("),
    ("unbalanced parenthesis", r"Unmatched ) or \)"),
    ("unterminated character set", "Unmatched [, [^, [:, [., or [="),
    ("bad character range", "Invalid range end"),
)


def run_search(
    plan: SearchPlan,
    read_file,
    listing: list[str],
    size_file=None,
    root: str = "",
    overlay=None,
    terminal: bool = False,
) -> SearchResult | ParseFailure:
    """Run a search against the files of `listing`, read with `read_file` (None for a file whose
    text is unknown). `overlay`, when given, says which paths may or may not exist and which
    directories exist without files: a search whose answer depends on those is not run.
    `terminal` says the output goes to the session's terminal, which `ls` lays names out on."""
    tree = _Tree(listing, read_file, size_file, overlay, root, terminal)
    try:
        run = _run(plan, tree)
        return _finish(plan, run)
    except _Decline as declined:
        return ParseFailure("unsupported_form", declined.detail)


def _run(plan: SearchPlan, tree: _Tree) -> _Run:
    if plan.print_cwd:
        # the working directory is a fact about the session, not the tree: unknown without a root
        if not tree.root:
            raise _Decline("working directory is unknown")
        run = _Run()
        run.out(tree.root)
        return run
    if plan.literal is not None:
        return _text_run(plan.literal)
    if plan.read_mode:
        return _run_read(plan, tree)
    if plan.tool == "ls":
        return _run_ls(plan, tree)
    if plan.tool in ("find", "xargs", "find_ls"):
        return _run_find(plan, tree)
    targets = tree.expand(plan.targets, plan)
    implicit = not targets and plan.grep.recursive
    return _run_grep(plan.grep, tree, targets or (["."] if implicit else []), implicit)


def _text_run(text: str) -> _Run:
    lines, open_end = _file_lines(text)
    return _Run([(line, False) for line in lines], open_end=open_end)


def _finish(plan: SearchPlan, run: _Run) -> SearchResult:
    """The terminal's text: messages where they go, and output through the pipe stages."""
    lines = run.lines
    if plan.errors_piped:
        piped, shown = [line for line, _ in lines], []
    else:
        piped = [line for line, error in lines if not error]
        shown = [line for line, error in lines if error and not plan.errors_dropped]
    open_end = run.open_end and bool(piped) and lines[-1][1] is False
    returncode = run.returncode
    if plan.pipeline:
        piped, open_end, returncode = _through(piped, open_end, plan.pipeline, run.unordered)
        output = [*shown, *([] if plan.output_dropped else piped)]
    elif plan.output_dropped:
        output = shown
    else:
        output = [line for line, error in lines if not (error and plan.errors_dropped)]
    return SearchResult(
        output="\n".join(output),
        empty=not output,
        missing=run.missing,
        returncode=returncode,
        open_end=open_end and bool(output),
    )


# pipe stages that keep their input's order, so that an unordered input stays unordered
_ORDER_KEEPING = {"filter", "wc", "passthrough"}


def _through(
    lines: list[str], open_end: bool, pipeline: list[Stage], unordered: bool
) -> tuple[list[str], bool, int]:
    returncode = 0
    for stage in pipeline:
        # taking at least as many lines as there are keeps them all, in whatever order
        keeps_all = stage.name in ("head", "tail") and int(stage.args[0]) >= len(lines)
        if unordered and not keeps_all and stage.name not in _ORDER_KEEPING | {"sort"}:
            raise _Decline(f"{stage.name} over lines in filesystem order")
        unordered = unordered and stage.name != "sort"
        lines, open_end, returncode = _stage(stage, lines, open_end)
    return lines, open_end, returncode


def apply_pipeline(lines: list[str], pipeline: list[Stage]) -> list[str]:
    """The lines after pipe stages (for a command whose output is known already)."""
    return _through(lines, False, pipeline, False)[0]


def _stage(stage: Stage, lines: list[str], open_end: bool) -> tuple[list[str], bool, int]:
    """One pipe stage: its output lines, whether they lack a final line break, its status."""
    name, args = stage.name, stage.args
    keeps_end = False
    if name == "passthrough":
        return lines, open_end, 0
    if name == "head":
        out, keeps_end = lines[: int(args[0])], int(args[0]) >= len(lines)
    elif name == "head_all_but":
        out, keeps_end = lines[: max(0, len(lines) - int(args[0]))], int(args[0]) == 0
    elif name == "tail":
        out, keeps_end = (lines[-int(args[0]) :] if int(args[0]) else []), True
    elif name == "tail_from":
        out, keeps_end = lines[max(int(args[0]), 1) - 1 :], True
    elif name == "slice":
        first = len(lines) if args[0] == "$" else int(args[0])
        last = len(lines) if args[1] == "$" else int(args[1])
        out = lines[first - 1 : max(first, last)] if first <= len(lines) else []
        keeps_end = max(first, last) >= len(lines)
    elif name == "number":
        out, keeps_end = [f"{n:>6}\t{line}" for n, line in enumerate(lines, 1)], True
    elif name == "sort":
        key = _numeric_key if "-n" in args else None
        out = sorted(set(lines) if "-u" in args else lines, key=key, reverse="-r" in args)
    elif name == "uniq":
        groups: list[list] = []
        for line in lines:
            if groups and groups[-1][0] == line:
                groups[-1][1] += 1
            else:
                groups.append([line, 1])
        out = [f"{count:>7} {line}" if args else line for line, count in groups]
    elif name == "wc":
        text = "\n".join(lines) + ("" if open_end or not lines else "\n")
        return [_wc_row(text, args[0], None, 7 if len(args[0]) > 1 else 1)], False, 0
    elif name == "filter":
        return _filter(stage.options, lines)
    else:
        raise _Decline(f"pipe stage {name}")
    return out, open_end and keeps_end and bool(out), 0


def _numeric_key(line: str) -> tuple[float, str]:
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", line)
    return (float(match.group(1)) if match else 0.0, line)


def _filter(options: GrepOptions, lines: list[str]) -> tuple[list[str], bool, int]:
    regex = _compile(options)
    hits = [i for i, line in enumerate(lines) if regex.search(line) != options.invert]
    if options.max_count:
        hits = hits[: options.max_count]
    if options.quiet:
        return [], False, 0 if hits else 1
    if options.count_only:
        return [str(len(hits))], False, 0 if hits else 1
    prefix = "(standard input)" if options.with_filename else ""
    shown = _context_lines(lines, hits, options.before, options.after, prefix, options.line_numbers)
    return shown, False, 0 if hits else 1


def _context_lines(
    lines: list[str], hits: list[int], before: int, after: int, prefix: str, numbered: bool
) -> list[str]:
    """The lines grep prints for these hits: with context and `--` between groups when asked,
    `N:` / `N-` prefixes when numbered, and `prefix` (the file name) before each."""
    hit_set = set(hits)
    wanted = sorted(
        {j for i in hits for j in range(max(0, i - before), min(len(lines), i + after + 1))}
    )
    out: list[str] = []
    previous: int | None = None
    for index in wanted:
        if (before or after) and previous is not None and index > previous + 1:
            out.append("--")
        separator = ":" if index in hit_set else "-"
        name = f"{prefix}{separator}" if prefix else ""
        number = f"{index + 1}{separator}" if numbered else ""
        out.append(f"{name}{number}{lines[index]}")
        previous = index
    return out


def _file_lines(text: str) -> tuple[list[str], bool]:
    """A file's lines, and whether its last line lacks a line break."""
    if not text:
        return [], False
    return text.removesuffix("\n").split("\n"), not text.endswith("\n")


_MISSING_READ = {
    "cat": ("cat: {q}: {why}", 1),
    "nl": ("nl: {q}: {why}", 1),
    "head": ("head: cannot open {Q} for reading: {why}", 1),
    "tail": ("tail: cannot open {Q} for reading: {why}", 1),
    # verified against GNU sed: it exits 2 when it cannot open its input
    "sed": ("sed: can't read {q}: {why}", 2),
    "wc": ("wc: {q}: {why}", 1),
}
_DIRECTORY_READ = {
    "cat": "cat: {q}: Is a directory",
    "nl": "nl: {q}: Is a directory",
    "head": "head: error reading {Q}: Is a directory",
    "tail": "tail: error reading {Q}: Is a directory",
}


def _run_read(plan: SearchPlan, tree: _Tree) -> _Run:
    """A read of files. `cat` and `nl` print each file's text and each message in the order of
    their operands; the other reads select from what they read, so a message among their
    output is not placed here."""
    run = _Run()
    tool = plan.tool
    targets = tree.expand(plan.targets, plan)
    texts: list[tuple[str, str]] = []
    stream = ""
    printed = 0
    headed = False
    for target in targets:
        path = tree.resolve(target)
        kind = tree.kind(path)
        if kind == "file":
            text = tree.text(path)
            if text is None:
                raise _Decline(f"{path} has unknown text")
            texts.append((target, text))
            stream += text
            if len(stream) > MAX_TEXT_CHARS:
                raise _Decline("more text than is answered here")
            continue
        if tool == "awk" or (kind == "dir" and tool not in _DIRECTORY_READ):
            raise _Decline(f"{tool} of a missing file or a directory")
        if texts and tool not in ("cat", "nl", "wc"):
            raise _Decline(f"{tool} message after output")
        if stream and not stream.endswith("\n"):
            raise _Decline("message after an unterminated line")
        if tool != "wc":
            printed = _emit(run, plan, stream, printed)
            stream = ""
        if kind is None:
            template, code = _MISSING_READ[tool]
            _missing(run, tree, path, template, code, q=_quotef(target), Q=_quote(target))
        else:
            if tool in ("head", "tail") and len(targets) > 1:
                # the directory opens, so its header is printed before the read fails
                if headed:
                    run.out("")
                run.out(f"==> {target} <==")
                headed = True
            run.error(_DIRECTORY_READ[tool].format(q=_quotef(target), Q=_quote(target)), 1)
    if tool == "wc":
        return _run_wc(plan, run, texts, len(targets))
    if tool in ("head", "tail") and len(targets) > 1:
        for position, (target, text) in enumerate(texts):
            if position or headed:
                run.out("")
            run.out(f"==> {target} <==")
            selected, open_end = _select(plan, *_file_lines(text))
            for line in selected:
                run.out(line)
            if open_end and position < len(texts) - 1:
                raise _Decline("banner after an unterminated line")
            run.open_end = open_end
        return run
    _emit(run, plan, stream, printed)
    return run


def _emit(run: _Run, plan: SearchPlan, stream: str, numbered: int) -> int:
    """Print a read's selection of `stream`, numbering on from `numbered` printed lines; the
    count of lines printed so far."""
    selected, open_end = _select(plan, *_file_lines(stream))
    for number, line in enumerate(selected, numbered + 1):
        run.out(f"{number:>6}\t{line}" if plan.number_lines else line)
    if selected:
        run.open_end = open_end
    return numbered + len(selected)


def _select(plan: SearchPlan, lines: list[str], open_end: bool) -> tuple[list[str], bool]:
    """The lines a read prints of its input, and whether they lack a final line break."""
    mode, bounds = plan.read_mode, plan.read_range
    if mode in ("cat", "nl"):
        return lines, open_end
    count = len(lines)
    if mode == "head":
        first, last = 1, bounds[0]
    elif mode == "head_all_but":
        first, last = 1, count - bounds[0]
    elif mode == "tail":
        first, last = count - bounds[0] + 1, count
    elif mode == "tail_from":
        first, last = bounds[0], count
    elif mode == "sed":
        first = count if bounds[0] == -1 else bounds[0]
        last = count if bounds[1] == -1 else bounds[1]
        # a range whose end comes before its start prints its first line only
        last = max(first, last)
    else:
        first, last = bounds
    first = max(first, 1)
    selected = lines[first - 1 : max(last, 0)]
    return selected, open_end and bool(selected) and last >= count


def _run_wc(plan: SearchPlan, run: _Run, texts: list[tuple[str, str]], operands: int) -> _Run:
    """`wc` of files: one row per file and a total for several, each count padded to the
    width of the files' total size."""
    total = sum(len(text.encode("utf-8", "replace")) for _, text in texts)
    width = 1 if len(plan.counts) == 1 and operands == 1 else len(str(total))
    for target, text in texts:
        run.out(_wc_row(text, plan.counts, target, width))
    if operands > 1:
        run.out(_wc_row("".join(text for _, text in texts), plan.counts, "total", width))
    return run


def _wc_row(text: str, counts: str, name: str | None, width: int) -> str:
    values = {
        "l": text.count("\n"),
        "w": len(text.split()),
        "c": len(text.encode("utf-8", "replace")),
    }
    row = " ".join(f"{values[c]:>{width}}" for c in counts)
    return f"{row} {name}" if name is not None else row


_LS_DIR_MODE = "drwxr-xr-x"
_LS_FILE_MODE = "-rw-r--r--"
_LS_OWNER = "root root"
_LS_DATE = "Jan  3 20:00"
_LS_DIR_SIZE = 4096
_LS_BLOCK = 4096


def _long_row(name: str, is_dir: bool, size: int) -> str:
    mode, links = (_LS_DIR_MODE, 2) if is_dir else (_LS_FILE_MODE, 1)
    return f"{mode} {links} {_LS_OWNER} {size} {_LS_DATE} {name}"


def _long_rows(tree: _Tree, shown: list[tuple[str, str, bool]], total: bool = False) -> list[str]:
    """`ls -l` rows, each column as wide as its widest value, as GNU `ls` aligns them: a
    directory links to itself, its parent's entry, and each subdirectory's `..`."""
    rows, blocks = [], 0
    for name, path, is_dir in shown:
        if is_dir:
            subdirectories = sum(tree.children(path).values()) if tree.kind(path) == "dir" else 0
            rows.append((_LS_DIR_MODE, 2 + subdirectories, _LS_DIR_SIZE, name))
            blocks += _LS_BLOCK // 1024
        else:
            size = tree.size(path)
            rows.append((_LS_FILE_MODE, 1, size, name))
            blocks += -(-size // _LS_BLOCK) * (_LS_BLOCK // 1024)
    links = max((len(str(row[1])) for row in rows), default=1)
    sizes = max((len(str(row[2])) for row in rows), default=1)
    lines = [
        f"{mode} {count:>{links}} {_LS_OWNER} {size:>{sizes}} {_LS_DATE} {name}"
        for mode, count, size, name in rows
    ]
    return [f"total {blocks}", *lines] if total else lines


def _run_ls(plan: SearchPlan, tree: _Tree) -> _Run:
    """`ls` of files and directories: messages for missing operands first, then the files
    named, then each directory's entries under a `name:` header when there are several."""
    run = _Run()
    files: list[tuple[str, str]] = []
    directories: list[tuple[str, str]] = []
    targets = tree.expand(plan.targets, plan)
    for target in targets:
        path = tree.resolve(target)
        if plan.long and path.startswith("/"):
            raise _Decline("the dates, sizes and modes of files outside the checkout")
        kind = tree.kind(path)
        if kind is None:
            _missing(run, tree, path, "ls: cannot access {Q}: {why}", 2, Q=_quote(target))
        elif kind == "file":
            files.append((target, path))
        else:
            directories.append((target, path))
    files.sort()
    directories.sort()
    if plan.long:
        rows = _long_rows(tree, [(target, path, False) for target, path in files])
    else:
        rows = _laid_out(plan, tree, [target for target, _ in files])
    for row in rows:
        run.out(row)
    for position, (target, path) in enumerate(directories):
        if position or files:
            run.out("")
        if len(targets) > 1:
            run.out(f"{target}:")
        for line in _directory_rows(plan, tree, path):
            run.out(line)
    return run


_MIN_COLUMN_WIDTH = 3
_TAB = 8


def ls_layouts(names: list[str], widths: tuple[int, ...]) -> dict[int, tuple[str, ...]]:
    """How GNU `ls` lays sorted `names` out on a terminal of each width: as many columns as
    fit, filled top to bottom, padded with tabs and blanks."""
    count = len(names)
    most = min(count, max(widths) // _MIN_COLUMN_WIDTH) if widths else 0
    fitting: dict[int, tuple[list[int], int]] = {}
    for columns in range(1, most + 1):
        rows = -(-count // columns)
        spans = [_MIN_COLUMN_WIDTH] * columns
        for index, name in enumerate(names):
            column = index // rows
            spans[column] = max(spans[column], len(name) + (0 if column == columns - 1 else 2))
        fitting[columns] = (spans, sum(spans))
    rendered: dict[int, tuple[str, ...]] = {}
    out: dict[int, tuple[str, ...]] = {}
    for width in widths:
        columns = max(
            (
                c
                for c, (_, length) in fitting.items()
                if c <= width // _MIN_COLUMN_WIDTH and length < width
            ),
            default=1,
        )
        if columns not in rendered:
            rendered[columns] = _columns_text(names, fitting.get(columns, ([0], 0))[0], columns)
        out[width] = rendered[columns]
    return out


def _columns_text(names: list[str], spans: list[int], columns: int) -> tuple[str, ...]:
    rows = -(-len(names) // columns)
    lines = []
    for row in range(rows):
        line, position = "", 0
        for column, index in enumerate(range(row, len(names), rows)):
            name = names[index]
            line += name
            if index + rows >= len(names):
                break
            start, goal = position + len(name), position + spans[column]
            while start < goal:
                if goal // _TAB > (start + 1) // _TAB:
                    line += "\t"
                    start += _TAB - start % _TAB
                else:
                    line += " "
                    start += 1
            position = goal
        lines.append(line)
    return tuple(lines)


def _laid_out(plan: SearchPlan, tree: _Tree, names: list[str]) -> list[str]:
    """The lines `ls` prints `names` on: one each, or in columns when it writes to a terminal.
    A layout that depends on a terminal width the session has not shown is not answered."""
    widths = getattr(tree.overlay, "columns", None)
    if len(names) < 2 or not tree.terminal or plan.one_per_line or plan.pipeline or not widths:
        return names
    if not all(_SAFE_NAME.fullmatch(name) for name in names):
        raise _Decline("a name ls quotes on a terminal")
    for seen, lines in tree.overlay.layouts:
        widths = tuple(w for w, text in ls_layouts(list(seen), widths).items() if text == lines)
    layouts = set(ls_layouts(names, widths).values())
    if len(layouts) != 1:
        raise _Decline("the terminal width the layout depends on")
    return list(layouts.pop())


def _directory_rows(plan: SearchPlan, tree: _Tree, path: str) -> list[str]:
    if path == ".git" or path.startswith(".git/"):
        raise _Decline("entries of .git")
    tree.check_walk(path)
    if tree.pycache_in(path):
        raise _Decline("a __pycache__ may be there")
    entries = tree.children(path)
    names = sorted(n for n in entries if plan.show_hidden or not n.startswith("."))
    if not plan.long:
        return _laid_out(plan, tree, ([".", ".."] if plan.show_dots else []) + names)
    if plan.show_dots and not path:
        raise _Decline("size of the directory holding the checkout")
    prefix = f"{path}/" if path else ""
    parent = path.rpartition("/")[0]
    shown = [(".", path, True), ("..", parent, True)] if plan.show_dots else []
    shown += [(name, prefix + name, entries.get(name, False)) for name in names]
    return _long_rows(tree, shown, total=True)


def _run_find(plan: SearchPlan, tree: _Tree) -> _Run:
    run = _Run()
    found: list[str] = []
    for target in tree.expand(plan.targets, plan):
        path = tree.resolve(target)
        if tree.kind(path) is None:
            _missing(run, tree, path, "find: {Q}: {why}", 1, Q=_quote(target))
            continue
        entries = _find_entries(plan, tree, path, target)
        run.unordered = run.unordered or len(entries) > 1
        found += entries
        for shown in entries if plan.grep is None else []:
            if plan.tool == "find_ls":
                path = tree.resolve(shown)
                if tree.kind(path) != "file":
                    raise _Decline("find -exec ls of a directory")
                shown = _long_row(shown, False, tree.size(path)) if plan.long else shown
            run.out(shown)
    if plan.grep is not None:
        _grep_found(plan, tree, run, found)
    return run


def _find_entries(plan: SearchPlan, tree: _Tree, start: str, target: str) -> list[str]:
    """What find prints under one starting point, as it prints it. A walk that would print
    an entry git keeps under `.git` is not reproduced."""
    shown: list[str] = []
    skipped: list[str] = []
    for path, is_dir, depth, internal in tree.walk(start):
        if any(path.startswith(s + "/") for s in skipped):
            continue
        rel = path[len(start) :].lstrip("/") if start else path
        if not rel:
            display = target
        else:
            display = target + rel if target.endswith("/") else f"{target}/{rel}"
        if plan.max_depth is not None and depth > plan.max_depth:
            continue
        name = posixpath.basename(display.rstrip("/")) or display
        if any(_find_match(kind, pattern, name, display) for kind, pattern in plan.pruned):
            skipped.append(path)
            continue
        keep = depth >= plan.min_depth and _find_keep(plan, name, display, is_dir)
        if internal and keep:
            raise _Decline("a walk into .git")
        if is_dir and (module := tree.pycache_in(path)):
            # what a __pycache__ beside the modules would add to the walk
            cache = f"{display.rstrip('/')}/__pycache__"
            for entry, is_cache_dir, down in (
                (cache, True, 1),
                (f"{cache}/{module}.cpython-311.pyc", False, 2),
            ):
                entry_name = posixpath.basename(entry)
                if any(
                    _find_match(kind, pattern, entry_name, entry) for kind, pattern in plan.pruned
                ):
                    break
                deep_enough = depth + down >= plan.min_depth
                shallow = plan.max_depth is None or depth + down <= plan.max_depth
                if deep_enough and shallow and _find_keep(plan, entry_name, entry, is_cache_dir):
                    raise _Decline("a __pycache__ may be there")
        if keep and not internal:
            shown.append(display)
    return shown


def _find_keep(plan: SearchPlan, name: str, display: str, is_dir: bool) -> bool:
    if plan.type_filter and (plan.type_filter == "d") != is_dir:
        return False
    results = [
        _find_match(kind, pattern, name, display) != negate
        for kind, pattern, negate in plan.find_tests
    ]
    return any(results) if plan.any_test else all(results)


def _find_match(kind: str, pattern: str, name: str, display: str) -> bool:
    fold = kind.endswith("_fold")
    subject = name if kind.startswith("name") else display
    if fold:
        subject, pattern = subject.lower(), pattern.lower()
    return fnmatch.fnmatchcase(subject, pattern)


def _grep_found(plan: SearchPlan, tree: _Tree, run: _Run, found: list[str]) -> None:
    """`find -exec grep` and `xargs grep` over the paths find printed. grep reports matches in
    each file named; `find` and `xargs` give it the names in find's order."""
    inner = plan.grep
    if plan.tool == "xargs" and any(re.search(r"[\s'\"\\]", shown) for shown in found):
        raise _Decline("xargs splitting a name")
    if plan.tool == "xargs" and not found:
        if plan.xargs_skip_empty:
            return
        # xargs runs grep once without operands; it reads an empty input and finds nothing
        run.returncode = 123
        return
    batches = [[shown] for shown in found] if inner.per_file else [found]
    failed = False
    for batch in batches:
        sub = _run_grep(inner, tree, batch)
        run.lines += sub.lines
        run.missing += sub.missing
        failed = failed or sub.returncode != 0
    if failed and plan.tool == "xargs":
        run.returncode = 123
    elif failed and plan.exec_batched:
        run.returncode = 1


def _run_grep(
    options: GrepOptions, tree: _Tree, targets: list[str], implicit: bool = False
) -> _Run:
    """grep over its operands; `implicit` when `grep -r` was given none and searches `.`,
    printing names without a `./`."""
    run = _Run()
    try:
        regex = _compile(options)
    except _PatternError as rejected:
        run.error(rejected.detail, 2)
        return run
    if not targets:
        raise _Decline("grep reading standard input")
    files: list[tuple[str, str]] = []
    for target in targets:
        path = tree.resolve(target)
        kind = tree.kind(path)
        if kind is None:
            files.append((target, ""))
            continue
        if kind == "dir":
            if not options.recursive:
                files.append((target, "\0dir"))
                continue
            walked = _grep_walk(options, tree, path, "" if implicit else target)
            run.unordered = run.unordered or len(walked) > 1
            files += walked
        else:
            files.append((target, path))
    if len(files) > _MAX_FILES:
        raise _Decline("too many files")
    show_name = options.with_filename
    if show_name is None:
        show_name = len(targets) > 1 or any(
            options.recursive and tree.kind(tree.resolve(t)) == "dir" for t in targets
        )
    matched = errored = False
    for shown, path in files:
        if path == "":
            errored = True
            if not options.no_messages:
                _missing(run, tree, tree.resolve(shown), "grep: {name}: {why}", 2, name=shown)
            continue
        if path == "\0dir":
            errored = True
            if not options.no_messages:
                run.error(f"grep: {shown}: Is a directory", 2)
            # grep goes on to report the directory as a file without matches
            if options.quiet:
                continue
            if options.count_only:
                run.out(f"{shown}:0" if show_name else "0")
            elif options.files_without:
                run.out(shown)
            continue
        text = tree.text(path)
        if text is None:
            raise _Decline(f"{path} has unknown text")
        binary = "\0" in text and not options.binary_text
        if binary and options.skip_binary:
            continue
        lines, _ = _file_lines(text)
        hits = [i for i, line in enumerate(lines) if regex.search(line) != options.invert]
        if options.max_count:
            hits = hits[: options.max_count]
        matched = matched or bool(hits)
        if options.quiet:
            continue
        if options.count_only:
            run.out(f"{shown}:{len(hits)}" if show_name else str(len(hits)))
        elif options.files_only:
            if hits:
                run.out(shown)
        elif options.files_without:
            if not hits:
                run.out(shown)
        elif hits and binary:
            run.lines.append((f"grep: {shown}: binary file matches", True))
        elif hits:
            if (options.before or options.after) and any(not error for _, error in run.lines):
                run.out("--")
            prefix = shown if show_name else ""
            for line in _context_lines(
                lines, hits, options.before, options.after, prefix, options.line_numbers
            ):
                run.out(line)
        if len(run.lines) > _MAX_LINES:
            raise _Decline("too many lines")
    if options.quiet and matched:
        run.returncode = 0
    elif errored:
        run.returncode = 2
    elif options.files_without:
        run.returncode = 0 if any(not error for _, error in run.lines) else 1
    else:
        run.returncode = 0 if matched else 1
    return run


def _grep_walk(options: GrepOptions, tree: _Tree, start: str, target: str) -> list[tuple[str, str]]:
    """The files `grep -r` searches under a directory, each with the name grep prints. A walk
    that would search a file git keeps under `.git` is not reproduced."""
    out: list[tuple[str, str]] = []
    skipped: list[str] = []
    for path, is_dir, _, internal in tree.walk(start):
        if any(path.startswith(s + "/") for s in skipped):
            continue
        name = path.rsplit("/", 1)[-1]
        if is_dir:
            if path != start and any(fnmatch.fnmatchcase(name, g) for g in options.exclude_dir):
                skipped.append(path)
            elif (module := tree.pycache_in(path)) and not any(
                fnmatch.fnmatchcase("__pycache__", g) for g in options.exclude_dir
            ):
                if _grep_selects(options, f"{module}.cpython-311.pyc"):
                    raise _Decline("a __pycache__ may be there")
            continue
        if not _grep_selects(options, name):
            continue
        if internal:
            raise _Decline("a walk into .git")
        rel = path[len(start) :].lstrip("/") if start else path
        out.append((rel if not target else f"{target.rstrip('/')}/{rel}", path))
    return out


def _grep_selects(options: GrepOptions, name: str) -> bool:
    if options.include and not any(fnmatch.fnmatchcase(name, g) for g in options.include):
        return False
    return not any(fnmatch.fnmatchcase(name, g) for g in options.exclude)


_QUOTED_SPAN = re.compile(r"'[^']*'|\"[^\"]*\"")


def mask_quoted(text: str) -> str:
    """The text with the inside of every quoted span replaced by its quote character."""
    return _QUOTED_SPAN.sub(lambda m: m.group(0)[0] * len(m.group(0)), text)


def number_lines(text: str, grep_style: bool) -> str:
    lines = text.split("\n")
    if grep_style:
        return "\n".join(f"{n}:{line}" for n, line in enumerate(lines, 1))
    return "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(lines, 1))
