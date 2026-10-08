from __future__ import annotations

import bisect
import copy
import fnmatch
import hashlib
import itertools
import posixpath
import re
import shlex
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from albedo_eval_service.shared.observation_format import (
    NO_OUTPUT_SENTENCE,
    OPENHANDS,
    RETURNCODE,
    SWE_AGENT,
    first_bash_block,
    is_scaffold_truncated,
    observation_body,
    observed_returncode,
)

from .command_search import (
    CHECKOUT_ROOT,
    MAX_TEXT_CHARS,
    TMP,
    ParseFailure,
    expand_braces,
    mask_quoted,
    parse_search,
    printf_output,
    repo_path,
    run_search,
    session_path,
)
from .git_sim import GitState, apply_git_stage, apply_hunks, learn_git, unified_diff
from .script_writes import script_writes
from .sed import sed_edit
from .shell import AND, OR, Redirect, SimpleCommand, Stage, Unsupported, Word, parse_command

_EXACT_OUTPUT = re.compile(r"\A<returncode>-?\d+</returncode>\n<output>\n(.*)</output>\Z", re.S)
_NUMBERED = re.compile(r"^\s*(\d+)\t(.*)$")
_NUMBERED_HIT = re.compile(r"^(\d+):(.*)$")
_SED_RANGE = re.compile(r"^(\d+),(\d+)p$")
_REPORTED_CWD = re.compile(r"^\[Current working directory: (/[^\]]*)\]\s*$", re.M)

# Observation shapes that prove a path exists, all emitted by the OpenHands scaffold.
_LISTING_BLOCK = re.compile(
    r"^Here's the files and directories up to \d+ levels? deep in (\S+?)[,:]", re.M
)
_VIEW_OF = re.compile(r"^\s*Here's the result of running `[^`]+` on (\S+?):\s*$", re.M)
_CREATED_AT = re.compile(r"^File created successfully at:\s*(\S+)\s*$", re.M)
_ENOENT = re.compile(
    r"No such file or directory|cannot access|can't read|can't open file|does not exist"
)
_LISTED_PATH = re.compile(r"^\s{0,4}(/[^\s:]+)\s*$")

FILE, DIRECTORY, UNKNOWN = "file", "directory", "unknown"


@dataclass
class Overlay:
    """The checkout as the session has left it: the base listing with the session's changes.

    A file is present when it is in `base` (the sorted listing at the commit) and not in
    `deleted`, or is in `created`. `content` holds the text of files whose text differs from the
    commit's or was read; `dirty` the files present whose text is unknown. `dirs` holds
    directories the files do not imply (made with mkdir, or emptied), `maybe` paths that may or
    may not exist, and `unsure` directories whose entries are unknown ("" is everything). `cwd`
    is the directory the next command starts in: a checkout path, an absolute path outside it,
    or None when unknown.

    A path outside the checkout is kept by its absolute path. `/tmp` holds only what the
    session put there, until it runs a program this does not follow (`tmp_open`); every other
    directory outside the checkout holds the machine's files, whose entries are unknown, so only
    the files the session wrote there are known.
    """

    base: tuple[str, ...] = ()
    read_base: Callable[[str], str | None] = field(
        default=lambda path: None, repr=False, compare=False
    )
    root: str = ""
    content: dict[str, str] = field(default_factory=dict)
    created: set[str] = field(default_factory=set)
    dirty: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    dirs: set[str] = field(default_factory=set)
    maybe: set[str] = field(default_factory=set)
    unsure: set[str] = field(default_factory=set)
    opaque: list[tuple[str | None, str]] = field(default_factory=list)
    git: GitState = field(default_factory=GitState)
    cwd: str | None = ""
    # a program ran that may have left files in /tmp this does not know of
    tmp_open: bool = False
    # the terminal widths `ls` may lay names out for (None: one name per line), and the layouts
    # recorded observations showed it print: (the names, the lines)
    columns: tuple[int, ...] | None = None
    layouts: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)
    # Python ran, so a `__pycache__` may sit beside any module
    pycache: bool = False
    # the note of the openhands editor whose view a `cat -n` read is (see `core._editor_view`),
    # "" when it is bash's own output
    editor: str = ""
    # what bash puts before its own error messages: `bash: line 1: ` when it runs the command
    # as a script (`bash -c`), `bash: ` in an interactive shell
    errors: str = "bash: "
    # the shell running the commands: `dash` words its own errors differently
    shell: str = "bash"

    def copy(self) -> Overlay:
        return replace(
            self,
            content=dict(self.content),
            created=set(self.created),
            dirty=set(self.dirty),
            deleted=set(self.deleted),
            dirs=set(self.dirs),
            maybe=set(self.maybe),
            unsure=set(self.unsure),
            opaque=list(self.opaque),
            git=copy.deepcopy(self.git),
        )

    def state(self, block: str, referenced: set[str]) -> str:
        parts = [block]
        parts += [cmd for path, cmd in self.opaque if path is None or path in referenced]
        return hashlib.sha1("\n".join(parts).encode("utf-8", "replace")).hexdigest()

    def in_base(self, path: str) -> bool:
        index = bisect.bisect_left(self.base, path)
        return index < len(self.base) and self.base[index] == path

    def has_file(self, path: str) -> bool:
        return path in self.created or (path not in self.deleted and self.in_base(path))

    def files_under(self, directory: str) -> list[str]:
        """The files present under a directory ("" is the whole checkout), sorted."""
        base = self.base
        if directory:
            low = bisect.bisect_left(base, directory + "/")
            # "0" follows "/", so every path under the directory sorts before directory + "0"
            base = base[low : bisect.bisect_left(base, directory + "0")]
        held = [path for path in base if path not in self.deleted] if self.deleted else list(base)
        inside = directory + "/"
        extra = [p for p in self.created if (p.startswith(inside) if directory else p[:1] != "/")]
        return sorted(held + extra) if extra else held

    def listing(self) -> list[str]:
        return self.files_under("")

    def kind(self, path: str) -> str | None:
        """FILE, DIRECTORY, UNKNOWN (may or may not exist), or None when absent."""
        if not path or path == TMP:
            return DIRECTORY
        if self._unsure_over(path) or self._maybe_over(path):
            return UNKNOWN
        if self.has_file(path):
            return FILE
        if path in self.dirs or self._holds(path):
            return DIRECTORY
        if any(entry.startswith(path + "/") for entry in self.maybe) or self.machine(path):
            return UNKNOWN
        parent, _, name = path.rpartition("/")
        if name == "__pycache__" and self.pycache_in(parent):
            return UNKNOWN
        return None

    def doubt(self, path: str) -> bool:
        """Whether what exists at or under `path` is not fully known."""

        def covered(entry: str) -> bool:
            if not path:
                return not entry.startswith("/")
            return entry == path or entry.startswith(path + "/")

        return (
            self.machine(path)
            or self._unsure_over(path)
            or any(map(covered, self.unsure))
            or any(map(covered, self.maybe))
        )

    def entries(self, directory: str) -> dict[str, bool]:
        """The names in a directory, each with whether it is a directory itself."""
        prefix = f"{directory}/" if directory else ""
        names: dict[str, bool] = {}
        made = [d for d in self.dirs if d.startswith(prefix)]
        for path in [*self.files_under(directory), *made]:
            head, _, tail = path[len(prefix) :].partition("/")
            if head:
                names[head] = names.get(head, False) or bool(tail) or path in self.dirs
        return names

    def text(self, path: str) -> str | None:
        """The text of a file present, or None when it is unknown or the file is absent."""
        if path in self.dirty or self.kind(path) != FILE:
            return None
        held = self.content.get(path)
        return held if held is not None or path.startswith("/") else self.read_base(path)

    def read(self, rel_path: str) -> str | None:
        return self.content.get(rel_path)

    def is_dirty(self, rel_path: str) -> bool:
        return rel_path in self.dirty

    def know(self, rel_path: str, text: str) -> None:
        if len(text) > MAX_TEXT_CHARS:
            self.forget(rel_path)
            return
        self.content[rel_path] = text
        self.dirty.discard(rel_path)

    def forget(self, rel_path: str) -> None:
        self.content.pop(rel_path, None)
        self.dirty.add(rel_path)

    def put(self, path: str, text: str | None, certain: bool = True) -> None:
        """A file written at `path` with `text` (None when unknown). An uncertain write leaves
        a file that was present with unknown text, and a path that was not maybe present."""
        if not certain:
            if self.kind(path) == FILE:
                self.forget(path)
            else:
                self.maybe.add(path)
            return
        self.maybe.discard(path)
        self.deleted.discard(path)
        if not self.in_base(path):
            self.created.add(path)
        if text is None:
            self.forget(path)
        else:
            self.know(path, text)

    def drop(self, path: str, certain: bool = True) -> None:
        """The file or directory tree at `path` removed; uncertainly, all of it may be."""

        def within(entry: str) -> bool:
            return not path or entry == path or entry.startswith(path + "/")

        doomed = self.files_under(path) + ([path] if self.has_file(path) else [])
        if not certain:
            if self.kind(path) is not None:
                self.maybe.add(path)
            self.maybe.update(doomed, [d for d in self.dirs if within(d)])
            return
        for gone in doomed:
            self.content.pop(gone, None)
            self.dirty.discard(gone)
            self.created.discard(gone)
            if self.in_base(gone):
                self.deleted.add(gone)
        self.dirs = {d for d in self.dirs if not within(d)}
        self.maybe = {m for m in self.maybe if not within(m)}
        self.unsure = {u for u in self.unsure if not within(u)}
        if parent := path.rpartition("/")[0]:
            self.dirs.add(parent)

    def make_dir(self, path: str, certain: bool = True) -> None:
        if certain:
            self.maybe.discard(path)
            self.dirs.add(path)
        else:
            self.maybe.add(path)

    def pycache_in(self, directory: str) -> str | None:
        """A module of the directory's own, beside which a `__pycache__` may be, once Python ran;
        None when there is none."""
        if not self.pycache:
            return None
        start = len(directory) + 1 if directory else 0
        return next(
            (
                path[start:-3]
                for path in self.files_under(directory)
                if path.endswith(".py") and "/" not in path[start:]
            ),
            None,
        )

    def machine(self, path: str) -> bool:
        """Whether a path lies outside the checkout where files this does not know of may be."""
        inside_tmp = path == TMP or path.startswith(TMP + "/")
        return path.startswith("/") and (self.tmp_open or not inside_tmp)

    def _unsure_over(self, path: str) -> bool:
        return any(not u or path == u or path.startswith(u + "/") for u in self.unsure)

    def _maybe_over(self, path: str) -> bool:
        """Whether the path, or a directory on the way to it, may or may not exist."""
        return any(path == m or path.startswith(m + "/") for m in self.maybe)

    def _holds(self, directory: str) -> bool:
        inside = directory + "/"
        for path in self.base[bisect.bisect_left(self.base, inside) :]:
            if not path.startswith(inside):
                break
            if path not in self.deleted:
                return True
        return any(p.startswith(inside) for p in self.created) or any(
            d.startswith(inside) for d in self.dirs
        )


def _observed_lines(observation: str) -> list[str]:
    """The lines an observation shows the command printing, in whichever scaffold's format."""
    if "<returncode>" in observation:
        fmt = RETURNCODE
    elif observation.lstrip().startswith("OBSERVATION:"):
        fmt = SWE_AGENT
    else:
        fmt = OPENHANDS
    body = observation_body(observation, fmt)
    return [] if body.strip() in ("", NO_OUTPUT_SENTENCE) else body.split("\n")


def _fit_range(lines: list[str], expected: int) -> list[str] | None:
    while len(lines) > expected and not lines[0].strip():
        lines.pop(0)
    while len(lines) > expected and not lines[-1].strip():
        lines.pop()
    return lines if len(lines) == expected else None


def _denumber(lines: list[str]) -> list[str]:
    # the openhands editor numbers the empty line after a file's final newline, and the
    # observation's trailing whitespace strips its tab: a bare number closing a numbered view
    if len(lines) > 1 and lines[-1].strip().isdigit() and _NUMBERED.match(lines[-2]):
        lines = lines[:-1]
    return [m.group(2) if (m := _NUMBERED.match(line)) else line for line in lines]


def session_root(messages) -> str:
    """The absolute directory the checkout lives at, as this transcript spells it.

    The frozen prefix names it on nearly every line, so the most frequent match wins over a
    root a later observation drifted into. Empty when the transcript works from a bare working
    directory (the mini-coder corpus), which is itself the answer: there is no absolute root to
    join paths onto.
    """
    counts = Counter(
        match.group(0)
        for message in messages or []
        for match in CHECKOUT_ROOT.finditer(str(message.get("content") or ""))
    )
    return max(counts, key=lambda root: (counts[root], -len(root))) if counts else ""


def attested_paths(messages) -> set[str]:
    """Paths an earlier observation in this transcript already showed to exist.

    The tracked listing covers the repository at its upstream commit; the container the
    trajectory was recorded in holds more than that — harness scripts (run_tests.sh), build
    output, vendored dependencies, downloads. Those are real, and the frozen prefix proves it
    by listing them, so they must never be reported missing.
    """
    found: set[str] = set()
    for message in messages or []:
        if str(message.get("role") or "").lower() == "assistant":
            continue
        text = str(message.get("content") or "")
        if not text:
            continue
        found.update(match.group(1) for match in _CREATED_AT.finditer(text))
        for match in _VIEW_OF.finditer(text):
            body = next(
                (line for line in text[match.end() :].split("\n") if line.strip()),
                "",
            )
            if not _ENOENT.search(body):
                found.add(match.group(1).strip("`"))
        for match in _LISTING_BLOCK.finditer(text):
            for line in text[match.end() :].split("\n")[1:]:
                if not line.strip():
                    break
                entry = _LISTED_PATH.match(line)
                if entry is None:
                    break
                found.add(entry.group(1).rstrip("/"))
    return {path for path in found if path and not _ENOENT.search(path)}


def build_overlay(
    messages, listing: list[str], read_base, root: str = "", recorded: int | None = None
) -> Overlay:
    """Replay a transcript's commands onto the checkout at its commit.

    A command is replayed once its observation is seen, so its exit status can settle which of
    its stages ran; a command whose observation never arrives is replayed without one. A
    command starts in the checkout root, unless the observation before it reports the shell's
    working directory (a scaffold whose shell persists between commands). Paths an observation
    showed to exist that the replay does not account for may or may not be present.

    `recorded` is how many of the transcript's commands were recorded on a real machine; the
    observations of the ones after them were simulated. File text and git facts are adopted
    only from recorded observations: adopting a simulated read would make whatever the
    simulator once showed the file's text for the rest of the session.
    """
    overlay = Overlay(base=tuple(sorted(listing)), read_base=read_base, root=root)
    pending: str | None = None
    commands = 0
    for turn, message in enumerate(messages or []):
        role = str(message.get("role") or "").lower()
        text = str(message.get("content") or "")
        real = recorded is None or commands <= recorded
        if role == "assistant":
            if pending is not None:
                _replay_turn(overlay, pending, None, turn, real)
            pending = text
            commands += 1
        elif role in ("user", "tool") and pending is not None:
            _replay_turn(overlay, pending, text, turn, real)
            pending = None
    if pending is not None:
        _replay_turn(overlay, pending, None, len(messages or []), False)
    for path in attested_paths(messages):
        rel = session_path(path, "", root)
        if rel and overlay.kind(rel) is None and rel not in overlay.deleted:
            overlay.maybe.add(rel)
    return overlay


def _replay_turn(
    overlay: Overlay, assistant_text: str, observation: str | None, turn: int, learn: bool
):
    # a SEARCH/REPLACE edit is not a shell command: its "Editing `path`:" header lives in the
    # prose around the fence, so it is matched on the whole turn
    if _apply_search_replace(overlay, assistant_text):
        return
    command = first_bash_block(assistant_text)
    if not command:
        return
    before = (set(overlay.dirty), set(overlay.maybe), set(overlay.unsure), overlay.git.unknown)
    returncode = observed_returncode(observation)
    start = overlay.cwd
    after = replay_command(overlay, command, returncode, turn)
    if observation is not None and learn:
        learn_git(overlay, command, observation)
    reported = _REPORTED_CWD.findall(observation or "")
    if reported:
        inside = repo_path(reported[-1], "", overlay.root)
        overlay.cwd = inside if inside is not None else posixpath.normpath(reported[-1])
    else:
        overlay.cwd = "" if observation is not None else after
    changed = (overlay.dirty - before[0]) | (overlay.maybe - before[1])
    overlay.opaque += [(path, command) for path in sorted(changed) if path[:1] != "/"]
    if overlay.unsure - before[2] or (overlay.git.unknown and not before[3]):
        overlay.opaque.append((None, command))
    if observation is not None and learn and not is_scaffold_truncated(observation):
        _learn(overlay, command, observation, start, returncode)
    for reported in _CREATED_AT.findall(observation or ""):
        # the openhands editor wrote the file, with the text exactly as given: the line break a
        # heredoc ends its body with is not in it
        path = session_path(reported, start, overlay.root)
        text = overlay.content.get(path) if path is not None else None
        if text is not None and path not in overlay.dirty:
            overlay.know(path, text.removesuffix("\n"))


def replay_command(overlay: Overlay, command: str, returncode: int | None, turn: int) -> str | None:
    """Apply what a command that ended with `returncode` (None when unknown) did to the checkout,
    as far as its text and its exit status show; the directory the shell is in afterwards. A
    command the shell parser refuses (a loop, a conditional, a substitution) leaves what it may
    have changed unknown."""
    parsed = parse_command(command)
    if isinstance(parsed, Unsupported):
        _refused(overlay, command, overlay.cwd)
        return overlay.cwd
    plan = schedule(parsed.stages, returncode, overlay.cwd, overlay)
    for stage, (runs, directory, status) in zip(parsed.stages, plan.stages):
        if runs is not False:
            apply_stage(overlay, stage, directory, runs is True, status, turn)
    return plan.cwd


@dataclass
class Schedule:
    # per stage: whether it runs (None when it may or may not), the directory it runs in (None
    # when unknown) and its exit status (None when unknown); the directory the list leaves
    stages: list[tuple[bool | None, str | None, int | None]]
    cwd: str | None


# commands whose exit status does not depend on the files they are given
_ALWAYS_SUCCEED = {"true", ":", "echo", "printf", "export", "set", "unset", "pwd", "sleep", "tee"}
# more stages than this whose status is open leave every stage's outcome open
_MAX_OPEN_STAGES = 10


def schedule(
    stages: tuple[Stage, ...], returncode: int | None, cwd: str | None, overlay: Overlay
) -> Schedule:
    """Which stages of a list ran, where, and how they ended, given how the list ended.

    Every way the stages whose status is not fixed could have ended is tried: `&&`, `||`,
    `set -e` and `exit` then decide which stages run, and the ways that do not end the way the
    list did are dropped. A stage runs when it runs in every way left, and may run when it runs
    in some; its directory and status are known when every way left agrees on them.
    """
    fixed = [
        _fixed_status(stage, overlay, cwd, stages[:index]) for index, stage in enumerate(stages)
    ]
    open_ = [index for index, status in enumerate(fixed) if status is None]
    unknown = Schedule([(None, None, None)] * len(stages), None)
    if len(open_) > _MAX_OPEN_STAGES:
        return unknown
    ways = []
    for outcome in itertools.product((0, 1), repeat=len(open_)):
        status = list(fixed)
        for index, code in zip(open_, outcome):
            status[index] = code
        ran, final, after = _simulate(stages, status, cwd)
        if returncode is None or (final == 0) == (returncode == 0):
            ways.append((ran, after))
    if not ways:
        return unknown
    plan = []
    for index in range(len(stages)):
        taken = [ran[index] for ran, _ in ways if ran[index] is not None]
        directories = {entry[0] for entry in taken}
        codes = {entry[1] for entry in taken}
        runs = True if len(taken) == len(ways) else (None if taken else False)
        plan.append(
            (
                runs,
                directories.pop() if len(directories) == 1 else None,
                codes.pop() if len(codes) == 1 else None,
            )
        )
    after = {cwd_after for _, cwd_after in ways}
    return Schedule(plan, after.pop() if len(after) == 1 else None)


def _simulate(stages, status, cwd):
    """Run the list with every stage's status given: for each stage its (directory, status),
    or None when it is skipped; the list's exit status; and the directory it leaves."""
    ran: list[tuple[str | None, int] | None] = []
    last, errexit = 0, False
    for index, stage in enumerate(stages):
        if (stage.separator == AND and last) or (stage.separator == OR and not last):
            ran.append(None)
            continue
        code = status[index]
        ran.append((cwd, code))
        command = stage.command if len(stage.pipeline) == 1 else None
        name = command.name if command else ""
        if name == "cd" and code == 0:
            cwd = cd_target(command, cwd)
        elif name in ("pushd", "popd"):
            cwd = None
        elif command:
            errexit = errexit_after(command, errexit)
        last = code
        following = stages[index + 1].separator if index + 1 < len(stages) else ""
        if name == "exit" or (errexit and code and following not in (AND, OR)):
            return ran + [None] * (len(stages) - index - 1), code, cwd
    return ran, last, cwd


def cd_target(command: SimpleCommand, cwd: str | None) -> str | None:
    """Where `cd` takes the shell: a checkout path, an absolute path outside the checkout, or
    None when that is not known (`cd -`, a variable)."""
    words = [word for word in command.words[1:] if word.text not in ("-P", "-L", "--")]
    if not words:
        return "/root"
    if len(words) > 1 or words[0].expands or words[0].text == "-":
        return None
    return session_path(words[0].text, cwd)


def cd_outcome(overlay: Overlay, command: SimpleCommand, cwd: str | None) -> tuple[str, int] | None:
    """What `cd DIR` prints and returns: nothing and 0 into a directory that exists, bash's
    error and 1 otherwise; None when that is not known here."""
    target = cd_target(command, cwd)
    if target is None or target.startswith("/"):
        return None
    words = [word.text for word in command.words[1:] if word.text not in ("-P", "-L", "--")]
    kind = overlay.kind(target)
    if kind == DIRECTORY:
        return "", 0
    if overlay.shell == "dash" and (kind is None or kind == FILE):
        return f"{overlay.errors}cd: can't cd to {words[0]}\n", 2
    if kind == FILE or (kind is None and _under_file(overlay, target)):
        return f"{overlay.errors}cd: {words[0]}: Not a directory\n", 1
    if kind is None:
        return f"{overlay.errors}cd: {words[0]}: No such file or directory\n", 1
    return None


def errexit_after(command: SimpleCommand, errexit: bool) -> bool:
    """Whether `set -e` is in effect after a command, given whether it was before."""
    if command.name != "set":
        return errexit
    for arg in command.args:
        if arg.startswith(("-", "+")) and "e" in arg[1:]:
            errexit = arg[0] == "-"
        elif arg == "errexit":
            errexit = True
    return errexit


# directories every sandbox has, outside the checkout
_SYSTEM_DIRECTORIES = {"/", "/tmp", "/root", "/usr", "/home", "/etc", "/var", "/opt"}


def _fixed_status(
    stage: Stage, overlay: Overlay, cwd: str | None, before: tuple[Stage, ...]
) -> int | None:
    """A stage's exit status when its words alone decide it: 0 for commands that cannot fail
    as written and for a `cd` into a directory that exists or a `mkdir` before it made (named
    by an absolute path once a stage before it moved the shell), 1 for `false` and for a read of
    files that are absent when no stage before it could have made them, the code of `exit N`."""
    command = stage.pipeline[-1]
    name, args = command.name, command.args
    # after the shell moved, only an absolute path names the same place as before
    if any(s.command.name in ("cd", "pushd", "popd") for s in before):
        cwd = None
    # stages before this one that may have changed the tree, and that may have removed from it
    changers = [
        c
        for s in before
        for c in s.pipeline
        if c.name not in _KEEPS_TMP or c.name == "git" or any(r.writes_file for r in c.redirects)
    ]
    removes = any(c.name not in _ONLY_ADDS for c in changers)
    if name == "cd" and len(stage.pipeline) == 1 and not command.redirects:
        target = cd_target(command, cwd)
        made = {
            session_path(word.text, cwd, overlay.root)
            for s in before
            if s.command.name == "mkdir"
            for word in s.command.words[1:]
            if not word.text.startswith("-") and not word.expands
        }
        if target in _SYSTEM_DIRECTORIES or (
            target is not None
            and (
                (overlay.kind(target) == DIRECTORY and not removes)
                or any(m == target or (m or "").startswith(target + "/") for m in made)
            )
        ):
            return 0
        return None
    if name in _ALWAYS_SUCCEED and not command.redirects:
        return 0
    if name == "false":
        return 1
    if name == "exit":
        return int(args[0]) % 256 if args and args[0].isdigit() else None
    if name == "sed" and len(stage.pipeline) == 1 and not command.expands:
        edit = sed_edit(command.words[1:])
        if edit is not None and edit.rejected:
            return 1
    if changers:
        return None
    if name in ("cat", "head", "tail") and _names_absent(command, overlay, cwd):
        return 1
    if len(stage.pipeline) > 1 or command.redirects or command.expands:
        return None
    if name == "rm":
        return _rm_status(command, overlay, cwd)
    if name == "mkdir" and "-p" in args:
        paths = [session_path(arg, cwd, overlay.root) for arg in args if not arg.startswith("-")]
        if not paths or None in paths:
            return None
        kinds = {kind for path in paths for kind in _kinds_along(overlay, path)}
        return None if UNKNOWN in kinds else int(FILE in kinds)
    return None


# commands that add or edit files but never remove one
_ONLY_ADDS = {"mkdir", "touch", "tee", "cp", "sed", "patch", "ln", "truncate", "chmod", "echo",
              "printf", "cat"}  # fmt: skip
# options of head and tail that take a value
_READ_VALUE_OPTIONS = frozenset(
    {"-n", "-c", "-s", "--lines", "--bytes", "--pid", "--sleep-interval"}
)


def _rm_status(command: SimpleCommand, overlay: Overlay, cwd: str | None) -> int | None:
    """What `rm` returns when its words decide it: 1 for `.` or `..`, for a directory without -r
    and for an absent file without -f; 0 with -f when every operand is a file or absent."""
    flags, _, operands = _options(command.words[1:])
    if not operands or flags - set("frRvd") - {"--force", "--recursive", "--verbose", "--dir"}:
        return None
    if any(word.text.rstrip("/").rsplit("/", 1)[-1] in (".", "..") for word in operands):
        return 1
    kinds = []
    for word in operands:
        paths = _paths(overlay, word, cwd)
        if paths is None:
            return None
        # `rm f/` of a file is refused like an absent operand: -f ignores it, otherwise rc 1
        kinds += [
            None if _slashed_file(overlay, word, path) else overlay.kind(path) for path in paths
        ]
    if UNKNOWN in kinds:
        return None
    force = bool(flags & {"f", "--force"})
    if None in kinds and not force:
        return 1
    if DIRECTORY in kinds and not flags & {"r", "R", "--recursive"}:
        return None if flags & {"d", "--dir"} else 1
    return 0 if force else None


def _names_absent(command: SimpleCommand, overlay: Overlay, cwd: str | None) -> bool:
    """Whether every file a read names is surely absent, so that the read fails."""
    _, _, operands = _options(command.words[1:], _READ_VALUE_OPTIONS)
    return bool(operands) and all(
        word.text != "-"
        and not word.globs
        and not word.expands
        and (path := session_path(word.text, cwd, overlay.root)) is not None
        and overlay.kind(path) is None
        for word in operands
    )


def _kinds_along(overlay: Overlay, path: str) -> list[str | None]:
    """The kind of each directory on the way to `path`, and of `path` itself."""
    parts = path.split("/")
    return [overlay.kind("/".join(parts[:end])) for end in range(1, len(parts) + 1)]


def _under_file(overlay: Overlay, path: str) -> bool:
    return FILE in _kinds_along(overlay, path)


def apply_stage(
    overlay: Overlay,
    stage: Stage,
    cwd: str | None,
    certain: bool = True,
    status: int | None = None,
    turn: int = 0,
) -> None:
    """Apply one stage's file changes, run in `cwd`; `certain` False when it may not have run,
    `status` its exit status when known."""
    first = unwrapped(stage.command)
    if first.name == "git":
        for redirect in first.redirects:
            if redirect.writes_file:
                _write(overlay, redirect.target, cwd, None, redirect.mode == "append", certain)
        apply_git_stage(overlay, stage.text, certain and cwd == "", status, turn)
        if first.args[:1] == ("apply",):
            _patch(overlay, first, cwd, certain, None)
    printed: str | None = None
    for position, command in enumerate(stage.pipeline):
        if position == 0 and first.name == "git":
            continue
        printed = _apply_command(
            overlay, unwrapped(command), cwd, certain, printed, position > 0, status, stage
        )


# commands that run the command after them, with the options that take a value
_WRAPPERS = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S"},
    "sudo": {"-u", "-g", "-C", "-D", "-h", "-p", "-U"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-i", "-o", "-e"},
    "nohup": set(),
    "time": set(),
    "command": set(),
    "exec": set(),
}
_SHELLS = {"bash", "sh", "dash", "zsh"}
_PYTHON = re.compile(r"^python[\d.]*$")
# formatters that rewrite their operands, and the flag that makes them write (None: always)
_FORMATTERS = {"black": None, "isort": None, "prettier": "--write", "gofmt": "-w",
               "rustfmt": None, "autopep8": "-i", "yapf": "-i", "clang-format": "-i"}  # fmt: skip
# programs that run Python, which leaves `__pycache__` beside the modules it imports
_RUNS_PYTHON = {"pytest", "py.test", "pip", "pip3", "tox", "nox", "coverage"}
# commands that leave /tmp as it was, other than by the writes this follows
_KEEPS_TMP = {"cat", "ls", "grep", "egrep", "fgrep", "rg", "head", "tail", "wc", "nl", "awk",
              "sort", "uniq", "cut", "tr", "diff", "cmp", "stat", "file", "which", "type", "echo",
              "printf", "cd", "pwd", "true", "false", ":", "test", "[", "export", "set", "unset",
              "env", "sleep", "basename", "dirname", "realpath", "readlink", "tree", "du", "git",
              "exit", "pushd", "popd", "xxd", "od", "md5sum", "sha256sum", "date"}  # fmt: skip
# programs whose changes cannot be read off their arguments
_OPAQUE_CHANGERS = {"tar", "unzip", "gunzip", "gzip", "bunzip2", "xz", "zip", "rsync", "dd",
                    "install", "cpio", "7z"}  # fmt: skip


def unwrapped(command: SimpleCommand) -> SimpleCommand:
    """The command `env`, `sudo`, `timeout 10` and the like run, their own options dropped."""
    words = list(command.words)
    while words and words[0].text.rsplit("/", 1)[-1] in _WRAPPERS:
        value_options = _WRAPPERS[words.pop(0).text.rsplit("/", 1)[-1]]
        while words and (words[0].text.startswith("-") or "=" in words[0].text):
            if words.pop(0).text in value_options and words:
                words.pop(0)
        if words and re.fullmatch(r"[\d.]+[smhd]?", words[0].text):
            words.pop(0)
    return replace(command, words=tuple(words)) if len(words) != len(command.words) else command


def _apply_command(
    overlay: Overlay,
    command: SimpleCommand,
    cwd: str | None,
    certain: bool,
    stdin: str | None,
    piped: bool,
    status: int | None,
    stage: Stage,
) -> str | None:
    """Apply one command of a pipeline; what it prints to the next one, when known."""
    printed = _printed(overlay, command, cwd, stdin, piped)
    output = _stdout_redirect(command)
    if output is not None and printed is None and command is stage.pipeline[-1]:
        printed = _computed(overlay, stage, cwd)
    for redirect in command.redirects:
        if not redirect.writes_file:
            continue
        if redirect.fd == 1:
            # only the last redirect of standard output receives it; the others are emptied
            text = printed if redirect is output else ""
            if _stderr_joins(command) and _stderr_text(command) != "":
                text = None
        else:
            text = _stderr_text(command)
        _write(overlay, redirect.target, cwd, text, redirect.mode == "append", certain)
    name = command.name
    # sed exits 1 when it rejects its script, before it writes anything
    rejected = name == "sed" and status == 1 and command is stage.pipeline[-1]
    if name in _EFFECTS and not rejected:
        _EFFECTS[name](overlay, command, cwd, certain, stdin if piped else None)
    elif name in _SHELLS or _PYTHON.match(name) or name == "node":
        _program(overlay, command, cwd, certain, stdin if piped else None, status)
    elif name in _FORMATTERS:
        _formatter(overlay, command, cwd, certain)
    elif name in _OPAQUE_CHANGERS:
        overlay.unsure.add("")
    elif name not in _KEEPS_TMP:
        overlay.tmp_open = True
    if _PYTHON.match(name) or name in _RUNS_PYTHON:
        overlay.pycache = True
        if "pytest" in (name, *command.args[1:2]):
            overlay.maybe.add(".pytest_cache")
    return "" if output is not None else printed


def quiet_outcome(overlay: Overlay, stage: Stage, cwd: str | None) -> tuple[str, int] | None:
    """The output and exit status of a stage whose whole work is changing files, when it surely
    succeeds: nothing and 0 (`tee` also prints what it writes). None when success is not
    certain here, or the stage does more than change files."""
    if len(stage.pipeline) != 1:
        return None
    command = unwrapped(stage.command)
    name, args = command.name, command.args
    if command.expands or any(arg in ("-v", "--verbose") for arg in args):
        return None
    for redirect in command.redirects:
        if redirect.mode == "read" or (
            redirect.writes_file and not writable(overlay, redirect.target, cwd)
        ):
            return None
    silenced = _stdout_redirect(command) is not None
    if (
        not command.words
        or name in ("echo", "printf", "true", ":")
        or (name == "cat" and not args and command.heredoc is not None)
    ):
        return ("", 0) if silenced else None
    if name == "tee":
        _, _, operands = _options(command.words[1:])
        if command.heredoc is None or not all(writable(overlay, w, cwd) for w in operands):
            return None
        printed = command.heredoc.literal
        return None if printed is None else ("" if silenced else printed, 0)
    succeeds = _SUCCEEDS.get(name)
    return ("", 0) if succeeds is not None and succeeds(overlay, command, cwd) else None


def writable(overlay: Overlay, word: Word, cwd: str | None) -> bool:
    """Whether a redirect or `tee` can surely write this file."""
    if word.text == "/dev/null":
        return True
    paths = _paths(overlay, word, cwd)
    if not paths or len(paths) > 1:
        return False
    parent = overlay.kind(paths[0].rpartition("/")[0])
    return parent == DIRECTORY and overlay.kind(paths[0]) in (FILE, None)


def _known_paths(overlay: Overlay, words: list[Word], cwd) -> list[tuple[str, str | None]] | None:
    """Each checkout path the words name with its kind; None when one is unknown or outside."""
    out = []
    for word in words:
        paths = _paths(overlay, word, cwd)
        if not paths:
            return None
        for path in paths:
            kind = overlay.kind(path)
            if kind == UNKNOWN:
                return None
            out.append((path, kind))
    return out


def _rm_succeeds(overlay: Overlay, command: SimpleCommand, cwd) -> bool:
    flags, _, operands = _options(command.words[1:])
    if flags - set("frR") - {"--force", "--recursive"} or command.name != "rm":
        return False
    if any(w.text.rstrip("/").rsplit("/", 1)[-1] in (".", "..") for w in operands):
        return False
    found = _known_paths(overlay, operands, cwd)
    recursive, force = bool(flags & {"r", "R", "--recursive"}), bool(flags & {"f", "--force"})
    return found is not None and all(
        (kind is not None or force) and (kind != DIRECTORY or recursive) for _, kind in found
    )


def _mkdir_succeeds(overlay: Overlay, command: SimpleCommand, cwd) -> bool:
    flags, _, operands = _options(command.words[1:], frozenset({"-m", "--mode"}))
    parents = bool(flags & {"p", "--parents"})
    for word in operands:
        for path in _paths(overlay, word, cwd) or [None]:
            if path is None:
                return False
            parts = path.split("/")
            kinds = [overlay.kind("/".join(parts[:end])) for end in range(1, len(parts) + 1)]
            if FILE in kinds or UNKNOWN in kinds:
                return False
            if not parents and (kinds[-1] is not None or (len(kinds) > 1 and kinds[-2] is None)):
                return False
    return bool(operands)


def _touch_succeeds(overlay: Overlay, command: SimpleCommand, cwd) -> bool:
    _, _, operands = _options(command.words[1:], frozenset({"-t", "-d", "-r"}))
    found = _known_paths(overlay, operands, cwd)
    return found is not None and all(
        kind is not None or overlay.kind(path.rpartition("/")[0]) == DIRECTORY
        for path, kind in found
    )


def _transfer_succeeds(overlay: Overlay, command: SimpleCommand, cwd) -> bool:
    flags, values, operands = _options(command.words[1:], frozenset({"-t"}))
    if flags - set("frRap") - {"--force", "--recursive", "--archive"} or "-t" in values:
        return False
    found = _known_paths(overlay, operands, cwd)
    if found is None or len(found) < 2:
        return False
    (destination, dest_kind), sources = found[-1], found[:-1]
    recursive = command.name == "mv" or bool(flags & {"r", "R", "a", "--recursive", "--archive"})
    if any(kind is None or (kind == DIRECTORY and not recursive) for _, kind in sources):
        return False
    if len(sources) > 1 or operands[-1].text.endswith("/"):
        return dest_kind == DIRECTORY
    parent = overlay.kind(destination.rpartition("/")[0])
    source_kind = sources[0][1]
    return (
        parent == DIRECTORY
        and not (dest_kind == FILE and source_kind == DIRECTORY)
        and destination != sources[0][0]
    )


def _sed_succeeds(overlay: Overlay, command: SimpleCommand, cwd) -> bool:
    edit = sed_edit(command.words[1:])
    if edit is None or edit.commands is None or command.redirects:
        return False
    found = _known_paths(overlay, list(edit.files), cwd)
    return found is not None and all(
        kind == FILE and overlay.text(path) is not None for path, kind in found
    )


def _chmod_succeeds(overlay: Overlay, command: SimpleCommand, cwd) -> bool:
    _, _, operands = _options(command.words[1:])
    found = _known_paths(overlay, operands[1:], cwd)
    return found is not None and bool(found) and all(kind is not None for _, kind in found)


_SUCCEEDS = {
    "rm": _rm_succeeds,
    "mkdir": _mkdir_succeeds,
    "touch": _touch_succeeds,
    "cp": _transfer_succeeds,
    "mv": _transfer_succeeds,
    "sed": _sed_succeeds,
    "chmod": _chmod_succeeds,
}


def _stdout_redirect(command: SimpleCommand) -> Redirect | None:
    """The redirect that receives the command's standard output, if one does."""
    writes = (r for r in reversed(command.redirects) if r.fd == 1 and r.mode in ("write", "append"))
    return next(writes, None)


def _stderr_joins(command: SimpleCommand) -> bool:
    """Whether the command's errors go to the file its output is written to."""
    for position, redirect in enumerate(command.redirects):
        if redirect.fd == 2 and redirect.mode == "duplicate" and redirect.target.text == "1":
            return any(r.fd == 1 and r.writes_file for r in command.redirects[:position])
    return False


def _stderr_text(command: SimpleCommand) -> str | None:
    """What a command writes to standard error: nothing for one that prints only its
    arguments or its input, unknown otherwise."""
    if not command.words or command.name in _ALWAYS_SUCCEED:
        return ""
    return "" if command.name == "cat" and not command.args else None


def _printed(
    overlay: Overlay, command: SimpleCommand, cwd: str | None, stdin: str | None, piped: bool
) -> str | None:
    """What a command prints to standard output, when its words and its input decide it."""
    if command.expands or any(word.globs or word.braces for word in command.words):
        return None
    name, args = command.name, list(command.args)
    if not command.words or name in ("true", ":"):
        return ""
    if name == "tee" or (name == "cat" and not args):
        if command.heredoc is not None:
            return command.heredoc.literal
        return stdin if piped else None
    if name == "cat":
        if any(arg.startswith("-") for arg in args):
            return None
        texts = []
        for word in command.words[1:]:
            paths = _paths(overlay, word, cwd)
            if not paths:
                return None
            texts += [overlay.text(path) for path in paths]
        if None in texts or sum(map(len, texts)) > MAX_TEXT_CHARS:
            return None
        return "".join(texts)
    if name == "echo":
        newline = "\n"
        while args and args[0] in ("-n", "-e", "-E"):
            flag = args.pop(0)
            if flag == "-e":
                return None
            if flag == "-n":
                newline = ""
        text = " ".join(args)
        return None if "\\" in text else text + newline
    if name == "printf":
        return printf_output(args)
    return None


def _computed(overlay: Overlay, stage: Stage, cwd: str | None) -> str | None:
    """The output of a read-only pipeline written to a file (`grep x f > out`), computed against
    the checkout; None when it is not a search this can run exactly."""
    if cwd != "" or any(command.expands for command in stage.pipeline):
        return None
    if stage.command.name in ("echo", "printf"):
        return None  # what these print depends on the shell running them
    plan = parse_search(" | ".join(shlex.join(command.argv) for command in stage.pipeline))
    if isinstance(plan, ParseFailure):
        return None
    result = run_search(plan, overlay.text, overlay.listing(), root=overlay.root, overlay=overlay)
    if isinstance(result, ParseFailure) or result.missing:
        return None
    return result.raw


def _paths(overlay: Overlay, word: Word, cwd: str | None) -> list[str] | None:
    """The paths a command word names (see `session_path`): several for a glob or a brace list;
    None when which paths it names is not known."""
    if word.expands:
        return None
    out: list[str] = []
    expanded = expand_braces(word.text) if word.braces else [word.text]
    if expanded is None:
        return None
    for text in expanded:
        path = session_path(text, cwd, overlay.root)
        if path is None:
            if cwd is None and not text.startswith("/"):
                return None
            continue
        if word.globs and any(char in text for char in "*?["):
            matches = _glob(overlay, path)
            if matches is None:
                return None
            if text.endswith("/"):
                # a trailing slash makes the pattern match directories only
                matches = [match for match in matches if overlay.kind(match) == DIRECTORY]
            out += matches or [path]
        else:
            out.append(path)
    return out


def _glob(overlay: Overlay, pattern: str) -> list[str] | None:
    """The paths a shell glob matches, in the order bash sorts them; None when a directory it
    walks has entries that are not known."""
    absolute = pattern.startswith("/")
    current = ["/" if absolute else ""]
    segments = pattern.lstrip("/").split("/") if absolute else pattern.split("/")
    for position, segment in enumerate(segments):
        last = position == len(segments) - 1
        following: list[str] = []
        for directory in current:
            joined = f"{directory.rstrip('/')}/{segment}" if directory else segment
            if not any(char in segment for char in "*?["):
                if overlay.kind(joined) is not None:
                    following.append(joined)
                continue
            if overlay.doubt(directory):
                return None
            for name, is_dir in sorted(overlay.entries(directory).items()):
                hidden = name.startswith(".") and not segment.startswith(".")
                if not hidden and fnmatch.fnmatchcase(name, segment) and (last or is_dir):
                    following.append(f"{directory}/{name}" if directory else name)
        current = following
    return current


def _options(
    words: tuple[Word, ...], value_options: frozenset[str] = frozenset()
) -> tuple[set[str], dict[str, str], list[Word]]:
    """A command's option letters and long options, the values of those that take one, and
    its operands."""
    flags: set[str] = set()
    values: dict[str, str] = {}
    operands: list[Word] = []
    index = 0
    while index < len(words):
        text = words[index].text
        index += 1
        if text == "--":
            operands += words[index:]
            break
        if text.startswith("--"):
            name, equals, value = text.partition("=")
            if not equals and name in value_options and index < len(words):
                value, index = words[index].text, index + 1
            values[name] = value
            flags.add(name)
        elif text.startswith("-") and len(text) > 1:
            if text[:2] in value_options:
                if not text[2:] and index < len(words):
                    values[text[:2]], index = words[index].text, index + 1
                else:
                    values[text[:2]] = text[2:]
                flags.add(text[:2])
            else:
                flags.update(text[1:])
        else:
            operands.append(words[index - 1])
    return flags, values, operands


def _write(
    overlay: Overlay, word: Word, cwd: str | None, text: str | None, append: bool, certain: bool
) -> None:
    """A file a redirect or `tee` writes. One among the machine's files is written when the
    write surely ran: the directories there are the machine's, which do not go away."""
    if word.text == "/dev/null":
        return
    paths = _paths(overlay, word, cwd)
    if paths is None or len(paths) > 1:
        overlay.unsure.add("")
        return
    if not paths:
        return
    path = paths[0]
    kind = overlay.kind(path)
    parent = overlay.kind(path.rpartition("/")[0])
    if kind == DIRECTORY or parent in (None, FILE):
        return
    if append and kind is not None:
        held = overlay.text(path)
        text = held + text if held is not None and text is not None else None
    overlay.put(path, text, certain and (overlay.machine(path) or UNKNOWN not in (kind, parent)))


def _slashed_file(overlay: Overlay, word: Word, path: str) -> bool:
    """Whether a word names a file with a trailing slash, which rm, mv and cp refuse."""
    return word.text.endswith("/") and overlay.kind(path) == FILE


def _rm(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """`rm`, `unlink`, and `rmdir`, which removes only empty directories. `rm` refuses `.`
    and `..`, and removes a directory only recursively (or, with -d, an empty one)."""
    flags, _, operands = _options(command.words[1:])
    name = command.name
    if name == "rm" and flags - set("frRvdI") - {"--force", "--recursive", "--verbose", "--dir"}:
        overlay.unsure.add("")
        return
    recursive = bool(flags & {"r", "R", "--recursive"})
    for word in operands:
        if word.text.rstrip("/").rsplit("/", 1)[-1] in (".", ".."):
            continue
        paths = _paths(overlay, word, cwd)
        if paths is None:
            overlay.unsure.add("")
            continue
        for path in paths:
            kind = overlay.kind(path)
            if (
                kind is None
                or (name == "unlink" and kind == DIRECTORY)
                or _slashed_file(overlay, word, path)
            ):
                continue
            if name == "rmdir":
                _rmdir(overlay, path, "p" in flags, certain)
            elif kind == DIRECTORY and not recursive:
                if "d" in flags and not overlay.entries(path) and not overlay.doubt(path):
                    overlay.drop(path, certain)
            else:
                overlay.drop(path, certain and kind != UNKNOWN)


def _rmdir(overlay: Overlay, path: str, parents: bool, certain: bool) -> None:
    parts = path.split("/")
    chain = ["/".join(parts[:end]) for end in range(len(parts), 0, -1)] if parents else [path]
    for directory in chain:
        kind = overlay.kind(directory)
        if kind == UNKNOWN or (kind == DIRECTORY and overlay.doubt(directory)):
            overlay.drop(directory, False)
        elif kind == DIRECTORY and not overlay.entries(directory):
            overlay.drop(directory, certain)
        else:
            return


def _transfer(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """`cp` and `mv`. Options that make the result depend on what is already there (-n, -u,
    -i) leave the destination with unknown text."""
    flags, values, operands = _options(command.words[1:], frozenset({"-t", "--target-directory"}))
    known = set("fiprRanuvTLHPd") | {"-t", "--force", "--recursive", "--archive", "--verbose",
                                     "--no-clobber", "--update", "--target-directory",
                                     "--no-target-directory"}  # fmt: skip
    if flags - known:
        overlay.unsure.add("")
        return
    move = command.name == "mv"
    recursive = move or bool(flags & {"r", "R", "a", "--recursive", "--archive"})
    conditional = bool(flags & {"n", "i", "u", "--no-clobber", "--update"})
    expanded = [expand_braces(word.text) if word.braces else [word.text] for word in operands]
    if None in expanded:
        overlay.unsure.add("")
        return
    words = [
        replace(word, text=text, braces=False)
        for word, texts in zip(operands, expanded)
        for text in texts
    ]
    target_dir = values.get("-t", values.get("--target-directory"))
    if target_dir is not None:
        destination_word, sources = Word(target_dir), words
    elif len(words) >= 2:
        destination_word, sources = words[-1], words[:-1]
    else:
        return
    found = [_paths(overlay, word, cwd) for word in sources]
    if None in found:
        overlay.unsure.add("")
        return
    into = target_dir is not None or sum(len(paths) for paths in found) > 1
    destinations = _paths(overlay, destination_word, cwd)
    if destinations is None or len(destinations) > 1:
        overlay.unsure.add("")
        return
    destination = destinations[0] if destinations else None
    dest_kind = overlay.kind(destination) if destination is not None else None
    for word, paths in zip(sources, found):
        for source in paths:
            if source.startswith("/") and overlay.kind(source) is None and destination:
                # a file outside the checkout this does not know of may still be there: what
                # it would bring is not known, which is safer than a checkout left stale
                name = source.rsplit("/", 1)[-1]
                target = f"{destination}/{name}" if dest_kind == DIRECTORY else destination
                overlay.put(target, None, False)
                continue
            _transfer_one(
                overlay, source, destination, dest_kind, word.text.endswith("/."), into,
                destination_word.text.endswith("/"), "T" in flags, move, recursive,
                certain and not conditional,
            )  # fmt: skip


def _transfer_one(
    overlay: Overlay,
    source: str,
    destination: str | None,
    dest_kind: str | None,
    contents: bool,
    into: bool,
    slash: bool,
    no_target_dir: bool,
    move: bool,
    recursive: bool,
    certain: bool,
) -> None:
    """Copy or move one source. `into` says the destination must be an existing directory
    (several sources, or `-t`), `slash` that it is written with a trailing slash, which a file
    cannot be copied to unless the directory exists; `contents` that the source is `dir/.`."""
    kind = overlay.kind(source)
    if kind is None or (kind == DIRECTORY and not recursive):
        return
    sure = certain and UNKNOWN not in (kind, dest_kind)
    if destination is None:
        if move:
            overlay.drop(source, sure)
        return
    if dest_kind == UNKNOWN:
        overlay.unsure.add(destination)
        if move:
            overlay.drop(source, False)
        return
    name = source.rsplit("/", 1)[-1]
    if contents:
        target = destination
    elif dest_kind == DIRECTORY and not no_target_dir:
        target = f"{destination}/{name}" if destination else name
    elif into or (slash and (kind == FILE or dest_kind is not None)):
        return
    elif dest_kind == FILE and kind == DIRECTORY:
        return
    else:
        target = destination
    if target == source:
        return
    if target.startswith(source + "/"):
        # mv refuses to move a directory into itself; cp -r copies part of it before noticing
        if not move:
            overlay.make_dir(target, sure)
            overlay.unsure.add(target)
        return
    if overlay.kind(target.rpartition("/")[0]) in (None, FILE):
        return
    if contents and kind == DIRECTORY:
        overlay.make_dir(target, sure)
        for entry in overlay.entries(source):
            _copy_tree(overlay, f"{source}/{entry}", f"{target}/{entry}", sure)
    else:
        _copy_tree(overlay, source, target, sure)
    if move:
        overlay.drop(source, sure)


def _copy_tree(overlay: Overlay, source: str, target: str, certain: bool) -> None:
    """Copy a file, or a directory and everything in it, onto `target`."""
    if overlay.kind(source) == FILE:
        overlay.put(target, overlay.text(source), certain)
        return
    overlay.make_dir(target, certain)
    if overlay.doubt(source):
        overlay.unsure.add(target)
        return
    for path in overlay.files_under(source):
        overlay.put(target + path[len(source) :], overlay.text(path), certain)
    for directory in [d for d in overlay.dirs if d.startswith(source + "/")]:
        overlay.make_dir(target + directory[len(source) :], certain)


def _mkdir(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    flags, _, operands = _options(command.words[1:], frozenset({"-m", "--mode"}))
    parents = bool(flags & {"p", "--parents"})
    for word in operands:
        paths = _paths(overlay, word, cwd)
        if paths is None:
            overlay.unsure.add("")
            continue
        for path in paths:
            parts = path.split("/")
            chain = ["/".join(parts[:end]) for end in range(1, len(parts) + 1)]
            kinds = [overlay.kind(p) for p in chain]
            if FILE in kinds or (not parents and kinds[-1] is not None):
                continue
            if not parents and len(chain) > 1 and kinds[-2] is None:
                continue
            sure = certain and UNKNOWN not in kinds
            for directory, kind in zip(chain, kinds) if parents else [(path, kinds[-1])]:
                if kind != DIRECTORY:
                    overlay.make_dir(directory, sure)


def _touch(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    flags, _, operands = _options(
        command.words[1:], frozenset({"-t", "-d", "-r", "--date", "--reference"})
    )
    if flags & {"c", "--no-create"}:
        return
    for word in operands:
        paths = _paths(overlay, word, cwd)
        if paths is None:
            overlay.unsure.add("")
            continue
        for path in paths:
            kind = overlay.kind(path)
            parent = overlay.kind(path.rpartition("/")[0])
            if kind in (FILE, DIRECTORY) or parent in (None, FILE):
                continue
            overlay.put(path, "", certain and kind is None and parent == DIRECTORY)


def _tee(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    flags, _, operands = _options(command.words[1:])
    text = command.heredoc.literal if command.heredoc is not None else stdin
    for word in operands:
        _write(overlay, word, cwd, text, bool(flags & {"a", "--append"}), certain)


def _ln(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """A link made at its last operand, with the text of whatever it points to unknown."""
    _, _, operands = _options(command.words[1:], frozenset({"-t", "-S"}))
    paths = _paths(overlay, operands[-1], cwd) if len(operands) == 2 else None
    if paths is None or len(paths) != 1 or overlay.kind(paths[0]) == DIRECTORY:
        overlay.unsure.add("")
    else:
        overlay.put(paths[0], None, certain)


def _truncate(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    _, values, operands = _options(command.words[1:], frozenset({"-s", "--size", "-r"}))
    size = values.get("-s", values.get("--size"))
    for word in operands:
        for path in _paths(overlay, word, cwd) or []:
            overlay.put(path, "" if size == "0" else None, certain)


def _sed(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    edit = sed_edit(command.words[1:])
    if edit is None or edit.rejected:
        return
    for word in edit.files:
        paths = _paths(overlay, word, cwd)
        if paths is None:
            overlay.unsure.add("")
            continue
        for path in paths:
            kind = overlay.kind(path)
            if kind not in (FILE, UNKNOWN):
                continue
            text = overlay.text(path)
            if edit.backup:
                _backup(overlay, path, edit.backup, text, certain)
            edited = edit.apply(text) if text is not None and edit.commands is not None else None
            overlay.put(path, edited, certain and kind == FILE)


def _backup(overlay: Overlay, path: str, suffix: str, text: str | None, certain: bool) -> None:
    """The copy an in-place edit with a suffix (`-i.bak`) keeps of the file it edits."""
    if "/" in suffix or "*" in suffix:
        overlay.unsure.add("")
    else:
        overlay.put(path + suffix, text, certain)


def _perl(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """`perl -i -pe ...` rewrites its operands in a way this does not reproduce."""
    backup, in_place, program_next, has_program = "", False, False, False
    files: list[Word] = []
    for word in command.words[1:]:
        if program_next:
            program_next = False
        elif word.text.startswith("-") and len(word.text) > 1:
            letters = word.text[1:]
            if "i" in letters:
                in_place, backup = True, letters[letters.index("i") + 1 :]
                letters = letters[: letters.index("i")]
            program_next = letters.endswith(("e", "E"))
            has_program = has_program or program_next
        else:
            files.append(word)
    if not in_place:
        return
    if not has_program:
        overlay.unsure.add("")
        return
    for word in files:
        for path in _paths(overlay, word, cwd) or []:
            if overlay.kind(path) != FILE:
                continue
            if backup:
                _backup(overlay, path, backup, overlay.text(path), certain)
            overlay.forget(path)


def _formatter(overlay: Overlay, command: SimpleCommand, cwd, certain) -> None:
    write_flag = _FORMATTERS[command.name]
    if write_flag is not None and write_flag not in command.args:
        return
    _, _, operands = _options(command.words[1:], frozenset({"-l", "--line-length", "--config"}))
    for word in operands:
        paths = _paths(overlay, word, cwd)
        if paths is None:
            overlay.unsure.add("")
            continue
        for path in paths:
            if overlay.kind(path) == FILE:
                overlay.forget(path)
            elif overlay.kind(path) is not None:
                overlay.unsure.add(path)


_EXEC_ACTIONS = ("-exec", "-execdir", "-ok", "-okdir")


def _find(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """`find ... -delete` and `find ... -exec rm` remove what find selects; `-exec` of an
    in-place edit changes the selected files, and of another file changer anything."""
    args = list(command.argv)
    start = next((i for i, arg in enumerate(args) if arg in _EXEC_ACTIONS), None)
    tests: list[str] | None
    if "-delete" in args:
        tests, action = [arg for arg in args if arg != "-delete"], "delete"
    elif start is not None:
        end = next((i for i in range(start + 1, len(args)) if args[i] in (";", "+")), None)
        action = _exec_action(args[start + 1 : end])
        if action is None:
            return
        simple = end == len(args) - 1 and args[start] == "-exec" and action != "unknown"
        tests = args[:start] if simple else None
    else:
        return
    matches = _find_matches(overlay, tests, cwd) if tests is not None else None
    if matches is None:
        overlay.unsure.add("")
        return
    for path in matches:
        if action == "delete":
            overlay.drop(path, certain and overlay.kind(path) == FILE)
        elif overlay.kind(path) == FILE:
            overlay.forget(path)


def _find_matches(overlay: Overlay, tests: list[str], cwd) -> list[str] | None:
    if cwd != "":
        return None
    plan = parse_search(shlex.join(tests))
    if isinstance(plan, ParseFailure):
        return None
    result = run_search(plan, overlay.text, overlay.listing(), root=overlay.root, overlay=overlay)
    if isinstance(result, ParseFailure):
        return None
    lines = [line for line in result.output.split("\n") if line and not line.startswith("find:")]
    paths = [session_path(line, "", overlay.root) for line in lines]
    return None if None in paths else paths


def _exec_action(words: list[str]) -> str | None:
    """What a command run on selected files (`find -exec`, `xargs`) does to them: "delete",
    "edit" (in place), "unknown" (changes files some other way), or None (reads them)."""
    if not words:
        return None
    name = words[0].rsplit("/", 1)[-1]
    if name in ("rm", "unlink"):
        return "delete"
    if name in ("sed", "perl"):
        edits = any(
            word.startswith("--in-place") or (word[:1] == "-" and word[1:2] != "-" and "i" in word)
            for word in words[1:]
        )
        return "edit" if edits else None
    if name in _FORMATTERS:
        return "edit"
    changers = {"mv", "cp", "touch", "mkdir", "rmdir", "tee", "patch", "chmod", "ln", "truncate"}
    changes = name in changers | _SHELLS | _OPAQUE_CHANGERS or _PYTHON.match(name)
    return "unknown" if changes or name == "git" else None


def _xargs(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """`... | xargs rm`: a change to files only known once the pipeline runs."""
    args = list(command.args)
    index = 0
    while index < len(args) and args[index].startswith("-"):
        takes_value = args[index] in {"-I", "-i", "-n", "-L", "-l", "-P", "-d", "-s", "-a", "-E"}
        index += 2 if takes_value else 1
    if _exec_action(args[index:]) is not None:
        overlay.unsure.add("")


def _patch(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """`patch` and `git apply`: each file a unified diff names gets the diff applied when its
    text is known and the diff applies at the lines it names, and unknown text otherwise."""
    args = list(command.args)
    if command.name == "git":
        args = args[1:]
        if set(args) & {"--check", "--stat", "--numstat", "--summary", "--cached", "--index"}:
            return
        strip = 1
    else:
        if set(args) & {"--dry-run", "--check"}:
            return
        flags, values, operands = _options(command.words[1:], frozenset({"-p", "--strip", "-i"}))
        strip = values.get("-p", values.get("--strip"))
        strip = int(strip) if strip is not None and strip.isdigit() else None
        original = operands[0].text if operands else None
    reverse = bool({"-R", "--reverse"} & set(args))
    text = command.heredoc.literal if command.heredoc is not None else stdin
    source = next((r.target for r in command.redirects if r.mode == "read"), None)
    if command.name == "git":
        original = None
        if source is None and text is None:
            source = next((Word(arg) for arg in args if not arg.startswith("-")), None)
    elif "-i" in values:
        source = Word(values["-i"])
    elif len(operands) > 1:
        source = operands[1]
    if source is not None:
        found = _paths(overlay, source, cwd)
        text = overlay.text(found[0]) if found else None
    files = unified_diff(text) if text else None
    if not files or (original is not None and len(files) != 1):
        overlay.unsure.add("")
        return
    for old, new, hunks, marked in files:
        if reverse:
            old, new = new, old
            swap = {"+": "-", "-": "+"}
            hunks = [(s, [swap.get(line[:1], line[:1]) + line[1:] for line in b]) for s, b in hunks]
        name = new if new != "/dev/null" else old
        name = name.rsplit("/", 1)[-1] if strip is None else "/".join(name.split("/")[strip:])
        name = original or name
        paths = _paths(overlay, Word(name), cwd) if name else None
        if not paths or len(paths) != 1:
            overlay.unsure.add("")
            continue
        path = paths[0]
        if new == "/dev/null":
            overlay.drop(path, certain)
            continue
        base = "" if old == "/dev/null" else overlay.text(path)
        patched = apply_hunks(base, hunks) if base is not None and not marked else None
        overlay.put(path, patched, certain)


def _apply_patch_tool(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """`apply_patch` with a `*** Begin Patch` body: an added file gets its text, an updated
    one unknown text, and a deleted one is removed."""
    text = command.heredoc.literal if command.heredoc is not None else stdin
    if not text:
        overlay.unsure.add("")
        return
    for match in re.finditer(r"^\*\*\* (Add|Update|Delete) File: (.+)$", text, re.M):
        paths = _paths(overlay, Word(match.group(2).strip()), cwd)
        if not paths:
            overlay.unsure.add("")
            continue
        verb, path = match.group(1), paths[0]
        if verb == "Delete":
            overlay.drop(path, certain)
        elif verb == "Add":
            body = text[match.end() + 1 :].split("\n*** ", 1)[0]
            added = [line[1:] for line in body.split("\n") if line.startswith("+")]
            overlay.put(path, "".join(f"{line}\n" for line in added), certain)
        else:
            overlay.put(path, None, certain)


def _chmod(overlay: Overlay, command: SimpleCommand, cwd, certain, stdin) -> None:
    """A mode change leaves a file's text as it was. Git reports it for a tracked file, which
    git answers do not model, so they are not computed once one changed."""
    _, _, operands = _options(command.words[1:])
    for word in operands[1:]:
        for path in _paths(overlay, word, cwd) or []:
            if overlay.kind(path) == FILE and (overlay.in_base(path) or path in overlay.git.index):
                overlay.git.modes.add(path)


_EFFECTS = {
    "rm": _rm,
    "rmdir": _rm,
    "unlink": _rm,
    "cp": _transfer,
    "mv": _transfer,
    "mkdir": _mkdir,
    "touch": _touch,
    "tee": _tee,
    "ln": _ln,
    "truncate": _truncate,
    "sed": _sed,
    "perl": _perl,
    "find": _find,
    "xargs": _xargs,
    "patch": _patch,
    "apply_patch": _apply_patch_tool,
    "chmod": _chmod,
}


def _program(
    overlay: Overlay, command: SimpleCommand, cwd, certain: bool, stdin: str | None, status
) -> None:
    """The file changes of a shell script, or of a Python or Node program, read from its code
    when the command or the session shows it. Shell code is not replayed: it is read as a
    command the parser refuses (see `_refused`). A program whose code is not shown - the
    repository's own scripts, installed tools - is assumed not to change the checkout."""
    shell = command.name in _SHELLS
    language = "shell" if shell else "node" if command.name == "node" else "python"
    code = _program_code(overlay, command, cwd, stdin, language)
    if code is None:
        overlay.tmp_open = True
        return
    if code is _UNREADABLE:
        overlay.unsure.add("")
        return
    if shell:
        _refused(overlay, code, cwd)
        return
    writes = script_writes(code, language)
    if writes.untraceable:
        overlay.unsure.add("")
        return
    sure = certain and status == 0

    def resolved(path: str) -> str | None:
        return session_path(path, cwd, overlay.root)

    for path, top in writes.made_dirs:
        if (target := resolved(path)) is not None:
            overlay.make_dir(target, sure and top)
    transfers = [(*c, False) for c in writes.copied] + [(*m, True) for m in writes.moved]
    for source, destination, top, into, moved in transfers:
        origin, target = resolved(source), resolved(destination)
        if target is not None and origin is not None and into and overlay.kind(target) == DIRECTORY:
            target = f"{target}/{origin.rsplit('/', 1)[-1]}" if target else origin
        if target is not None and origin is None:
            overlay.put(target, None, sure and top)
        elif origin is not None and overlay.kind(origin) is not None:
            if target is not None:
                _copy_tree(overlay, origin, target, sure and top)
            if moved:
                overlay.drop(origin, sure and top)
    for path, top in writes.written:
        if (target := resolved(path)) is not None and overlay.kind(target) != DIRECTORY:
            overlay.put(target, None, sure and top)
    for path, top in writes.removed:
        if (target := resolved(path)) is not None and overlay.kind(target) is not None:
            overlay.drop(target, sure and top and overlay.kind(target) != UNKNOWN)


_CODE_OPTIONS = {"python": {"-c"}, "node": {"-e", "-p", "--eval", "--print"}, "shell": {"-c"}}
_VALUE_OPTIONS = {"python": {"-W", "-X"}, "node": {"-r", "--require"}, "shell": {"-o", "+o"}}
# the code of a script the session wrote, but with text that is not known
_UNREADABLE = "\0unreadable"


def _program_code(
    overlay: Overlay, command: SimpleCommand, cwd, stdin: str | None, language: str
) -> str | None:
    """The code a shell, Python or Node command runs when the session shows it: the text of
    `-c`, of its standard input, or of a script the session wrote; None when it shows none."""
    words = command.words[1:]
    index = 0
    while index < len(words):
        text = words[index].text
        inline = text in _CODE_OPTIONS[language] or (
            language == "shell" and re.fullmatch(r"-[a-z]*c", text)
        )
        if inline:
            code = words[index + 1] if index + 1 < len(words) else None
            return None if code is None else (_UNREADABLE if code.expands else code.text)
        if text == "-m" and language == "python":
            return None
        if text in _VALUE_OPTIONS[language]:
            index += 2
            continue
        if text == "-" or not text.startswith(("-", "+")):
            break
        index += 1
    script = words[index] if index < len(words) and words[index].text != "-" else None
    if script is None:
        if command.heredoc is not None:
            return command.heredoc.literal or _UNREADABLE
        script = next((r.target for r in command.redirects if r.mode == "read"), None)
        if script is None:
            return stdin
    paths = _paths(overlay, script, cwd)
    if not paths or paths[0] not in overlay.created:
        return None
    return overlay.text(paths[0]) or _UNREADABLE


# a command that may change files, spotted in text the shell parser refuses
_CHANGER = re.compile(
    r"(?:^|[\s;&|({`])(?:rm|rmdir|unlink|mv|cp|mkdir|touch|ln|tee|patch|truncate|install|tar|"
    r"unzip|rsync|dd|chmod|xargs|sed\s+(?:-\w+\s+)*-\w*i|perl\s+-\w*i|find\b[^|;&]*-(?:delete|"
    r"exec)|git\s+(?:checkout|restore|reset|stash|clean|apply|mv|rm|am|merge|rebase|pull|switch))"
    r"(?=[\s;&|)]|$)"
)
_REDIRECT_TARGET = re.compile(r"(?<![0-9<>&])>>?\s*([^\s;&|<>()]+)")


def _refused(overlay: Overlay, command: str, cwd: str | None = "") -> None:
    """A command the shell parser refuses: the files it redirects output to get unknown text,
    and any command in it that changes files leaves the whole checkout unknown."""
    masked = mask_quoted(command)
    if _CHANGER.search(masked):
        overlay.unsure.add("")
        return
    for match in _REDIRECT_TARGET.finditer(masked):
        target = command[match.start(1) : match.end(1)].strip("'\"")
        if target == "/dev/null" or target.startswith("&"):
            continue
        path = session_path(target, cwd, overlay.root)
        if any(char in target for char in "$`*?[{") or (path is None and target[:1] != "/"):
            overlay.unsure.add("")
            return
        if path is not None:
            overlay.put(path, None, False)


def _learn_layout(overlay: Overlay, stages: tuple[Stage, ...], observation: str) -> None:
    """Keep the names a plain `ls` (after any `cd`) printed and how it laid them out, which
    shows how wide the terminal is."""
    if not stages or len(stages[-1].pipeline) != 1:
        return
    last = stages[-1].command
    if any(stage.command.name != "cd" for stage in stages[:-1]):
        return
    flags = "".join(word.text[1:] for word in last.words[1:] if word.text.startswith("-"))
    if last.name != "ls" or last.redirects or set(flags) - set("aA"):
        return
    lines = [line for line in _observed_lines(observation) if line]
    names = [name for line in lines for name in line.split()]
    if len(names) > 1 and all(_LAID_OUT_NAME.fullmatch(name) for name in names):
        overlay.layouts.append((tuple(sorted(names)), tuple(lines)))


_LAID_OUT_NAME = re.compile(r"[A-Za-z0-9%+,\-./=@^_]+")


def _learn(
    overlay: Overlay, command: str, observation: str, cwd: str | None, returncode: int | None
) -> None:
    """Adopt what a successful single read showed: the whole text of a file (`cat`, `nl -ba`),
    a range of its lines (`sed -n 'A,Bp'`), or lines that contradict what is known (`grep -n`);
    and how `ls` laid names out on the terminal."""
    parsed = parse_command(command)
    if isinstance(parsed, Unsupported) or returncode not in (0, None):
        return
    _learn_layout(overlay, parsed.stages, observation)
    if len(parsed.stages) != 1:
        return
    stage = parsed.stages[0]
    command_ = stage.command
    if len(stage.pipeline) != 1 or command_.redirects or command_.expands or command_.name == "git":
        return
    operands = [word for word in command_.words[1:] if not word.text.startswith("-")]
    flags = [word.text for word in command_.words[1:] if word.text.startswith("-")]
    if not operands or operands[-1].globs or operands[-1].braces:
        return
    path = session_path(operands[-1].text, cwd, overlay.root)
    if path is None or overlay.kind(path) == DIRECTORY:
        return
    name = command_.name
    whole = (name == "cat" and set("".join(f[1:] for f in flags)) <= {"n"}) or (
        name == "nl" and flags in (["-ba"], ["-b", "a"])
    )
    if whole and len(operands) == 1:
        # the returncode format shows the output exactly, so whether the file ends with a line
        # break with it; other formats trim the output, and a line break is assumed. A read that
        # differs from the known text only in its final line break never replaces it: a served
        # observation always ends with one
        exact = _EXACT_OUTPUT.search(observation)
        raw = exact.group(1) if exact else "\n".join(_observed_lines(observation)) + "\n"
        lines = _denumber(raw.removesuffix("\n").split("\n"))
        if raw.strip() and not _ENOENT.search(lines[0]):
            text = "\n".join(lines) + ("\n" if raw.endswith("\n") else "")
            current = overlay.text(path)
            if current is None or current.rstrip("\n") != text.rstrip("\n"):
                overlay.put(path, text)
    elif name == "grep" and len(flags) == 1 and "n" in flags[0] and len(operands) == 2:
        _verify_lines(overlay, path, observation)
    elif name == "sed" and flags == ["-n"] and len(operands) == 2:
        if ranged := _SED_RANGE.match(operands[0].text):
            start, end = int(ranged.group(1)), int(ranged.group(2))
            observed = _fit_range(_observed_lines(observation), end - start + 1)
            if observed:
                _splice_range(overlay, path, observed, start)


def _splice_range(overlay: Overlay, path: str, observed: list[str], start_line: int) -> None:
    current = overlay.text(path)
    if current is None or start_line < 1:
        return
    lines = current.split("\n")
    if start_line - 1 > len(lines):
        return
    lines[start_line - 1 : start_line - 1 + len(observed)] = observed
    overlay.know(path, "\n".join(lines))


def _verify_lines(overlay: Overlay, path: str, observation: str) -> None:
    current = overlay.content.get(path)
    if current is None:
        return
    lines = current.split("\n")
    for entry in _observed_lines(observation):
        if not (match := _NUMBERED_HIT.match(entry)):
            continue
        number = int(match.group(1))
        if number > len(lines) or lines[number - 1] != match.group(2):
            overlay.forget(path)
            return


_SEARCH_REPLACE = re.compile(
    r"Editing\s+`([^`]+)`:.*?<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S
)


def _apply_search_replace(overlay: Overlay, text: str) -> bool:
    """Whether the turn is a SEARCH/REPLACE edit; applied when its search text occurs exactly
    once in a file whose text is known, and leaving the file with unknown text otherwise."""
    match = _SEARCH_REPLACE.search(text or "")
    if match is None:
        return False
    raw, old, new = match.groups()
    path = session_path(raw, overlay.cwd, overlay.root)
    if not path:
        return True
    current = overlay.text(path)
    if current is not None and current.count(old) == 1:
        overlay.put(path, current.replace(old, new))
    elif overlay.kind(path) == FILE:
        overlay.forget(path)
    return True
