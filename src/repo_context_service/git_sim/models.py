from __future__ import annotations

from dataclasses import dataclass, field

from .templates import DEFAULT_ABBREV, DEFAULT_BRANCH


@dataclass
class StashEntry:
    """What a `git stash` took away: the working-tree text of tracked files it reset, the
    tracked files it brought back after their deletion, the new files it removed that were
    staged (which come back staged) and the untracked ones it removed (-u, -a)."""

    content: dict[str, str | None] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    staged_new: dict[str, str | None] = field(default_factory=dict)
    created: dict[str, str | None] = field(default_factory=dict)

    def paths(self) -> list[str]:
        return sorted(set(self.content) | self.deleted | set(self.staged_new) | set(self.created))


@dataclass
class GitState:
    index: dict[str, str | None] = field(default_factory=dict)
    staged_deleted: set[str] = field(default_factory=set)
    stash: list[StashEntry] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)
    unknown: bool = False
    branch: str | None = None
    detached: bool = False
    abbrev: int | None = None
    head_short: str | None = None
    head_subject: str | None = None
    history_dirty: bool = False
    # tracked files whose mode changed, which git reports and this does not model
    modes: set[str] = field(default_factory=set)

    def record(self, turn: int, command: str, effect: dict) -> None:
        self.ledger.append({"turn": turn, "command": command, "effect": effect})

    def poison(self, reason: str, history: bool = False) -> None:
        self.unknown = True
        self.history_dirty = self.history_dirty or history


@dataclass
class GitMeta:
    sha: str = ""
    owner: str = ""
    repo: str = ""
    # the length of a short hash; None when only an observation can show it
    abbrev: int | None = DEFAULT_ABBREV
    branch: str = DEFAULT_BRANCH
    detached: bool = False
    # whether git names refs beside commits, as on a terminal; None when that is not known
    decorate: bool | None = False
    # the history is one local commit, whose hash and subject only an observation shows
    squashed: bool = False
    # what `git status` says of the upstream after the branch line
    tracking: str = ""
    history: object = None
    commit_patch: object = None
    # the header `git show` prints for the one served commit of a swesmith mirror, a root
    # commit whose patch adds every tracked file; () when it is not known
    root_header: tuple[str, ...] = ()

    @property
    def short(self) -> str:
        return self.sha[: self.abbrev or DEFAULT_ABBREV]


@dataclass
class GitPlan:
    """A git command: its subcommand and arguments, the pipe stages after it, and where its
    output goes. The arguments are split into flags (`-a` letters, `--long` names, `--`),
    values of the options that take one, and operands."""

    sub: str
    args: list[str] = field(default_factory=list)
    pipeline: list = field(default_factory=list)
    raw: str = ""
    dropped_stderr: bool = False
    evidence: bool = False
    redirect: str | None = None
    flags: set[str] = field(init=False, default_factory=set)
    values: dict[str, str] = field(init=False, default_factory=dict)
    paths: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        takes_value = _VALUE_OPTIONS.get(self.sub, frozenset())
        index = 0
        while index < len(self.args):
            token = self.args[index]
            index += 1
            if token == "--":
                self.flags.add("--")
                self.paths += self.args[index:]
                return
            name, equals, value = token.partition("=")
            if token.startswith("--") and equals:
                self.values[name] = value
            elif token in takes_value and index < len(self.args):
                self.values[token] = self.args[index]
                index += 1
            elif token.startswith("--"):
                self.flags.add(token)
            elif token.startswith("-") and len(token) > 1:
                self.flags.update("-" + letter for letter in token[1:])
            else:
                self.paths.append(token)


# per subcommand, the options whose value is the next argument
_VALUE_OPTIONS = {
    "diff": frozenset({"--unified", "-U"}),
    "checkout": frozenset({"-b", "-B"}),
    "switch": frozenset({"-c", "-C"}),
    "restore": frozenset({"--source"}),
}


@dataclass
class GitResult:
    output: str
    returncode: int = 0
    empty: bool = False
    exact: bool = True
