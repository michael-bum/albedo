from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx
from loguru import logger

from albedo_config import RepoContextSettings
from albedo_eval_service.remote.dataset import (
    extract_turns,
    parse_sample_id,
    read_parquet_row,
    turn_content,
    turn_role,
    unwrap_column,
)
from albedo_eval_service.shared.dataset_manifest import load_manifest_file
from albedo_eval_service.shared.observation_format import (
    OPENHANDS,
    OPENHANDS_TRUNCATION_NOTICE,
    RETURNCODE,
    SWE_AGENT,
    absent_tool_output,
    command_contract,
    detect_format,
    first_bash_block,
    heredoc_bodies,
)
from albedo_eval_service.shared.submit_protocol import ANY_MARKER_RE

from .command_search import (
    ParseFailure,
    SearchResult,
    number_lines,
    parse_search,
    repo_path,
    run_search,
    session_path,
)
from .git_sim import (
    DEFAULT_ABBREV,
    DEFAULT_BRANCH,
    GitMeta,
    GitResult,
    explain_git,
    git_evidence,
    is_git_command,
    ledger_block,
    parse_git,
    root_commit_header,
    run_git,
)
from .overlay import (
    FILE,
    Overlay,
    apply_stage,
    attested_paths,
    build_overlay,
    cd_outcome,
    cd_target,
    errexit_after,
    quiet_outcome,
    session_root,
    unwrapped,
    writable,
)
from .shell import AND, OR, Stage, Unsupported, parse_command

_API_BASE = "https://api.github.com"
_DONE_MARKER = ".albedo-repo-context-done"
_LISTING_NAME = ".albedo-listing.json"
_REPO_META_NAME = ".albedo-repo-meta.json"
_HISTORY_NAME = ".albedo-git-log.{key}.json"
_HISTORY_PAGE = 100
_PATCH_NAME = ".albedo-commit.{key}.patch"
_MAX_PATCH_CHARS = 400_000
_NEGATIVE_TTL_SECONDS = 24 * 3600.0
_TRANSIENT_TTL_SECONDS = 900.0
_UPSTREAM_ABSENT_TTL_SECONDS = 7 * 24 * 3600.0
_MAX_MEMBER_BYTES = 2 * 1024 * 1024
_MAX_MEMBERS = 200_000
_MAX_MISSING_PATHS = 10
_RETRIES = 3

_DETACHED_SOURCE = re.compile(r"re[-_]?bench", re.I)


# what the openhands editor shows of a file too long to show whole, in the two versions the
# datasets were recorded with
_EDITOR_LIMIT = 16000
_EDITOR_RETRY = (
    "You should retry this tool after you have searched inside the file with `grep -n` in order "
    "to find the line numbers of what you are looking for.</NOTE>"
)
_EDITOR_CLIPPED = (
    "<response clipped><NOTE>Due to the max output limit, only part of this file has been shown "
    f"to you. {_EDITOR_RETRY}"
)
_EDITOR_CLIPPED_TO_SAVE = (
    f"<response clipped><NOTE>To save on context only part of this file has been shown to you. "
    f"{_EDITOR_RETRY}"
)


@dataclass(frozen=True)
class Scaffold:
    """How the machine a dataset's trajectories ran on prints what they ask, as the recorded
    trajectories of that dataset show.

    `columns` holds the terminal widths `ls` may lay names out for (None: one name per line);
    `decorate` whether git names refs beside commits, as it does on a terminal (None: not known);
    `detached` whether the checkout is at a detached HEAD; `abbrev` the length of a short hash
    (None: it grows with the repository, so only an observation shows it); `squashed` whether
    the history is one local commit whose hash only an observation shows; `tracking` what
    `git status` says of the upstream after the branch line; `pycache` whether `__pycache__`
    directories may be there before the session runs Python; `editor` the note the openhands
    editor ends a clipped view with, when a `cat -n` of a file is that editor's view of it (see
    `_editor_view`), "" when it is bash's own output; `errors` what the shell puts before its
    own error messages; `shell` which shell runs the commands (`bash`, or `dash` as `/bin/sh`);
    `branch` the branch the harness checked out, when it names its own rather than the
    repository's default; `detached_at` whether a detached HEAD is `HEAD detached at <short>`
    rather than `Not currently on any branch.`.
    """

    columns: tuple[int, ...] | None = None
    decorate: bool | None = False
    detached: bool = False
    abbrev: int | None = DEFAULT_ABBREV
    squashed: bool = False
    tracking: str = ""
    pycache: bool = False
    editor: str = ""
    errors: str = "bash: "
    shell: str = "bash"
    branch: str = ""
    detached_at: bool = False


# a terminal at least as wide as the narrowest one these datasets were recorded on
_ANY_WIDTH = tuple(range(80, 1001))
# bash running the command as a script (`bash -c`) names the line of an error
_SCRIPT_ERRORS = "bash: line 1: "
_DIVERGED = (
    "Your branch and 'origin/main' have diverged,\n"
    "and have 1 and 1 different commits each, respectively.\n"
    '  (use "git pull" to merge the remote branch into yours)'
)
# a versioned source (`open-swe-traces-v1.1`) is recorded by its family's harnesses
_SOURCE_VERSION = re.compile(r"-v\d+(?:\.\d+)*$")
# dash, mini-swe-agent's `/bin/sh`, numbers the line of an error
_DASH_ERRORS = "/bin/sh: 1: "
# bash started by its path (`/bin/bash -c`) names itself by that path
_BIN_BASH_ERRORS = "/bin/bash: line 1: "
_SCALE_SWE = "scale-swe"


def source_family(source: str) -> str:
    return _SOURCE_VERSION.sub("", source)


def task_source(instance_id: str) -> str:
    """The upstream task set an instance comes from, where its machines differ: Scale-SWE
    instances are `owner_repo_pr<N>`, SWE-rebench-V2 ones `owner__repo-<N>`."""
    return _SCALE_SWE if _PULL_ID_RE.fullmatch(instance_id or "") else ""


def scaffold_for(source: str, fmt: str, instance_id: str = "") -> Scaffold:
    """The most specific scaffold recorded: an instance's own task set before the source's
    default, the source itself before its family."""
    names, task = (source, source_family(source)), task_source(instance_id)
    keys = [(name, fmt, task) for name in names] + [(name, fmt) for name in names]
    return next((SCAFFOLDS[key] for key in keys if key in SCAFFOLDS), Scaffold())


SCAFFOLDS: dict[tuple[str, ...], Scaffold] = {
    ("mini-coder", RETURNCODE): Scaffold(errors=_SCRIPT_ERRORS),
    ("mini-coder-rs", RETURNCODE): Scaffold(
        squashed=True, tracking=_DIVERGED, errors=_SCRIPT_ERRORS
    ),
    ("mini-coder-rs", OPENHANDS): Scaffold(squashed=True, tracking=_DIVERGED),
    ("open-swe-traces", OPENHANDS): Scaffold(
        columns=_ANY_WIDTH, decorate=True, detached=True, abbrev=None, editor=_EDITOR_CLIPPED
    ),
    ("open-swe-traces", SWE_AGENT): Scaffold(
        columns=_ANY_WIDTH,
        decorate=None,
        detached=True,
        abbrev=None,
        editor=_EDITOR_CLIPPED_TO_SAVE,
    ),
    # mini-swe-agent runs each command in its own `/bin/sh -c`, off a terminal. The later third of
    # v1.2's SWE-rebench-V2 rows is on `main` instead; the id does not tell them apart
    ("open-swe-traces", RETURNCODE): Scaffold(
        detached=True, abbrev=None, errors=_DASH_ERRORS, shell="dash"
    ),
    # Scale-SWE machines: openhands checks out the harness's own `scaleswe` branch, swe-agent
    # leaves HEAD detached at the task's commit, v1.2's mini-swe-agent is on the branch. v1.1's
    # mini-swe-agent is detached in its first 58% of rows and on the branch after
    ("open-swe-traces", OPENHANDS, _SCALE_SWE): Scaffold(
        columns=_ANY_WIDTH, decorate=True, abbrev=None, branch="scaleswe", editor=_EDITOR_CLIPPED
    ),
    ("open-swe-traces", SWE_AGENT, _SCALE_SWE): Scaffold(
        columns=_ANY_WIDTH,
        decorate=None,
        detached=True,
        detached_at=True,
        abbrev=None,
        editor=_EDITOR_CLIPPED_TO_SAVE,
    ),
    ("open-swe-traces", RETURNCODE, _SCALE_SWE): Scaffold(
        detached=True, detached_at=True, abbrev=None, errors=_DASH_ERRORS, shell="dash"
    ),
    ("open-swe-traces-v1.2", RETURNCODE, _SCALE_SWE): Scaffold(
        abbrev=None, branch="scaleswe", errors=_DASH_ERRORS, shell="dash"
    ),
    ("swe-hero", OPENHANDS): Scaffold(
        columns=_ANY_WIDTH,
        decorate=True,
        detached=True,
        abbrev=None,
        pycache=True,
        editor=_EDITOR_CLIPPED,
    ),
    # SWE-Lego's openhands transcripts, converted to returncode turns: an interactive terminal on
    # the repository's own branch. Every action was rewritten as a bash command, so `cat -n` and a
    # heredoc are bash's own and the editor is not emulated.
    ("affine-openhands", RETURNCODE): Scaffold(columns=_ANY_WIDTH, decorate=True, abbrev=None),
}

# Affine's own machines (`affine-<machine>` for PR tasks, `mini-coder-affine-<machine>` for swesmith
# ones), all converted to returncode turns and run off a terminal; they differ in the shell: the
# mini-swe-agent text harness (`mswea`) runs commands in dash, its bash tool (`bash`) in `bash -c`,
# and claude_code, pi, kimi_code and hermes_agent (`tools`) in `/bin/bash -c`. A PR task's machine
# depends on its upstream task set, which its id does not show: SWE-Lego's (the most) are on the
# repository's own branch, R2E-Gym's and Multi-SWE's detached at the commit, SWE-rebench-V2's
# detached unnamed. A Scale-SWE task is on the harness's `scaleswe` branch like Open-SWE's.
for _machine, _shell in (
    ("mswea", {"errors": _DASH_ERRORS, "shell": "dash"}),
    ("bash", {"errors": _SCRIPT_ERRORS}),
    ("tools", {"errors": _BIN_BASH_ERRORS}),
):
    SCAFFOLDS[(f"affine-{_machine}", RETURNCODE)] = Scaffold(abbrev=None, **_shell)
    SCAFFOLDS[(f"affine-{_machine}", RETURNCODE, _SCALE_SWE)] = Scaffold(
        abbrev=None, branch="scaleswe", **_shell
    )
    SCAFFOLDS[(f"mini-coder-affine-{_machine}", RETURNCODE)] = Scaffold(**_shell)
# the sources whose machines can sit elsewhere than another source's for the same instance id
_OWN_SHA_CACHE = ("affine-", "mini-coder-affine-")


# the commands a trajectory's editor views were converted to: a whole file, or a range of it
_EDITOR_READ = re.compile(r"cat -n (\S+)|sed -n '(\d+),(\d+|\$)p' (\S+) \| cat -n")


def _editor_view(
    text: str, first: int = 1, last: int | None = None, clipped: str = _EDITOR_CLIPPED
) -> str | None:
    """The body of the openhands editor's view of a file, or of its lines `first` to `last`:
    the text clipped at the editor's limit and every piece between line breaks numbered from
    `first` (the empty one after a final line break too, for the whole file); the scaffold
    drops the trailing blanks and adds the header itself. None for a range the editor refuses,
    which it answers with an error, and for one ending on the empty piece after a final line
    break, which some versions of the editor show and later ones refuse."""
    lines = text.split("\n")
    certain = len(lines) - (lines[-1] == "")
    if first < 1 or first > len(lines) or (last is not None and not first <= last <= certain):
        return None
    shown = "\n".join(lines[first - 1 : last])
    if len(shown) > _EDITOR_LIMIT:
        shown = shown[:_EDITOR_LIMIT] + clipped
    numbered = enumerate(shown.split("\n"), first)
    return "\n".join(f"{number:6}\t{line}" for number, line in numbered).rstrip()


def _absolute_search(text: str) -> bool:
    """Whether a search names only absolute paths, so that where the shell is does not matter."""
    plan = parse_search(text)
    return (
        not isinstance(plan, ParseFailure)
        and bool(plan.targets)
        and all(target.startswith("/") for target in plan.targets)
    )


def _editor_answer(overlay: Overlay, viewed: re.Match) -> str | None:
    """The editor's view a converted read asks for, when the file's text is known and the
    editor shows it; None for a directory, a missing file or a range the editor refuses."""
    path = session_path(viewed.group(1) or viewed.group(4), overlay.cwd, overlay.root)
    text = overlay.text(path) if path is not None else None
    if text is None:
        return None
    first, last = viewed.group(2), viewed.group(3)
    if first is None:
        return _editor_view(text, clipped=overlay.editor)
    return _editor_view(text, int(first), None if last == "$" else int(last), overlay.editor)


def _emptied_blank_lines(text: str | None, fmt: str) -> str | None:
    """The openhands scaffold prints a line of only blanks as an empty line."""
    return re.sub(r"(?m)^[ \t]+$", "", text) if text and fmt == OPENHANDS else text


_TRUNCATION_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number "
    "of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more "
    "selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect "
    "output to a file and then search in that file."
)
_SCAFFOLD_LIMIT = {"returncode": 10_000, "openhands": 30_000}


def _scaffold_truncate(raw: str, fmt: str) -> str:
    """What the scaffold shows of a command's whole output `raw` (its final line break
    included, when it printed one), without the final line break."""
    limit = _SCAFFOLD_LIMIT.get(fmt)
    if limit is None or len(raw) <= limit:
        return raw.removesuffix("\n")
    half = limit // 2
    if fmt == "openhands":
        return f"{raw[:half]}\n{OPENHANDS_TRUNCATION_NOTICE}\n{raw[-half:]}"
    return (
        f"<warning>\n{_TRUNCATION_WARNING}\n</warning><output_head>\n{raw[:half]}\n</output_head>\n"
        f"<elided_chars>\n{len(raw) - limit} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{raw[-half:]}\n</output_tail>"
    )


_RENDER_RULE = """- Print each path the way the command itself would print it: a command given an absolute search
  root (`find /`, `find {root}/x`, `ls {root}/x`, `realpath`) prints the absolute path, a command
  given a relative one (`find .`, `ls src`) prints it relative to the working directory, and `ls`
  of a single directory prints bare entry names. NEVER answer an absolutely-rooted search with a
  bare relative path — `find /` cannot return `pkg/mod.py`, only `{root}/pkg/mod.py`.
"""

LISTING_HEADER = """REPOSITORY FILE LISTING — tracked files at the current commit relevant to the
command (paths relative to the repo root, sorted; the filesystem returns files in this order).
The repo root is the directory this session starts in and has not left unless a command in the
transcript ran `cd`.
Derive the output of exploration commands (find, ls, grep -l, ...) EXACTLY from this list,
applying the command's filters and pipe limits:
- Output matching paths in EXACTLY the order they appear in this listing — never re-sort them.
- Print each path the way the command itself would print it: `find .` prefixes results with `./`,
  `find src` prefixes them with `src/`, `ls` of a single directory prints bare entry names. A
  search rooted at an absolute directory prints absolute paths — never a bare relative path.
- Do not invent paths that are not in this list and do not omit paths that match.
- This listing is complete for the scope shown: if nothing in it matches the command's
  filters, the command's output is empty.
"""

LISTING_HEADER_ROOTED = (
    """REPOSITORY FILE LISTING — tracked files at the current commit relevant to the command.
The repository is checked out at {root} and stays there for this entire session: it is never
moved, never renamed, and the same file is never served under a second root. The paths below are
absolute, exactly as this filesystem holds them, sorted (the filesystem returns files in this
order).
Derive the output of exploration commands (find, ls, grep -l, ...) EXACTLY from this list,
applying the command's filters and pipe limits:
- Output matching paths in EXACTLY the order they appear in this listing — never re-sort them.
"""
    + _RENDER_RULE
    + """- Do not invent paths that are not in this list and do not omit paths that match.
- This listing is complete for the scope shown: if nothing in it matches the command's
  filters, the command's output is empty.
"""
)

UNCERTAIN_LISTING_NOTE = """Except for these paths, which earlier commands in this session changed in ways
not reproduced here: what exists at or under each of them may differ from this listing, so do not
claim it present or absent from the listing alone, and derive it from the transcript instead:
"""

SESSION_PATHS_HEADER = """PATHS ALREADY ESTABLISHED — earlier observations in this session showed each path below
exists. They are outside the tracked listing because the listing covers only what the repository
tracks at this commit, while the machine also holds harness scripts, build output, vendored
dependencies, downloads and files this session created. Being absent from the listing is NOT
evidence of absence: treat every path below as present, never report it missing, and derive its
content or entries from what the earlier observation showed:
"""

CONTENTS_HEADER = """FILE CONTENTS — exact current content of files referenced by the command:
"""

LINE_NUMBER_NOTE_GREP = """Each content line below is prefixed with its line number as "N:", exactly the form grep -n
emits. The prefix is an annotation, not part of the file's text: reproduce it for lines the
command matches, and strip it anywhere the command does not report line numbers.
"""

LINE_NUMBER_NOTE_COLUMN = """Each content line below is prefixed with its line number, right-aligned then a tab, exactly
the form cat -n and nl emit. The prefix is an annotation, not part of the file's text:
reproduce it for lines the command prints, and strip it anywhere line numbers are not reported.
"""

NOT_PRESENT_HEADER = """FILES NOT PRESENT — ONLY the paths listed immediately below are absent
from this machine. Every other path the command names does exist; never claim otherwise, and in
particular never infer absence from a path being missing from the listing above. Do not invent
content for a listed path — report the terminal error shown beside it, inside the observation,
following the OUTPUT FORMAT exactly. When the command produces no output at all, reply with
exactly the empty observation the OUTPUT FORMAT specifies:
"""

COMPUTED_HEADER = """COMMAND OUTPUT — this search was executed against the repository at this commit, so
the text below is the command's exact output. Reply with it verbatim inside the OUTPUT FORMAT.
Do not add, reorder, re-sort or omit lines, and do not explain it:
"""

COMPUTED_EMPTY = """COMMAND OUTPUT — this search was already executed against the repository at this
commit and matched NOTHING: zero lines of output. That is the verified result, not a gap in what
you were given — the repository was searched in full and no file matched. Do NOT list any path, do
NOT guess at plausible matches, and do NOT reason about which files "should" have matched. An empty
result is a normal outcome for a search. Reply with exactly the empty observation the OUTPUT FORMAT
specifies:
"""

GIT_SEMANTICS_HEADER = """GIT SEMANTICS — how this git command behaves in this checkout; both lines are facts about
the repository state you are simulating, not guidance to repeat in the output:
"""

CHAIN_EVIDENCE_HEADER = """CHAIN STAGES — this command joins several stages, listed below in order, each with the
`&&` or `||` that joins it to the one before. A stage with text under it was executed against the
repository at this commit, after the changes of the stages before it: that text is its exact
output, to reproduce unchanged in your reply. A stage marked as not executed is yours to derive:
"""

CHAIN_GAP = "(not executed here — derive this stage's output yourself)"
CHAIN_MAYBE_GAP = (
    "(not executed here, and it runs only if the stage before it ended the way its `&&` or `||`"
    " requires — derive whether it runs and what it prints yourself)"
)
CHAIN_NO_OUTPUT = "(no output)"
PRIOR_STAGES_HEADER = """EARLIER STAGES OF THIS COMMAND — the command you are answering is the rest of a longer one.
These stages ran before it, in this order, with this output; the repository below is as they left
it. Do not repeat their output: answer only the command shown to you:
"""
PRIOR_GAP_STAGE = (
    "(its output was simulated apart and is not shown here — derive from the command itself what"
    " it changed in the repository)"
)
CHAIN_MAYBE_RUN = (
    "(this is what it prints if it runs; it runs only if the stage before it ended the way its"
    " `&&` or `||` requires)"
)
_MAX_GAP_GROUPS = 3

PRE_EDIT_SUFFIX = (
    "   (as of this commit — the session has since edited this file in a way that could "
    "not be reproduced, so the text below is the version BEFORE those edits)"
)

TRAJECTORY_HEADER = """REFERENCE EXCHANGES — real command -> observation exchanges recorded in this
repository during a reference session on the same task. Use them ONLY to derive real paths, file
contents and output style; do not invent paths or contents they contradict, and never copy task
hints or solution steps into your output:
"""


@dataclass(frozen=True)
class RepoRef:
    instance_id: str
    source: str
    owner: str
    repo: str
    pr: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class GroundingContext:
    context: str | None
    kind: str
    reason: str | None = None
    exact_output: str | None = None
    exact_returncode: int | None = None
    state: str = ""
    # exact output of every stage before the last of an `&&` chain whose last stage is capped by a
    # trailing `| head -N` / `| tail -N` (see cap_last_stage); None unless all of them ran and succeeded
    leading_output: str | None = None
    # the command split into what the repository answers and what is left to simulate (see
    # RepoContextService._parts); None when it is not split
    parts: list[dict] | None = None


@dataclass
class _Ran:
    """A stage of a command as `_execute` ran it: its exact answer (None when not computed),
    whether it surely runs, its position in the command, and the checkout as it found it."""

    stage: Stage
    answer: tuple[str, int] | None
    surely: bool
    index: int
    before: Overlay


class _NotFound(Exception):
    pass


class _SnapshotTooLarge(Exception):
    pass


def _is_permanent_github_error(exc: Exception) -> bool:
    if isinstance(exc, _NotFound):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (400, 410, 422, 451)
    return False


_PULL_ID_RE = re.compile(r"(.+?)_(.+)_pr(\d+)")


def parse_instance(source: str, instance_id: str) -> RepoRef | None:
    try:
        if source.startswith("mini-coder"):
            parts = instance_id.split("__")
            if len(parts) < 2:
                return None
            owner, rest = parts[0], parts[1]
            tokens = rest.split(".")
            for index in range(1, len(tokens)):
                if re.fullmatch(r"[0-9a-f]{6,40}", tokens[index]):
                    return RepoRef(
                        instance_id=instance_id,
                        source=source,
                        owner=owner,
                        repo=".".join(tokens[:index]),
                        commit=tokens[index],
                    )
            return None
        pull = _PULL_ID_RE.fullmatch(instance_id)
        if pull is not None:
            return RepoRef(
                instance_id=instance_id,
                source=source,
                owner=pull.group(1),
                repo=pull.group(2),
                pr=pull.group(3),
            )
        owner_repo, tail = instance_id.rsplit("-", 1)
        owner, repo = owner_repo.split("__", 1)
        if tail.isdigit():
            return RepoRef(instance_id=instance_id, source=source, owner=owner, repo=repo, pr=tail)
        if re.fullmatch(r"[0-9a-f]{40}", tail):
            return RepoRef(
                instance_id=instance_id, source=source, owner=owner, repo=repo, commit=tail
            )
        return None
    except ValueError:
        return None


_SMITH_MIRROR_OWNER = "swesmith"
_SMITH_REMOVE_F2P_SUBJECT = "Remove F2P Tests"
_SMITH_SQUASHED_SUBJECT = "Initial commit"
_SMITH_PR_ID = re.compile(r"\.pr_(\d+)$")
_SHA_RULE = 2


def _smith_mirror(ref: RepoRef) -> RepoRef | None:
    if not ref.source.startswith("mini-coder") or not ref.commit:
        return None
    return replace(
        ref,
        owner=_SMITH_MIRROR_OWNER,
        repo=f"{ref.owner}__{ref.repo}.{ref.commit}",
        commit=ref.instance_id,
    )


def _smith_env_is_bug_patch(ref: RepoRef, head: dict) -> bool:
    if not ref.source.startswith("mini-coder-rs"):
        return False
    message = str((head.get("commit") or {}).get("message") or "")
    return message.startswith(_SMITH_REMOVE_F2P_SUBJECT) and bool(head.get("parents"))


def _squashed_history(sha: str, subject: str = _SMITH_SQUASHED_SUBJECT) -> dict:
    return {"commits": [{"sha": sha, "subject": subject}], "complete": True}


def _smith_upstream(mirror_repo: str) -> tuple[str, str]:
    owner, _, rest = mirror_repo.partition("__")
    return owner, rest.rpartition(".")[0]


def _names_mirrored_pr(subject: str, instance_id: str) -> bool:
    match = _SMITH_PR_ID.search(instance_id or "")
    return bool(match) and re.search(rf"#{match.group(1)}(?!\d)", subject) is not None


def _read_cached_sha(path: Path) -> dict | None:
    """A cached SHA entry, or None when it was written under an older resolution rule."""
    cached = _read_json(path)
    if not isinstance(cached, dict) or cached.get("rule") != _SHA_RULE:
        return None
    return cached


def _gap_groups(executed: list[_Ran]) -> list[list[_Ran]]:
    """Runs of consecutive stages this could not answer, each run gated as one by the separator of
    its first stage. A run gated by `&&` or `||` holds only stages joined by that same separator:
    bash decides a `;` stage, or one joined the other way, against the last status on its own, so
    such a stage starts a run of its own."""
    groups: list[list[_Ran]] = []
    for position, ran in enumerate(executed):
        if ran.answer is not None:
            continue
        leading = groups[-1][0].stage.separator if groups else None
        continues = position and executed[position - 1].answer is None
        if continues and (leading not in (AND, OR) or ran.stage.separator == leading):
            groups[-1].append(ran)
        else:
            groups.append([ran])
    return groups


def _joined_text(stages: list[Stage]) -> str:
    """Stages as one command, each joined to the one before by its own `&&`, `||` or `;`. A `;`
    after a stage spanning several lines (a heredoc) is a line break, as a script writes it, so the
    text still parses; only a stage an `&&` or `||` joins to is grouped in braces."""
    text = ""
    for position, stage in enumerate(stages):
        if position:
            gated = stage.separator in (AND, OR)
            text += f" {stage.separator} " if gated else "\n" if "\n" in text else "; "
        following = stages[position + 1].separator if position + 1 < len(stages) else ""
        braced = "\n" in stage.text and following in (AND, OR)
        text += f"{{ {stage.text}\n}}" if braced else stage.text
    return text


def _canned(stage: Stage) -> tuple[str, int] | None:
    """The answer of a program this environment does not have (pytest, a package install),
    decided from the words of the stage's first command. Piped on into `head` or `tail`, the
    message passes through and the pipeline succeeds; any other pipe is not answered."""
    command = unwrapped(stage.command)
    argv = command.argv
    python = bool(re.fullmatch(r"python[\d.]*", command.name))
    runs_tool = command.name in ("pytest", "py.test") or re.fullmatch(r"pip[\d.]*", command.name)
    if not (runs_tool or (python and argv[1:2] == ("-m",))):
        return None
    canned = absent_tool_output(shlex.join(argv))
    rest = [c.name for c in stage.pipeline[1:]]
    if canned is None or any(name not in ("head", "tail") for name in rest):
        return None
    return canned[0] + "\n", 0 if rest else canned[1]


def _name_patterns(cmd: str) -> list[str]:
    return (
        re.findall(r"-name\s+\"([^\"]+)\"", cmd)
        + re.findall(r"-name\s+'([^']+)'", cmd)
        + re.findall(
            r"-name\s+([^\s\"';|]+)",
            cmd.replace('-name "', '-nameQ"').replace("-name '", "-nameQ'"),
        )
    )


def _filter_listing(paths: list[str], cmd: str) -> tuple[list[str], bool]:
    pats = _name_patterns(cmd)
    if not (pats and cmd.startswith(("find", "ls"))):
        return paths, False
    regexes = [
        re.compile(re.escape(p).replace(r"\*", ".*").replace(r"\?", ".") + "$") for p in pats
    ]
    kept = [p for p in paths if any(r.match(p.rsplit("/", 1)[-1]) or r.match(p) for r in regexes)]
    return kept, True


_WRITE_TARGET = re.compile(r">>?\s*[^\s|;&()\"'<>]+")
_WRITE_DEST = re.compile(
    r"\b(?:cp|mv|install)\s+(?:-\S+\s+)*\S+\s+(\S+)|\b(?:tee|touch|mkdir)\s+(?:-\S+\s+)*(\S+)"
)


# commands whose first operand, or the value of these options, is data rather than a path
_DATA_OPERAND = {"sed": ("-e", "--expression"), "grep": ("-e", "--regexp"), "rg": ("-e",),
                 "awk": (), "egrep": ("-e",), "fgrep": ("-e",)}  # fmt: skip
_CODE_OPTION = {"python": "-c", "node": "-e", "perl": "-e", "bash": "-c", "sh": "-c"}


def _data_words(cmd: str) -> list[str]:
    """The arguments of a command that are data, not files: a sed script, a grep or awk pattern,
    the code of `python -c`, `node -e` or `sh -c`. Tokenising them as shell words invents paths
    out of the text they match or run (`s/self.font.bold/None/`, `from pkg.util import x`)."""
    parsed = parse_command(cmd)
    if isinstance(parsed, Unsupported):
        return []
    data: list[str] = []
    for stage in parsed.stages:
        for command in stage.pipeline:
            name = re.sub(r"[\d.]+$", "", command.name)
            args = list(command.args)
            if name in _CODE_OPTION and _CODE_OPTION[name] in args[:-1]:
                data.append(args[args.index(_CODE_OPTION[name]) + 1])
            if name not in _DATA_OPERAND:
                continue
            valued = [args[i + 1] for i, a in enumerate(args[:-1]) if a in _DATA_OPERAND[name]]
            operands = [a for a in args if not a.startswith("-")]
            data += valued or operands[:1]
    return data


def _python_sources(cmd: str, read) -> list[tuple[str, str]]:
    """The Python code a command runs, with the directory its imports also resolve from: the
    code of `-c`, a heredoc fed to the interpreter, a script (its text read with `read`), or the
    module of `-m`."""
    parsed = parse_command(cmd)
    if isinstance(parsed, Unsupported):
        return []
    sources: list[tuple[str, str]] = []
    for stage in parsed.stages:
        for command in stage.pipeline:
            if not re.fullmatch(r"python[\d.]*", command.name):
                continue
            args = list(command.args)
            if "-c" in args[:-1]:
                sources.append((args[args.index("-c") + 1], ""))
            elif "-m" in args[:-1]:
                sources.append((f"import {args[args.index('-m') + 1]}", ""))
            elif command.heredoc is not None and command.heredoc.literal:
                sources.append((command.heredoc.literal, ""))
            elif (script := next((a for a in args if not a.startswith("-")), None)) is not None:
                path = repo_path(script)
                if path is not None and (text := read(path)) is not None:
                    sources.append((text, posixpath.dirname(path)))
    return sources


def _imported_files(sources: list[tuple[str, str]], listing: set[str], read) -> list[str]:
    """The checkout's files that Python code imports, and those they import in turn (two levels
    deep): each module looked for from the checkout root, `src/`, and the importing file's
    directory, as a module file or a package's `__init__.py`."""
    found: list[str] = []
    for _ in range(2):
        following: list[tuple[str, str]] = []
        for code, directory in sources:
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names += [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                base = name.replace(".", "/")
                for root in ("", "src", directory):
                    for candidate in (f"{base}.py", f"{base}/__init__.py"):
                        path = posixpath.join(root, candidate) if root else candidate
                        if path in listing and path not in found:
                            found.append(path)
                            if (text := read(path)) is not None:
                                following.append((text, posixpath.dirname(path)))
        sources = following
    return found


def _referenced_paths(cmd: str, listing: list[str], read=None) -> tuple[list[str], list[str]]:
    listing_set = set(listing)
    top_dirs = {p.split("/", 1)[0] for p in listing_set if "/" in p}
    present: list[str] = []
    missing: list[str] = []
    # a copy/move/tee destination does not exist yet, which is normal rather than an error
    dests = {m.group(1) or m.group(2) for m in _WRITE_DEST.finditer(cmd)}
    # a heredoc body is the data being written, not shell words naming files. Its tokens may still
    # resolve to real files worth listing as present, but one that does not resolve must never be
    # asserted absent: `dt.datetime.now` is an attribute chain, and telling the simulator it is a
    # missing file is a fabrication that invites it to invent a failure.
    # a sed script is data too: `s/self.font.bold/None/` and `/has_metadata_file/d` are patterns,
    # and tokenising them invents paths out of the text being matched
    data = list(heredoc_bodies(cmd)) + _data_words(cmd)
    in_data = {token for span in data for token in re.split(r"[\s|;&<>()\"']+", span) if token}
    for tok in re.split(r"[\s|;&<>()\"']+", _WRITE_TARGET.sub(" ", cmd)):
        if not tok or tok.startswith("-"):
            continue
        p = repo_path(tok) or tok
        if "://" not in tok and not any(ch in tok for ch in "*?[]{}$`=") and p in listing_set:
            if p not in present:
                present.append(p)
            continue
        if "://" in tok or any(ch in tok for ch in "*?[]{}$`="):
            continue
        norm = p.rstrip("/")
        if not norm or norm in (".", "..") or norm in missing or tok in dests:
            continue
        if any(x.startswith(norm + "/") for x in listing_set):
            continue
        if tok in in_data:
            continue
        # only the checkout's own paths are known absent: an installed package, /tmp or the
        # home directory holds whatever the machine put there
        if repo_path(tok) is not None and (
            tok.startswith("/") or _plausible_repo_path(norm, top_dirs)
        ):
            missing.append(norm)
    if read is not None:
        # what a program imports decides what it prints, as much as the file it runs
        imported = _imported_files(_python_sources(cmd, read), listing_set, read)
        present += [path for path in imported if path not in present]
    return present, missing[:_MAX_MISSING_PATHS]


def _split_attested(missing: list[str], attested: set[str] | None) -> tuple[list[str], list[str]]:
    """Peel the paths the transcript already vouched for out of the not-present list.

    `missing` holds whatever the command names that the tracked listing does not, which is a
    strictly narrower question than whether the path is on the machine: the harness scripts,
    build output, vendored dependencies and downloads a trajectory was recorded against are all
    real and all untracked. Anything an earlier observation showed to exist moves to the vouched
    list instead, so the block states it is present rather than staying silent about it.
    """
    if not attested:
        return missing, []
    known = {repo_path(path) or path: path for path in attested}
    kept: list[str] = []
    vouched: list[str] = []
    for path in missing:
        hit = known.get(path) or known.get(repo_path(path) or path)
        if hit is None:
            kept.append(path)
        elif hit not in vouched:
            vouched.append(hit)
    return kept, vouched


_SEPARATOR = r"(?:^|\||&&|\|\||;)\s*"
_WANTS_LINE_NUMBERS = re.compile(
    _SEPARATOR + r"(?:grep|rg)\b[^|&;]*?\s-\w*n\w*\b"
    r"|" + _SEPARATOR + r"cat\b[^|&;]*?\s-\w*[nb]\w*\b"
    r"|" + _SEPARATOR + r"nl\b"
)


_GREP_STYLE = re.compile(_SEPARATOR + r"(?:grep|rg)\b[^|&;]*?\s-\w*n\w*\b")


def _plausible_repo_path(path: str, top_dirs: set[str]) -> bool:
    segments = path.split("/")
    if segments[0] in top_dirs:
        return True
    if len(segments) == 1 and path.startswith("."):
        return True
    last = segments[-1]
    return bool(re.fullmatch(r"\.?[\w@.-]+\.[A-Za-z][A-Za-z0-9]*", last))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


@lru_cache(maxsize=64)
def _load_listing(listing_path: str) -> tuple[str, ...]:
    return tuple(json.loads(Path(listing_path).read_text()))


@lru_cache(maxsize=4096)
def _iid_from_parquet(dataset_root: str, shard_name: str, row_idx: int) -> str | None:
    import pyarrow.parquet as pq

    path = Path(dataset_root) / shard_name
    try:
        seen = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["instance_id"]):
            if seen + batch.num_rows <= row_idx:
                seen += batch.num_rows
                continue
            value = batch.column("instance_id")[row_idx - seen].as_py()
            return str(value) if value else None
    except Exception:
        return None
    return None


class RepoContextService:
    def __init__(self, settings: RepoContextSettings):
        if not settings.cache_dir:
            raise ValueError(
                "ALBEDO_REPO_CONTEXT_CACHE_DIR is required: the snapshot download directory "
                "must be configured explicitly"
            )
        self.settings = settings
        self.cache_dir = Path(settings.cache_dir).expanduser()
        self._shas_dir = self.cache_dir / "shas"
        self._upstream_dir = self.cache_dir / "upstream"
        self._snapshots_dir = self.cache_dir / "snapshots"
        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
            headers={"User-Agent": "albedo-repo-context", "Accept": "application/vnd.github+json"},
        )
        self._github_semaphore = threading.BoundedSemaphore(8)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._shards: dict[str, tuple[str, list]] | None = None
        self._manifest_error_logged = False
        if not self._auth_headers():
            logger.warning(
                "repo_context_no_github_token: running UNAUTHENTICATED (60 req/hr) — SHA "
                "resolution will rate-limit and PR/commit-based datasets (open-swe-traces, "
                "swe-hero) will fail to ground. Set ALBEDO_REPO_CONTEXT_GITHUB_TOKEN."
            )

    def close(self) -> None:
        self._client.close()

    def context_for(
        self, sample_id: str, assistant_output: str, messages: list[dict[str, str]] | None = None
    ) -> GroundingContext:
        try:
            return self._context_for(sample_id, assistant_output, messages)
        except Exception as exc:
            logger.warning(
                "repo_context_fallback sample_id={} kind=none reason=unexpected error={}",
                sample_id,
                f"{type(exc).__name__}: {exc}",
            )
            return GroundingContext(context=None, kind="none", reason="unexpected")

    def prefetch(self, sample_ids: list[str]) -> dict[str, int]:
        instances: dict[str, str] = {}
        for sample_id in sample_ids:
            try:
                shard_name, row_idx, _ = parse_sample_id(sample_id)
            except ValueError:
                continue
            source_iid = self._iid_for(shard_name, row_idx)
            if source_iid is not None:
                instances[source_iid[1]] = source_iid[0]

        def warm(instance_id: str, source: str) -> bool:
            try:
                ref = parse_instance(source, instance_id)
                if ref is None:
                    return False
                resolved = self._resolve_sha(ref)
                if resolved is None:
                    return False
                return self._ensure_snapshot(*resolved) is not None
            except Exception as exc:
                logger.warning(
                    "repo_context_prefetch_instance_failed instance={} error={}",
                    instance_id,
                    f"{type(exc).__name__}: {exc}",
                )
                return False

        ready = 0
        if instances:
            with ThreadPoolExecutor(max_workers=8) as pool:
                ready = sum(pool.map(lambda item: warm(*item), instances.items()))
        summary = {"samples": len(sample_ids), "instances": len(instances), "ready": ready}
        logger.info(
            "repo_context_prefetch_done samples={} instances={} ready={}",
            summary["samples"],
            summary["instances"],
            summary["ready"],
        )
        return summary

    def repo_context_for_instance(
        self,
        source: str,
        instance_id: str,
        assistant_output: str,
        messages: list[dict[str, str]] | None = None,
        fmt: str = "",
        recorded: int | None = None,
    ) -> GroundingContext:
        """The grounding for a command of an instance's session. `recorded` is how many of the
        transcript's commands ran on a real machine (the rest were answered by the simulator)."""
        ref = parse_instance(source, instance_id)
        if ref is None:
            return GroundingContext(context=None, kind="none", reason="instance_unparsed")
        resolved = self._resolve_sha(ref)
        if resolved is None:
            return GroundingContext(context=None, kind="none", reason="sha_unresolved")
        owner, repo, sha = resolved
        snapshot = self._ensure_snapshot(owner, repo, sha)
        if snapshot is None:
            return GroundingContext(context=None, kind="none", reason="snapshot_unavailable")
        base = list(_load_listing(str(snapshot / _LISTING_NAME)))
        root = session_root(messages)
        overlay = build_overlay(
            messages, base, lambda rel: self._read_snapshot_file(snapshot, rel), root, recorded
        )
        scaffold = scaffold_for(source, fmt, instance_id)
        overlay.columns = scaffold.columns
        overlay.pycache = overlay.pycache or scaffold.pycache
        overlay.editor = scaffold.editor
        overlay.errors = scaffold.errors
        overlay.shell = scaffold.shell
        listing = overlay.listing()
        meta = self.git_meta(snapshot, source, owner, repo, sha, scaffold, instance_id)
        command = first_bash_block(assistant_output)
        attested = attested_paths(messages)
        block, exact, returncode = self._build_repo_block(
            snapshot,
            listing,
            command,
            overlay,
            fmt,
            meta,
            root=root,
            attested=attested,
        )
        present, missing = _referenced_paths(command, listing)
        parts = None
        if exact is None and not (overlay.editor and _EDITOR_READ.fullmatch(command.strip())):
            parts = self._parts(snapshot, command, overlay, fmt, meta, root, attested)
        for part in parts or []:
            if part["kind"] == "exact":
                part["output"] = _emptied_blank_lines(part["output"], fmt)
        return GroundingContext(
            context=block,
            kind="repo",
            exact_output=_emptied_blank_lines(exact, fmt),
            exact_returncode=returncode,
            state=overlay.state(block, set(present) | set(missing)),
            leading_output=None
            if exact is not None
            else self._leading_output(snapshot, command, overlay, meta, root),
            parts=parts,
        )

    def _context_for(
        self, sample_id: str, assistant_output: str, messages: list[dict[str, str]] | None = None
    ) -> GroundingContext:
        try:
            shard_name, row_idx, turn_idx = parse_sample_id(sample_id)
        except ValueError:
            return GroundingContext(context=None, kind="none", reason="bad_sample_id")
        reason = "iid_unresolved"
        source_iid = self._iid_for(shard_name, row_idx)
        if source_iid is not None:
            result = self.repo_context_for_instance(
                *source_iid,
                assistant_output,
                messages,
                detect_format(sample_id, messages),
                recorded=turn_idx,
            )
            if result.context is not None:
                return result
            reason = result.reason or "repo_unavailable"
        block = self._trajectory_block(shard_name, row_idx, turn_idx)
        if block is not None:
            logger.info(
                "repo_context_fallback sample_id={} kind=trajectory reason={}", sample_id, reason
            )
            return GroundingContext(
                context=block,
                kind="trajectory",
                reason=reason,
                state=hashlib.sha1(block.encode("utf-8", "replace")).hexdigest(),
            )
        logger.warning("repo_context_fallback sample_id={} kind=none reason={}", sample_id, reason)
        return GroundingContext(context=None, kind="none", reason=reason)

    def _iid_for(self, shard_name: str, row_idx: int) -> tuple[str, str] | None:
        if self.settings.dataset_manifest_path:
            resolved = self._iid_from_manifest(shard_name, row_idx)
            if resolved is not None:
                return resolved
        if self.settings.dataset_root:
            iid = _iid_from_parquet(self.settings.dataset_root, shard_name, row_idx)
            if iid:
                return (shard_name.split("/data/", 1)[0], iid)
        return None

    def _iid_from_manifest(self, shard_name: str, row_idx: int) -> tuple[str, str] | None:
        try:
            shards = self._manifest_shards()
        except Exception as exc:
            if not self._manifest_error_logged:
                self._manifest_error_logged = True
                logger.warning(
                    "repo_context_manifest_unavailable path={} error={}",
                    self.settings.dataset_manifest_path,
                    f"{type(exc).__name__}: {exc}",
                )
            return None
        entry = shards.get(shard_name)
        if entry is None:
            return None
        source, rows_meta = entry
        if not 0 <= row_idx < len(rows_meta):
            return None
        iid = rows_meta[row_idx].get("iid") if isinstance(rows_meta[row_idx], dict) else None
        return (source, str(iid)) if iid else None

    def _manifest_shards(self) -> dict[str, tuple[str, list]]:
        if self._shards is None:
            with self._manifest_lock:
                if self._shards is None:
                    manifest = load_manifest_file(
                        self.settings.dataset_manifest_path,
                        expected_sha256=self.settings.dataset_manifest_hash,
                    )
                    shards: dict[str, tuple[str, list]] = {}
                    for source in manifest.get("sources", []):
                        for shard in source.get("shards", []):
                            shards[shard["path"]] = (
                                str(source.get("name", "")),
                                shard.get("rows_meta") or [],
                            )
                    self._shards = shards
        return self._shards

    def _resolve_sha(self, ref: RepoRef) -> tuple[str, str, str] | None:
        ref = _smith_mirror(ref) or ref
        # Affine's Rust swesmith machines sat at the branch head where mini-coder-rs's sat at `Bug
        # Patch`: the same instance id resolves apart, so neither serves the other's tree
        own = ref.source.startswith(_OWN_SHA_CACHE)
        cache_name = f"{_safe_name(ref.instance_id)}{'@affine' if own else ''}.json"
        cache_path = self._shas_dir / cache_name
        cached = _read_cached_sha(cache_path)
        if cached is not None:
            if cached.get("sha"):
                return cached["owner"], cached["repo"], cached["sha"]
            if self._negative_fresh(cached):
                return None
        with self._key_lock(f"sha:{ref.instance_id}"):
            cached = _read_cached_sha(cache_path)
            if cached is not None and cached.get("sha"):
                return cached["owner"], cached["repo"], cached["sha"]
            owner, repo = ref.owner, ref.repo
            try:
                try:
                    sha = self._sha_from_api(owner, repo, ref)
                except _NotFound:
                    data = self._github_json(f"/repos/{owner}/{repo}")
                    owner, repo = data["full_name"].split("/", 1)
                    sha = self._sha_from_api(owner, repo, ref)
            except Exception as exc:
                permanent = _is_permanent_github_error(exc)
                logger.info(
                    "repo_context_sha_unresolved instance={} kind={} error={}",
                    ref.instance_id,
                    "permanent" if permanent else "transient",
                    f"{type(exc).__name__}: {exc}",
                )
                _write_json_atomic(
                    cache_path,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": time.time(),
                        "kind": "permanent" if permanent else "transient",
                        "rule": _SHA_RULE,
                    },
                )
                return None
            _write_json_atomic(
                cache_path, {"owner": owner, "repo": repo, "sha": sha, "rule": _SHA_RULE}
            )
            return owner, repo, sha

    @staticmethod
    def _negative_fresh(cached: dict) -> bool:
        ttl = _NEGATIVE_TTL_SECONDS if cached.get("kind") == "permanent" else _TRANSIENT_TTL_SECONDS
        return time.time() - float(cached.get("failed_at", 0)) < ttl

    def git_meta(
        self,
        snapshot: Path,
        source: str,
        owner: str,
        repo: str,
        sha: str,
        scaffold: Scaffold = Scaffold(),
        instance_id: str = "",
    ) -> GitMeta:
        # a swesmith mirror's own history is the task's answer (`Bug Patch`, `Remove F2P Tests`):
        # it is served as one commit, the upstream commit the mirror was built from when GitHub
        # still has it and its subject does not name the mirrored PR
        mirrored = owner == _SMITH_MIRROR_OWNER
        shown_owner, shown_repo = _smith_upstream(repo) if mirrored else (owner, repo)
        head, subject, root = sha, _SMITH_SQUASHED_SUBJECT, ()
        if mirrored:
            point = self._upstream_point(shown_owner, shown_repo, repo.rpartition(".")[2])
            if point is not None and not _names_mirrored_pr(point["subject"], instance_id):
                head, subject = point["sha"], point["subject"]
                root = root_commit_header(point)
        return GitMeta(
            sha=head,
            owner=shown_owner,
            repo=shown_repo,
            # a squashed history is a local repository on the branch its upstream is named for
            branch=scaffold.branch
            or (
                DEFAULT_BRANCH if scaffold.squashed else self._default_branch(snapshot, owner, repo)
            ),
            detached=scaffold.detached or _DETACHED_SOURCE.search(source or "") is not None,
            detached_at=scaffold.detached_at,
            abbrev=scaffold.abbrev,
            decorate=scaffold.decorate,
            squashed=scaffold.squashed,
            tracking=scaffold.tracking,
            root_header=root,
            history=(
                (lambda path: _squashed_history(head, subject))
                if mirrored
                else (lambda path: self._commit_history(snapshot, owner, repo, sha, path))
            ),
            commit_patch=(
                None
                if mirrored
                else (lambda rev: self._served_patch(snapshot, owner, repo, sha, rev))
            ),
        )

    def _served_patch(
        self, snapshot: Path, owner: str, repo: str, sha: str, rev: str
    ) -> str | None:
        served = [self._commit_history(snapshot, owner, repo, sha, None)]
        served += [_read_json(path) for path in snapshot.glob(_HISTORY_NAME.format(key="*"))]
        if not any(
            entry["sha"].startswith(rev)
            for payload in served
            if isinstance(payload, dict)
            for entry in payload.get("commits") or []
        ):
            return None
        return self._commit_patch(snapshot, owner, repo, rev)

    def _upstream_point(self, owner: str, repo: str, short: str) -> dict | None:
        """The upstream commit a swesmith mirror was built from: sha, subject and the header
        `git show` prints for it (author, author date in UTC, message body)."""
        cache_path = self._upstream_dir / f"{_safe_name(owner)}__{_safe_name(repo)}__{short}.json"
        cached = _read_json(cache_path)
        if isinstance(cached, dict):
            if cached.get("sha") and "author" in cached:
                return cached
            absent = cached.get("kind") == "absent"
            ttl = _UPSTREAM_ABSENT_TTL_SECONDS if absent else _TRANSIENT_TTL_SECONDS
            if time.time() - float(cached.get("failed_at", 0)) < ttl:
                return None
        try:
            data = self._github_json(f"/repos/{owner}/{repo}/commits/{short}")
            full = str(data.get("sha") or "")
            commit = data.get("commit") or {}
            subject, _, body = str(commit.get("message") or "").partition("\n")
            subject = subject.strip()
            if not (short and full.startswith(short) and subject):
                raise _NotFound(f"{owner}/{repo}@{short}")
            author = commit.get("author") or {}
        except Exception as exc:
            kind = "absent" if _is_permanent_github_error(exc) else "transient"
            logger.info(
                "repo_context_upstream_point_unavailable repo={}/{} commit={} kind={} error={}",
                owner,
                repo,
                short,
                kind,
                f"{type(exc).__name__}: {exc}",
            )
            _write_json_atomic(cache_path, {"sha": None, "failed_at": time.time(), "kind": kind})
            return None
        point = {
            "sha": full,
            "subject": subject,
            "author": f"{author.get('name') or ''} <{author.get('email') or ''}>",
            "date": str(author.get("date") or ""),
            "body": body.strip("\n"),
        }
        _write_json_atomic(cache_path, point)
        return point

    def _commit_patch(self, snapshot: Path, owner: str, repo: str, rev: str) -> str | None:
        cache_path = snapshot / _PATCH_NAME.format(key=_hashed(rev))
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8", errors="replace")
            return text or None
        try:
            text = self._github_text(
                f"/repos/{owner}/{repo}/commits/{quote(rev, safe='')}",
                "application/vnd.github.patch",
            )
        except Exception as exc:
            logger.info(
                "repo_context_patch_unavailable repo={}/{} rev={} error={}",
                owner,
                repo,
                rev,
                f"{type(exc).__name__}: {exc}",
            )
            return None
        if len(text) > _MAX_PATCH_CHARS:
            return None
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, cache_path)
        return text or None

    def _commit_history(
        self, snapshot: Path, owner: str, repo: str, sha: str, path: str | None
    ) -> dict | None:
        key = _safe_name(path) if path else "__repo__"
        cache_path = snapshot / _HISTORY_NAME.format(key=_hashed(key))
        cached = _read_json(cache_path)
        if isinstance(cached, dict):
            if cached.get("commits") is not None:
                return cached
            if self._failure_fresh(cache_path):
                return None
        query = f"/repos/{owner}/{repo}/commits?sha={sha}&per_page={_HISTORY_PAGE}"
        if path:
            query += f"&path={quote(path, safe='')}"
        try:
            data = self._github_json(query)
        except Exception as exc:
            kind = "absent" if isinstance(exc, _NotFound) else "transient"
            logger.info(
                "repo_context_history_unavailable repo={}/{} path={} kind={} error={}",
                owner,
                repo,
                path or "-",
                kind,
                f"{type(exc).__name__}: {exc}",
            )
            _write_json_atomic(
                cache_path, {"commits": None, "failed_at": time.time(), "kind": kind}
            )
            return None
        if not isinstance(data, list):
            _write_json_atomic(
                cache_path, {"commits": None, "failed_at": time.time(), "kind": "absent"}
            )
            return None
        payload = {
            "commits": [
                {
                    "sha": str(entry.get("sha") or ""),
                    "subject": str((entry.get("commit") or {}).get("message") or "").split("\n")[0],
                }
                for entry in data
                if entry.get("sha")
            ],
            "complete": len(data) < _HISTORY_PAGE,
        }
        _write_json_atomic(cache_path, payload)
        return payload

    def _default_branch(self, snapshot: Path, owner: str, repo: str) -> str:
        meta_path = snapshot / _REPO_META_NAME
        cached = _read_json(meta_path)
        if isinstance(cached, dict) and cached.get("default_branch"):
            return str(cached["default_branch"])
        branch = DEFAULT_BRANCH
        try:
            data = self._github_json(f"/repos/{owner}/{repo}")
            branch = str(data.get("default_branch") or DEFAULT_BRANCH)
        except Exception as exc:
            logger.info(
                "repo_context_default_branch_unavailable repo={}/{} error={}",
                owner,
                repo,
                f"{type(exc).__name__}: {exc}",
            )
            return branch
        _write_json_atomic(meta_path, {"default_branch": branch})
        return branch

    def _sha_from_api(self, owner: str, repo: str, ref: RepoRef) -> str:
        if ref.pr is not None:
            data = self._github_json(f"/repos/{owner}/{repo}/pulls/{ref.pr}")
            return data["base"]["sha"]
        data = self._github_json(f"/repos/{owner}/{repo}/commits/{ref.commit}")
        # R2E-Gym ids (swe-hero, and Affine's PR sources) name the fix commit; the agent worked on
        # its parent
        if ref.source == "swe-hero" or ref.source.startswith("affine-"):
            return data["parents"][0]["sha"]
        if owner == _SMITH_MIRROR_OWNER and _smith_env_is_bug_patch(ref, data):
            return data["parents"][0]["sha"]
        return data["sha"]

    def _ensure_snapshot(self, owner: str, repo: str, sha: str) -> Path | None:
        key = f"{_safe_name(owner)}__{_safe_name(repo)}__{sha[:12]}"
        final = self._snapshots_dir / key
        if (final / _DONE_MARKER).exists():
            return final
        failed_path = self._snapshots_dir / f"{key}.failed.json"
        if self._failure_fresh(failed_path):
            return None
        with self._key_lock(f"snapshot:{key}"):
            if (final / _DONE_MARKER).exists():
                return final
            if self._failure_fresh(failed_path):
                return None
            self._enforce_cache_limit()
            self._snapshots_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(dir=self._snapshots_dir, prefix=f".{key}.partial-"))
            try:
                with tempfile.NamedTemporaryFile(dir=self._snapshots_dir, suffix=".tar.gz") as tar:
                    self._download_tarball(owner, repo, sha, Path(tar.name))
                    listing, extracted_bytes = self._extract_tarball(Path(tar.name), tmp_dir)
                (tmp_dir / _LISTING_NAME).write_text(json.dumps(listing))
                (tmp_dir / _DONE_MARKER).write_text(json.dumps({"bytes": extracted_bytes}))
                os.replace(tmp_dir, final)
            except Exception as exc:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                if (final / _DONE_MARKER).exists():
                    return final
                ttl_kind = "oversized" if isinstance(exc, _SnapshotTooLarge) else "transient"
                logger.info(
                    "repo_context_snapshot_failed repo={}/{} sha={} kind={} error={}",
                    owner,
                    repo,
                    sha[:12],
                    ttl_kind,
                    f"{type(exc).__name__}: {exc}",
                )
                _write_json_atomic(
                    failed_path,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": time.time(),
                        "kind": ttl_kind,
                    },
                )
                return None
            return final

    @staticmethod
    def _failure_fresh(failed_path: Path) -> bool:
        failed = _read_json(failed_path)
        if failed is None:
            return False
        settled = failed.get("kind") in ("oversized", "absent")
        ttl = _NEGATIVE_TTL_SECONDS if settled else _TRANSIENT_TTL_SECONDS
        return time.time() - float(failed.get("failed_at", 0)) < ttl

    def _download_tarball(self, owner: str, repo: str, sha: str, dest: Path) -> None:
        url = f"{_API_BASE}/repos/{owner}/{repo}/tarball/{sha}"
        max_bytes = self.settings.max_snapshot_mb * 1024 * 1024
        with self._github_semaphore:
            with self._client.stream("GET", url, headers=self._auth_headers()) as response:
                if response.status_code == 404:
                    raise _NotFound(url)
                response.raise_for_status()
                declared = int(response.headers.get("content-length") or 0)
                if declared > max_bytes:
                    raise _SnapshotTooLarge(f"tarball {declared} bytes > {max_bytes}")
                written = 0
                with dest.open("wb") as out:
                    for chunk in response.iter_bytes(1 << 20):
                        written += len(chunk)
                        if written > max_bytes:
                            raise _SnapshotTooLarge(f"tarball exceeds {max_bytes} bytes")
                        out.write(chunk)

    def _enforce_cache_limit(self) -> None:
        limit = int(self.settings.max_cache_gb * 1024**3)
        if limit <= 0:
            return
        usage = 0
        for marker in self._snapshots_dir.glob(f"*/{_DONE_MARKER}"):
            data = _read_json(marker)
            usage += int(data.get("bytes", 0)) if data else 0
        if usage < limit:
            return
        logger.warning(
            "repo_context_cache_cleared usage_bytes={} limit_bytes={} dir={}",
            usage,
            limit,
            self._snapshots_dir,
        )
        shutil.rmtree(self._snapshots_dir, ignore_errors=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        _load_listing.cache_clear()

    def _extract_tarball(self, tar_path: Path, dest_dir: Path) -> tuple[list[str], int]:
        max_extracted = self.settings.max_snapshot_mb * 1024 * 1024 * 4
        total = 0
        paths: list[str] = []
        with tarfile.open(tar_path, mode="r:gz") as tar:
            for member in tar:
                if len(paths) >= _MAX_MEMBERS:
                    raise _SnapshotTooLarge(f"tarball has more than {_MAX_MEMBERS} files")
                if not (member.isreg() or member.issym()) or member.name.startswith("/"):
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or any(part in ("..", "") for part in parts):
                    continue
                if parts[1] in (_LISTING_NAME, _DONE_MARKER):
                    continue
                if member.issym() or member.size > _MAX_MEMBER_BYTES:
                    # listed without its text: a link, or a file too large to keep
                    paths.append("/".join(parts[1:]))
                    continue
                total += member.size
                if total > max_extracted:
                    raise _SnapshotTooLarge(f"extracted size exceeds {max_extracted} bytes")
                rel = "/".join(parts[1:])
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                paths.append(rel)
        return sorted(paths), total

    _LISTING_MIN_CHARS = 8000

    def _run_command(
        self, snapshot_dir: Path, cmd: str, overlay: Overlay, root: str, terminal: bool = True
    ) -> SearchResult | None:
        plan = parse_search(cmd)
        if isinstance(plan, ParseFailure):
            return None
        result = run_search(
            plan,
            overlay.text,
            overlay.listing(),
            size_file=lambda rel: self._file_size(snapshot_dir, rel, overlay),
            root=root,
            overlay=overlay,
            terminal=terminal,
        )
        return None if isinstance(result, ParseFailure) else result

    def _stage_answer(
        self,
        snapshot_dir: Path,
        working: Overlay,
        stage: Stage,
        cwd: str | None,
        fmt: str,
        meta: GitMeta,
        root: str,
        alone: bool,
    ) -> tuple[str, int] | None:
        """The exact output, as the terminal shows it, and exit status of one stage run in `cwd`
        against `working`; None when this cannot tell. `alone`: the stage is the whole command."""
        command = unwrapped(stage.command)
        if command.name == "cd" and len(stage.pipeline) == 1:
            return cd_outcome(working, command, cwd)
        canned = _canned(stage)
        if canned is not None:
            return canned
        if command.name == "git":
            plan = parse_git(stage.text)
            if cwd != "" or isinstance(plan, ParseFailure):
                return None
            result = run_git(plan, working, working.read_base, working.listing(), meta)
            if not isinstance(result, GitResult) or not result.exact:
                return None
            return (result.output + "\n" if result.output else ""), result.returncode
        quiet = quiet_outcome(working, stage, cwd) or self._redirected_read(
            snapshot_dir, working, stage, cwd, root
        )
        if quiet is not None:
            # the openhands scaffold writes a heredoc with its editor, which reports back
            heredoc = any(c.heredoc is not None for c in stage.pipeline)
            # a heredoc write that is the whole command was the editor's own in the datasets
            # converted from it, which reports the file it created
            return None if heredoc and (fmt == OPENHANDS or (working.editor and alone)) else quiet
        if cwd != "" and not _absolute_search(stage.text):
            return None
        result = self._run_command(snapshot_dir, stage.text, working, root)
        if result is None or result.returncode is None:
            return None
        return result.raw, result.returncode

    def _redirected_read(
        self, snapshot_dir: Path, working: Overlay, stage: Stage, cwd: str | None, root: str
    ) -> tuple[str, int] | None:
        """A search whose output goes to a file (`grep x f > out`): it prints nothing, and its
        status is the search's, when the search reports no missing file."""
        last = stage.pipeline[-1]
        target = next((r for r in last.redirects if r.fd == 1 and r.writes_file), None)
        rest = [r for r in last.redirects if r is not target]
        if target is None or cwd != "" or rest or not writable(working, target.target, cwd):
            return None
        words = " | ".join(shlex.join(command.argv) for command in stage.pipeline)
        result = self._run_command(snapshot_dir, words, working, root, terminal=False)
        if result is None or result.missing or result.returncode is None:
            return None
        return "", result.returncode

    def _execute(
        self,
        snapshot_dir: Path,
        cmd: str,
        overlay: Overlay,
        fmt: str,
        meta: GitMeta,
        root: str,
    ) -> tuple[list[_Ran], Overlay] | None:
        """Run a command's stages in order against a copy of the session's checkout, each after
        the changes of the stages before it, `&&` and `||` gating a stage on the exit status of
        the one before; the stages that run or may run, and the checkout they leave.

        A stage gated on one this could not answer may or may not run: its answer is what it
        would print, and its changes are applied as ones that may have happened. None when the
        shell parser refuses the command."""
        parsed = parse_command(cmd)
        if isinstance(parsed, Unsupported) or not parsed.stages:
            return None
        working = overlay.copy()
        cwd, last, errexit = working.cwd, 0, False
        alone = len(parsed.stages) == 1
        out: list[_Ran] = []
        for index, stage in enumerate(parsed.stages):
            separator, name = stage.separator, stage.command.name
            following = parsed.stages[index + 1].separator if index + 1 < len(parsed.stages) else ""
            if last is not None and ((separator == AND and last) or (separator == OR and not last)):
                continue
            surely = last is not None or separator not in (AND, OR)
            before = working.copy()
            answer = self._stage_answer(snapshot_dir, working, stage, cwd, fmt, meta, root, alone)
            code = answer[1] if answer is not None and surely else None
            apply_stage(working, stage, cwd, surely, code)
            out.append(_Ran(stage, answer, surely, index, before))
            if name == "cd" and len(stage.pipeline) == 1:
                cwd = cd_target(stage.command, cwd) if code == 0 else (cwd if code else None)
            elif name in ("pushd", "popd"):
                cwd = None
            errexit = errexit_after(stage.command, errexit)
            last = code
            if name == "exit" or (errexit and last and following not in (AND, OR)):
                break
            if errexit and last is None and following not in (AND, OR):
                for position, rest in enumerate(parsed.stages[index + 1 :], index + 1):
                    out.append(_Ran(rest, None, False, position, working.copy()))
                    apply_stage(working, rest, cwd, certain=False)
                break
        return out, working

    def _parts(
        self,
        snapshot_dir: Path,
        cmd: str,
        overlay: Overlay,
        fmt: str,
        meta: GitMeta,
        root: str,
        attested: set[str],
    ) -> list[dict] | None:
        """Every stage the command may run, in order, as the parts the repository answers and the
        gaps left to simulate, each with the `&&`, `||` or `;` that joins it to the one before:
        which of them run is the simulator's to settle, from the gaps' exit statuses (see
        `stitch_parts`). An exact part holds the text the terminal shows for its stage (final
        line break included) and its exit status; a stage after a gap is answered only when its
        answer holds whether or not the stages before it ran. A gap holds the command text of
        consecutive stages this cannot answer and the context to simulate them with (the
        checkout as the stages before them may have left it). None when splitting would not
        help: nothing answered prints, there are more than `_MAX_GAP_GROUPS` gaps, or what runs
        after a gap is not decided by exit statuses alone (`set -e`, an `exit` that may run)."""
        result = self._execute(snapshot_dir, cmd, overlay, fmt, meta, root)
        parsed = parse_command(cmd)
        if not result or isinstance(parsed, Unsupported):
            return None
        executed, _ = result
        groups = _gap_groups(executed)
        answered = any(ran.answer and ran.answer[0] for ran in executed)
        errexit = any(errexit_after(stage.command, False) for stage in parsed.stages)
        # `exit` ends the command where it runs; one that may run leaves the stages after it out
        exits = any(ran.stage.command.name == "exit" and not ran.surely for ran in executed)
        if not groups or len(groups) > _MAX_GAP_GROUPS or not answered or errexit or exits:
            return None
        parts: list[dict] = []
        for position, ran in enumerate(executed):
            separator = ran.stage.separator if parts else ""
            if ran.answer is not None:
                output, returncode = ran.answer
                parts.append(
                    {
                        "kind": "exact",
                        "separator": separator,
                        "output": output,
                        "returncode": returncode,
                    }
                )
                continue
            group = next((g for g in groups if g[0] is ran), None)
            if group is None:
                continue
            text = _joined_text([member.stage for member in group])
            context = self._build_repo_block(
                snapshot_dir, ran.before.listing(), text, ran.before, fmt, meta, root, attested
            )[0]
            earlier = [
                f"$ {p.stage.text}\n"
                + (p.answer[0].removesuffix("\n") if p.answer else PRIOR_GAP_STAGE)
                for p in executed[:position]
            ]
            if earlier:
                context = PRIOR_STAGES_HEADER + "\n" + "\n\n".join(earlier) + "\n\n" + context
            parts.append(
                {"kind": "gap", "separator": separator, "command": text, "context": context}
            )
        return parts

    def _leading_output(
        self, snapshot_dir: Path, cmd: str, overlay: Overlay, meta: GitMeta, root: str
    ) -> str | None:
        """The joined output of every stage before the last of an `&&` chain whose last stage has
        its own head/tail cap, or None when any of them cannot be run exactly or fails: a failing
        stage stops the chain, and an unrun one leaves the length of what precedes the last stage
        unknown."""
        parsed = parse_command(cmd)
        if isinstance(parsed, Unsupported) or len(parsed.stages) < 2:
            return None
        if any(stage.separator != AND for stage in parsed.stages[1:]):
            return None
        if not command_contract(parsed.stages[-1].text):
            return None
        executed = (self._execute(snapshot_dir, cmd, overlay, "", meta, root) or ([], None))[0]
        earlier = executed[: len(parsed.stages) - 1]
        if len(earlier) < len(parsed.stages) - 1 or any(
            ran.answer is None or ran.answer[1] != 0 for ran in earlier
        ):
            return None
        return "".join(ran.answer[0] for ran in earlier).removesuffix("\n")

    def _run_chain(
        self,
        snapshot_dir: Path,
        cmd: str,
        overlay: Overlay,
        fmt: str,
        meta: GitMeta,
        root: str = "",
    ) -> tuple[tuple[str, str | None, int | None] | None, str, Overlay]:
        """The command's exact answer when every stage computed; otherwise, as evidence for the
        simulator, the stages in order: each computed one with its output, each gap marked (at
        most `_MAX_GAP_GROUPS` runs of gaps, beyond which the evidence would not help). Last, the
        checkout as the stages leave it, which the simulator's facts describe."""
        result = self._execute(snapshot_dir, cmd, overlay, fmt, meta, root)
        if not result:
            return None, "", overlay
        executed, working = result
        if all(ran.answer is not None and ran.surely for ran in executed):
            raw = "".join(ran.answer[0] for ran in executed)
            text = raw.removesuffix("\n")
            exact = _scaffold_truncate(raw, fmt) if text else ""
            block = COMPUTED_HEADER + "\n" + exact + "\n" if text else COMPUTED_EMPTY
            returncode = executed[-1].answer[1]
            computed = (_truncate(block, self.settings.max_context_chars), exact, returncode)
            return computed, "", working
        if len(_gap_groups(executed)) > _MAX_GAP_GROUPS or not any(
            ran.answer and ran.answer[0] for ran in executed
        ):
            return None, "", working
        shown = []
        for ran in executed:
            stage, answer, surely = ran.stage, ran.answer, ran.surely
            joined = f"{stage.separator} " if stage.separator in (AND, OR) else ""
            if answer is None:
                note = CHAIN_GAP if surely else CHAIN_MAYBE_GAP
                shown.append(f"{joined}$ {stage.text}\n{note}")
            else:
                body = _truncate(answer[0].removesuffix("\n"), self.settings.max_file_chars)
                note = "" if surely else f"\n{CHAIN_MAYBE_RUN}"
                shown.append(f"{joined}$ {stage.text}\n{body or CHAIN_NO_OUTPUT}{note}")
        evidence = "\n" + CHAIN_EVIDENCE_HEADER + "\n" + "\n\n".join(shown) + "\n"
        return None, evidence, working

    def _build_repo_block(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        fmt: str = "",
        meta: GitMeta | None = None,
        root: str = "",
        attested: set[str] | None = None,
    ) -> tuple[str, str | None, int | None]:
        meta = meta or GitMeta()
        viewed = _EDITOR_READ.fullmatch(cmd.strip()) if overlay.editor else None
        evidence = ""
        if viewed is not None:
            # the editor's view, or none at all: bash's answer is not what the scaffold shows
            view = _editor_answer(overlay, viewed)
            if view is not None:
                return "", view, 0
        else:
            computed_chain, evidence, overlay = self._run_chain(
                snapshot_dir, cmd, overlay, fmt, meta, root
            )
            if computed_chain is not None:
                return computed_chain
        listing = overlay.listing()
        evidence += self._git_hints(snapshot_dir, listing, cmd, overlay, meta)
        listing_paths, _ = _filter_listing(listing, cmd)
        present, missing = _referenced_paths(cmd, listing, overlay.text)
        missing, vouched = _split_attested(missing, attested)
        # a path the session may have made, or made a directory of, is not reported absent
        missing = [path for path in missing if overlay.kind(repo_path(path) or path) is None]
        show = lambda path: f"{root}/{path}" if root else f"./{path}"  # noqa: E731
        listing_header = LISTING_HEADER_ROOTED.format(root=root) if root else LISTING_HEADER
        # the checkout's own doubts; outside it only the session's files are ever shown
        doubtful = [p for p in sorted(overlay.unsure) + sorted(overlay.maybe) if p[:1] != "/"]
        if doubtful:
            shown = [f"- {show(p) if p else '(the whole repository)'}" for p in doubtful[:10]]
            listing_header += UNCERTAIN_LISTING_NOTE + "\n".join(shown) + "\n"
        missing_text = (
            "\n"
            + NOT_PRESENT_HEADER
            + "\n".join(
                f"- {show(p) if root and not p.startswith(('/', '..')) else p}"
                f"   ->   No such file or directory"
                for p in missing
            )
            + "\n"
            if missing
            else ""
        )
        vouched_text = (
            "\n" + SESSION_PATHS_HEADER + "\n".join(f"- {p}" for p in vouched) + "\n"
            if vouched
            else ""
        )

        contents_budget = (
            self.settings.max_context_chars
            - len(listing_header)
            - len(missing_text)
            - len(vouched_text)
            - len(evidence)
            - self._LISTING_MIN_CHARS
        )
        wants_numbers = bool(_WANTS_LINE_NUMBERS.search(cmd or ""))
        grep_style = bool(_GREP_STYLE.search(cmd or ""))
        note = LINE_NUMBER_NOTE_GREP if grep_style else LINE_NUMBER_NOTE_COLUMN
        contents_header = CONTENTS_HEADER + (note if wants_numbers else "")
        contents_parts: list[str] = []
        contents_used = len(contents_header) + 2
        for path in present[: self.settings.max_files]:
            text = overlay.text(path)
            stale = False
            if text is None and overlay.is_dirty(path):
                # edited by something we cannot model: the pre-edit text still beats nothing
                text, stale = self._read_snapshot_file(snapshot_dir, path), True
            if text is None:
                continue
            body = _truncate(text, self.settings.max_file_chars)
            if wants_numbers:
                body = number_lines(body, grep_style)
            label = f"{show(path)}{PRE_EDIT_SUFFIX if stale else ''}"
            part = f"--- {label} ---\n{body}\n"
            if contents_used + len(part) > contents_budget:
                break
            contents_parts.append(part)
            contents_used += len(part) + 1
        contents_text = (
            "\n" + contents_header + "\n" + "\n".join(contents_parts) if contents_parts else ""
        )

        listing_budget = (
            self.settings.max_context_chars
            - len(listing_header)
            - len(contents_text)
            - len(missing_text)
            - len(vouched_text)
            - len(evidence)
            - 64
        )
        if not listing_paths:
            listing_text = (
                "(no files in this repository match the command's filters — "
                "the exploration output is empty)"
            )
        else:
            lines: list[str] = []
            used = 0
            for path in listing_paths[: self.settings.max_paths]:
                line = show(path)
                if used + len(line) + 1 > listing_budget:
                    break
                lines.append(line)
                used += len(line) + 1
            over = len(listing_paths) - len(lines)
            listing_text = "\n".join(lines)
            if over > 0:
                listing_text += f"\n... (+{over} more matching files)"

        block = (
            listing_header
            + "\n"
            + listing_text
            + "\n"
            + contents_text
            + vouched_text
            + missing_text
        )
        return _truncate(evidence + block, self.settings.max_context_chars), None, None

    def _git_hints(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        meta: GitMeta,
    ) -> str:
        read_base = lambda rel: self._read_snapshot_file(snapshot_dir, rel)  # noqa: E731
        note = explain_git(cmd, overlay, listing, read_base, meta)
        hint = "\n" + GIT_SEMANTICS_HEADER + "\n" + note if note else ""
        evidence = _truncate(
            git_evidence(cmd, overlay, read_base, listing, meta), self.settings.max_file_chars
        )
        ledger = ledger_block(overlay.git) if is_git_command(cmd) else ""
        return hint + evidence + ledger

    def _file_size(self, snapshot_dir: Path, rel_path: str, overlay: Overlay) -> int | None:
        if overlay.kind(rel_path) != FILE or overlay.is_dirty(rel_path):
            return None
        if (held := overlay.read(rel_path)) is not None:
            return len(held.encode("utf-8", "replace"))
        if rel_path.startswith("/"):
            return None
        root = snapshot_dir.resolve()
        try:
            target = (snapshot_dir / rel_path).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                return None
            return target.stat().st_size
        except OSError:
            return None

    def _read_snapshot_file(self, snapshot_dir: Path, rel_path: str) -> str | None:
        root = snapshot_dir.resolve()
        try:
            target = (snapshot_dir / rel_path).resolve()
        except OSError:
            return None
        if not target.is_relative_to(root) or not target.is_file():
            return None
        try:
            return target.read_text(errors="replace")
        except OSError:
            return None

    def _trajectory_block(self, shard_name: str, row_idx: int, turn_idx: int) -> str | None:
        if not self.settings.dataset_root:
            return None
        try:
            row = read_parquet_row(Path(self.settings.dataset_root) / shard_name, row_idx)
        except Exception as exc:
            logger.info(
                "repo_context_trajectory_unavailable shard={} row={} error={}",
                shard_name,
                row_idx,
                f"{type(exc).__name__}: {exc}",
            )
            return None
        normalized = {key: unwrap_column(value) for key, value in row.items()}
        turns = extract_turns(normalized)
        assistant_positions = [i for i, turn in enumerate(turns) if turn_role(turn) == "assistant"]
        pairs: list[tuple[str, str]] = []
        for assistant_index, position in enumerate(assistant_positions):
            if assistant_index < turn_idx:
                continue
            command = first_bash_block(turn_content(turns[position]))
            if not command or position + 1 >= len(turns):
                continue
            if turn_role(turns[position + 1]) == "assistant":
                continue
            observation = turn_content(turns[position + 1]).strip()
            if not observation:
                continue
            if ANY_MARKER_RE.search(command) or ANY_MARKER_RE.search(observation):
                continue
            pairs.append((command, observation))
            if len(pairs) >= self.settings.max_trajectory_pairs:
                break
        if not pairs:
            return None
        parts = [TRAJECTORY_HEADER]
        for step, (command, observation) in enumerate(pairs, 1):
            parts.append(
                f"REFERENCE STEP {step}:\n```bash\n{command}\n```\n\n"
                f"ENVIRONMENT OBSERVATION:\n{_truncate(observation, self.settings.max_file_chars)}"
            )
        return _truncate("\n\n".join(parts), self.settings.max_context_chars)

    def _auth_headers(self) -> dict[str, str]:
        token = (
            self.settings.github_token
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or ""
        )
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _github_text(self, path: str, accept: str) -> str:
        headers = {**self._auth_headers(), "Accept": accept}
        last_error: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                with self._github_semaphore:
                    response = self._client.get(_API_BASE + path, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            if response.status_code == 404:
                raise _NotFound(path)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"github {response.status_code} for {path}")
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            response.raise_for_status()
            return response.text
        raise RuntimeError(f"github request failed for {path}: {last_error}")

    def _github_json(self, path: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                with self._github_semaphore:
                    response = self._client.get(_API_BASE + path, headers=self._auth_headers())
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            if response.status_code == 404:
                raise _NotFound(path)
            if (
                response.status_code == 429
                or response.status_code >= 500
                or (
                    response.status_code == 403
                    and response.headers.get("x-ratelimit-remaining") == "0"
                )
            ):
                last_error = RuntimeError(f"github {response.status_code} for {path}")
                retry_after = float(response.headers.get("retry-after") or 0)
                time.sleep(min(30.0, max(retry_after, 1.5 * 2**attempt)))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"github request failed for {path}: {last_error}")

    def _key_lock(self, key: str) -> threading.Lock:
        with self._locks_mutex:
            return self._locks.setdefault(key, threading.Lock())


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _hashed(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
