"""The repo-context service against real bash, GNU coreutils, sed and git.

Each case runs its commands in a real git checkout, replays the transcript they make through
`build_overlay`, and then asks the service each probe command. An exact answer must be the real
tool's output and exit status to the byte; a declined one (left to the simulator) is always
acceptable, except where a case expects an answer, so that declining everything cannot pass.
"""

from __future__ import annotations

import asyncio
import functools
import random
import shutil
import string
import subprocess
import tempfile
from pathlib import Path

import pytest

from albedo_config import RepoContextSettings
from albedo_eval_service.repo_context_client import stitch_parts
from repo_context_service.command_search import ls_layouts
from repo_context_service.core import SCAFFOLDS, RepoContextService
from repo_context_service.git_sim import GitMeta
from repo_context_service.overlay import build_overlay
from repo_context_service.shell import parse_command

pytestmark = pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("bash", "git", "sed", "grep", "find")),
    reason="needs the real tools",
)

FILES = {
    "setup.py": "from setuptools import setup\n",
    "README.md": "# demo\npip install demo\n",
    "noeol.txt": "no newline",
    "pkg/app.py": "SIZE = 1\nNAME = 'app'\n",
    "pkg/util.py": "import os\ndef helper():\n    return 1\n",
    "pkg/sub/deep.py": "# deep\nSIZE = 3\n",
    ".gitignore": "*.log\nbuild/\n",
}
ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C",
    "GIT_AUTHOR_NAME": "a",
    "GIT_AUTHOR_EMAIL": "a@e",
    "GIT_COMMITTER_NAME": "a",
    "GIT_COMMITTER_EMAIL": "a@e",
}

# (commands the transcript ran, probes asked afterwards, probes that must be answered exactly)
CASES = {
    "a removed file is gone from listings, reads and git": (
        ["rm pkg/util.py"],
        ["ls pkg", "cat pkg/util.py", "git status --short", "git diff --stat"],
        ["ls pkg", "cat pkg/util.py", "git status --short"],
    ),
    "a moved directory carries its files": (
        ["mv pkg lib"],
        ["cat lib/app.py", "ls pkg", "find lib -type f | sort", "git status --short"],
        ["cat lib/app.py", "ls pkg", "git status --short"],
    ),
    "a stage after a failed && never ran": (
        ["cd missing && rm -rf pkg", "false && echo x > out.txt"],
        ["ls pkg", "cat out.txt", "git status --short"],
        ["ls pkg", "cat out.txt"],
    ),
    "rm refuses . and ..": (
        ["cd pkg && rm -rf ..; echo done"],
        ["ls", "cat setup.py"],
        ["cat setup.py"],
    ),
    "a glob with a trailing slash removes only directories": (
        ["rm -rf */"],
        ["ls", "cat setup.py"],
        ["ls", "cat setup.py"],
    ),
    "cp of several files into a missing directory copies nothing": (
        ["cp setup.py README.md nodir"],
        ["ls nodir", "git status --short"],
        ["ls nodir"],
    ),
    "sed edits in place the way GNU sed does": (
        ["sed -i '$a\\' noeol.txt", "sed -i 's/SIZE/\\\\&/' pkg/app.py", "sed -i '2,1d' README.md"],
        ["cat noeol.txt", "cat pkg/app.py", "cat README.md", "git diff"],
        ["cat noeol.txt"],
    ),
    "a heredoc written behind a cd lands in that directory": (
        ["cd pkg && cat > new.py <<'EOF'\nprint(1)\nEOF"],
        ["cat pkg/new.py", "cat new.py", "git status --short"],
        ["cat pkg/new.py", "cat new.py", "git status --short"],
    ),
    "a python program's writes are followed or left unknown": (
        [
            "python3 -c \"open('gen.txt', 'w').write('x')\"",
            # a parameter shadowing a module-level name must not be read as that name
            "python3 - <<'EOF'\npath = 'a.txt'\ndef fix(path):\n"
            "    open(path, 'w').write('y')\nfix('setup.py')\nEOF",
        ],
        ["cat gen.txt", "cat setup.py", "ls"],
        [],
    ),
    "ignored files stay out of status and git add": (
        [
            "echo x > debug.log && echo y > new.py && mkdir -p build && echo z > build/o.txt",
            "git add .",
        ],
        ["git status --short", "git status", "git diff --cached --name-only"],
        ["git status --short", "git diff --cached --name-only"],
    ),
    "an untracked directory shows as one entry": (
        ["mkdir -p fresh/deeper && touch fresh/deeper/a.py fresh/b.py"],
        ["git status --short", "git status"],
        ["git status --short"],
    ),
    "a stash takes deletions and staged new files, and pop brings them back": (
        ["rm README.md && echo n > new.py && git add new.py && echo more >> new.py", "git stash"],
        ["git status --short", "cat README.md", "ls new.py", "git diff --cached"],
        ["git status --short", "cat README.md"],
    ),
    "git mv is a staged rename": (
        ["git mv setup.py setup2.py"],
        ["git status --short", "git status", "git diff --cached", "cat setup.py"],
        ["git status --short", "git diff --cached"],
    ),
    "searches and reads behave like the real tools": (
        [],
        [
            "cat noeol.txt README.md",
            "cat pkg/app.py nothere.py pkg/util.py",
            "echo -n a; echo b",
            "cat noeol.txt && echo && cat README.md",
            "grep -n -e SIZE -e NAME pkg/app.py",
            "grep SIZE pkg",
            "grep -c SIZE pkg/app.py pkg/util.py",
            "grep -rn SIZE pkg | sort",
            "grep -rn SIZE nothere pkg 2>&1 | wc -l",
            "find pkg -type d",
            "find pkg -name '*.py' -exec grep -n SIZE {} \\;",
            "find pkg -name nomatch | xargs grep SIZE",
            "ls -a",
            "ls nothere | wc -l",
            "wc pkg/app.py pkg/util.py",
            "tail -n +2 README.md",
            "sed -n '2,$p' pkg/util.py",
            "head -1 pkg/app.py pkg/util.py",
            "find . -type f | wc -l",
            "cat .git/HEAD",
        ],
        [
            "cat noeol.txt README.md",
            "echo -n a; echo b",
            "cat noeol.txt && echo && cat README.md",
            "grep SIZE pkg",
            "find pkg -type d",
        ],
    ),
    "a chain reads a file and enters a directory its earlier stages made": (
        [
            "sed -i 's/SIZE = 1/SIZE = 2/' pkg/app.py",
            "git diff > patch.txt && cat patch.txt",
            "mkdir -p out/deep && cd out/deep && echo hi > f.txt",
        ],
        ["cat pkg/app.py", "ls out/deep", "cat out/deep/f.txt", "cat patch.txt"],
        ["cat pkg/app.py", "ls out/deep", "cat out/deep/f.txt", "cat patch.txt"],
    ),
    "a sed script sed rejects edits nothing and keeps no backup": (
        [
            "sed -i.bak 's/SIZE/N//' pkg/app.py; sed -i 's/import/x/&&/' pkg/util.py",
            "sed -i 's/SIZE/N/g/' pkg/app.py && echo yes > out.txt",
        ],
        ["ls pkg", "cat pkg/app.py", "cat pkg/util.py", "ls", "cat out.txt"],
        ["ls pkg", "cat pkg/app.py", "cat pkg/util.py", "ls", "cat out.txt"],
    ),
    "an escaped punctuation mark in a sed replacement stands for itself": (
        ["sed -i 's/SIZE = 1/SIZE\\.x = 2 \\[\\-\\]/' pkg/app.py"],
        ["cat pkg/app.py"],
        ["cat pkg/app.py"],
    ),
    "a mode change keeps the text, and git answers that would show it are not given": (
        ["chmod +x pkg/app.py"],
        ["cat pkg/app.py", "git status --short", "git diff"],
        ["cat pkg/app.py"],
    ),
    "python leaves a __pycache__ that a listing or walk would show": (
        ["python3 -c 'import pkg.app'"],
        [
            "ls pkg",
            "find pkg -type f",
            "find . -name '*.py'",
            "grep -rn SIZE pkg",
            "grep -rn SIZE --include=*.py pkg",
            "cat pkg/app.py",
        ],
        ["find . -name '*.py'", "grep -rn SIZE --include=*.py pkg", "cat pkg/app.py"],
    ),
    "an option's value is not a file the read could miss": (
        [
            "cat pkg/app.py | tail -n 1 || cp README.md pkg/app.py",
            "cat pkg/app.py | head -n 1 && echo yes > out.txt",
        ],
        ["cat pkg/app.py", "cat out.txt", "ls"],
        [],
    ),
    "rm -f of a directory fails, so what && and || guard follows": (
        [
            "rm -f pkg || touch fallback.txt",
            "rm -f pkg && echo yes > out.txt",
            "rm -f nothere.py && echo made > made.txt",
        ],
        ["ls", "cat fallback.txt", "cat out.txt", "cat made.txt", "cat pkg/app.py"],
        ["ls", "cat fallback.txt", "cat out.txt", "cat made.txt", "cat pkg/app.py"],
    ),
    "a status fixed from file names follows the directory a cd moved to": (
        [
            "cd pkg && cat app.py || echo x > n.txt",
            "cd pkg && rm -f util.py && echo yes > ok.txt",
            "cd pkg && mkdir -p sub/deep.py || echo new > notes.txt",
        ],
        ["cat pkg/n.txt", "cat pkg/ok.txt", "cat pkg/notes.txt", "cat n.txt", "ls pkg"],
        ["cat n.txt"],
    ),
    "a directory is never moved into itself": (
        ["mv pkg pkg/sub/moved", "mv nothere setup.py pkg/sub/moved"],
        ["ls pkg", "ls pkg/sub", "cat setup.py", "cat pkg/app.py", "git status --short"],
        ["ls pkg", "ls pkg/sub", "cat setup.py", "cat pkg/app.py"],
    ),
    "a file named with a trailing slash is not removed": (
        ["touch flat && rm -f flat/", "rm noeol.txt/"],
        ["ls", "cat flat", "cat noeol.txt"],
        ["ls", "cat flat", "cat noeol.txt"],
    ),
    "what lies under a directory that may be gone is not known either": (
        ["git diff --stat && mv pkg moved && touch pkg/sub/deep.py pkg"],
        ["grep -n SIZE pkg/sub", "cat pkg/sub/deep.py", "ls pkg", "ls moved"],
        [],
    ),
    "a rejected sed script writes nothing, not even its backup": (
        ["sed -i.bak 's/SIZE/\\1/' pkg/app.py", "sed -i 's/[0-9]///' pkg/util.py"],
        ["cat pkg/app.py", "cat pkg/util.py", "ls pkg", "cat pkg/app.py.bak"],
        ["cat pkg/app.py", "cat pkg/util.py", "ls pkg", "cat pkg/app.py.bak"],
    ),
}


def _bash(command: str, cwd: Path, tmp: Path | None = None) -> subprocess.CompletedProcess:
    """`command` run by bash in `cwd`; with `tmp`, that directory is the machine's `/tmp`."""
    argv = ["bash", "-c", command]
    if tmp is not None:
        argv = ["unshare", "-rm", "sh", "-c", 'mount --bind "$0" /tmp && exec "$@"', tmp, *argv]
    return subprocess.run(
        argv,
        cwd=cwd,
        env={**ENV, "HOME": str(cwd)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


@functools.cache
def _service() -> RepoContextService:
    return RepoContextService(RepoContextSettings(_env_file=None, cache_dir=tempfile.mkdtemp()))


def _write_files(*roots: Path) -> None:
    for root in roots:
        for path, text in FILES.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)


# sessions that work in /tmp, each run with a /tmp of its own that starts empty
TMP_CASES = {
    "a script written to /tmp is read back, searched and listed": (
        ["cat > /tmp/check.py <<'EOF'\nimport pkg.app\nprint(pkg.app.SIZE)\nEOF"],
        [
            "cat /tmp/check.py",
            "grep -n SIZE /tmp/check.py",
            "wc -l /tmp/check.py",
            "ls /tmp",
            "cat /tmp/other.py",
            "ls /tmp/nothere",
        ],
        [
            "cat /tmp/check.py",
            "grep -n SIZE /tmp/check.py",
            "wc -l /tmp/check.py",
            "ls /tmp",
            "cat /tmp/other.py",
            "ls /tmp/nothere",
        ],
    ),
    "a file in /tmp is edited, appended to and removed": (
        [
            "printf 'a=1\\nb=2\\n' > /tmp/conf",
            "sed -i 's/a=1/a=9/' /tmp/conf",
            "echo c=3 >> /tmp/conf",
            "echo gone > /tmp/old && rm /tmp/old",
        ],
        ["cat /tmp/conf", "ls /tmp", "cat /tmp/old"],
        ["cat /tmp/conf", "ls /tmp", "cat /tmp/old"],
    ),
    "files cross between the checkout and /tmp": (
        [
            "cp pkg/app.py /tmp/app.bak",
            "sed -i 's/SIZE = 1/SIZE = 2/' pkg/app.py",
            "mv pkg/util.py /tmp/",
            "cp /tmp/app.bak pkg/restored.py",
        ],
        [
            "cat /tmp/app.bak",
            "cat /tmp/util.py",
            "ls pkg",
            "cat pkg/restored.py",
            "ls /tmp",
            "git status --short",
        ],
        [
            "cat /tmp/app.bak",
            "cat /tmp/util.py",
            "ls pkg",
            "cat pkg/restored.py",
            "ls /tmp",
            "git status --short",
        ],
    ),
    "directories made in /tmp are walked": (
        ["mkdir -p /tmp/work/sub && touch /tmp/work/a.txt && echo x > /tmp/work/sub/b.txt"],
        ["ls /tmp/work", "find /tmp -type f", "cat /tmp/work/sub/b.txt", "ls /tmp/work/sub"],
        ["ls /tmp/work", "find /tmp -type f", "cat /tmp/work/sub/b.txt", "ls /tmp/work/sub"],
    ),
    "a program this does not follow may leave files in /tmp": (
        ["echo keep > /tmp/keep.txt", "mktemp"],
        ["cat /tmp/keep.txt", "ls /tmp", "find /tmp -type f"],
        ["cat /tmp/keep.txt"],
    ),
    "a command that is not followed leaves /tmp unknown": (
        [
            "echo keep > /tmp/keep.txt",
            "python3 -c \"import subprocess; subprocess.run(['rm', '/tmp/keep.txt'])\"",
        ],
        ["cat /tmp/keep.txt", "ls /tmp"],
        [],
    ),
}


@pytest.mark.parametrize("name", list(CASES) + list(TMP_CASES))
def test_every_exact_answer_is_the_real_one(name, tmp_path, request):
    commands, probes, answered = CASES.get(name) or TMP_CASES[name]
    tmp = None
    if name in TMP_CASES:
        # the checkout must not lie under the /tmp the commands get instead of the real one
        if not Path("/dev/shm").is_dir() or _bash("true", tmp_path, tmp_path).returncode:
            pytest.skip("needs a private /tmp (unshare) and /dev/shm")
        tmp_path = Path(tempfile.mkdtemp(dir="/dev/shm"))
        request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
        tmp = tmp_path / "tmp"
        tmp.mkdir()
    checkout, snapshot = tmp_path / "checkout", tmp_path / "snapshot"
    _write_files(checkout, snapshot)
    _bash("git init -q -b main && git add -A && git commit -qm head", checkout)
    sha = _bash("git rev-parse HEAD", checkout).stdout.decode().strip()
    messages = []
    for command in commands:
        done = _bash(command, checkout, tmp)
        messages += [
            {"role": "assistant", "content": f"```bash\n{command}\n```"},
            {
                "role": "user",
                "content": f"<returncode>{done.returncode}</returncode>\n<output>\n"
                f"{done.stdout.decode('utf-8', 'replace')}</output>",
            },
        ]
    service = _service()
    overlay = build_overlay(
        messages, sorted(FILES), lambda rel: service._read_snapshot_file(snapshot, rel)
    )
    for probe in probes:
        _, exact, returncode = service._build_repo_block(
            snapshot, overlay.listing(), probe, overlay, "returncode", GitMeta(sha=sha)
        )
        real = _bash(probe, checkout, tmp)
        if exact is None:
            assert probe not in answered, f"{probe}: declined"
            continue
        # the service's text leaves out the final line break the scaffold adds back
        want = real.stdout.decode("utf-8", "replace").removesuffix("\n")
        if probe.startswith(("find ", "grep -r")):
            # a walk visits entries in the filesystem's order, which no answer can know
            exact, want = sorted(exact.split("\n")), sorted(want.split("\n"))
        assert (exact, returncode) == (want, real.returncode), probe


# commands the service answers in part; the python3 stages are the gaps, some of them failing
SPLIT = [
    "cat pkg/app.py && python3 -c 'print(42)' && cat README.md",
    "sed -i 's/1/2/' pkg/app.py && python3 -c \"print(open('pkg/app.py').read())\"; "
    "grep -n SIZE pkg/app.py",
    "ls pkg; python3 -c 'print(1)'; cat noeol.txt",
    "echo start && python3 -c 'print(2)' || echo failed; cat setup.py",
    'python3 -c \'open("gen.txt", "w").write("x")\' && cat setup.py && python3 -c \'print(3)\'',
    # a failing gap: what `&&` guards is skipped, what `||` guards runs
    "cat setup.py && python3 -c 'import sys; print(5); sys.exit(3)' && cat README.md; "
    "cat noeol.txt",
    "cat setup.py && python3 -c 'import sys; sys.exit(1)' || cat README.md && echo after",
    "python3 -c 'import sys; sys.exit(2)' || echo recovered || cat README.md; echo done",
    "ls pkg && python3 -c 'import sys; sys.exit(4)' && cat README.md && cat setup.py",
]


@pytest.mark.parametrize("command", SPLIT)
def test_stitched_parts_with_the_real_gaps_are_the_real_output(command, tmp_path):
    """Each gap that runs answered with its real output, the parts run as the shell runs them,
    must give the command's real output and status: the parts split at the right stages, in
    order, with the right separators, and the stages after a gap answered whichever way it
    ends."""
    checkout, snapshot = tmp_path / "checkout", tmp_path / "snapshot"
    _write_files(checkout, snapshot)
    _bash("git init -q -b main && git add -A && git commit -qm head", checkout)
    service = _service()
    overlay = build_overlay(
        [], sorted(FILES), lambda rel: service._read_snapshot_file(snapshot, rel)
    )
    parts = service._parts(snapshot, command, overlay, "returncode", GitMeta(), "", set())
    assert parts is not None, command
    stages = parse_command(command).stages
    ran = []

    async def answer(gap):
        # the stages before the gap ran first, quietly: the gap then runs where they left it
        assert gap["context"]
        before = _gap_start(stages, gap["command"])
        copy = tmp_path / f"copy{len(ran)}"
        shutil.copytree(checkout, copy)
        prefix = f"{{ {before}\n}} >/dev/null 2>&1; " if before else ""
        done = _bash(prefix + gap["command"], copy)
        ran.append(gap["command"])
        return done.stdout.decode().removesuffix("\n"), done.returncode

    real = _bash(command, checkout)
    stitched = asyncio.run(stitch_parts(parts, answer))
    assert stitched == (real.stdout.decode().removesuffix("\n"), real.returncode)


def test_an_exit_that_may_run_after_a_gap_leaves_the_command_whole():
    """Whether the stages after `exit` run is not decided by statuses the parts carry."""
    service = _service()
    overlay = build_overlay([], sorted(FILES), lambda rel: FILES.get(rel))
    command = "python3 -c 'import sys; sys.exit(1)' || exit 4; cat setup.py"
    parts = service._parts(
        Path("/nonexistent"), command, overlay, "returncode", GitMeta(), "", set()
    )
    assert parts is None


def _gap_start(stages, gap_command: str) -> str:
    """The command text of the stages before a gap."""
    for index, stage in enumerate(stages):
        if gap_command.startswith(stage.text):
            return "".join(
                (f" {s.separator} " if s.separator in ("&&", "||") else "; ") * bool(i) + s.text
                for i, s in enumerate(stages[:index])
            )
    raise AssertionError(gap_command)


def test_names_are_laid_out_in_columns_as_gnu_ls_does(tmp_path):
    """On a terminal `ls` fills columns top to bottom, as many as fit, padded with tabs."""
    rng = random.Random(3)
    for trial in range(40):
        folder = tmp_path / str(trial)
        folder.mkdir()
        names = {
            "".join(rng.choice(string.ascii_lowercase + "_.0123456789") for _ in range(size))
            for size in rng.choices([1, 3, 6, 11, 24], k=rng.choice([2, 7, 19, 60]))
        }
        names = sorted({name.lstrip(".") or "x" for name in names})
        for name in names:
            (folder / name).touch()
        widths = (80, 23, 131, 400)
        ours = ls_layouts(names, widths)
        for width in widths:
            real = subprocess.run(
                ["ls", "-C"],
                cwd=folder,
                env={**ENV, "COLUMNS": str(width), "LC_ALL": "C"},
                capture_output=True,
                text=True,
            ).stdout.removesuffix("\n")
            assert "\n".join(ours[width]) == real, (width, names)


def test_status_names_how_the_branch_diverged_from_its_upstream(tmp_path):
    """A checkout whose one local commit sits beside an upstream of its own reports it after
    the branch line, as the recorded mini-coder-rs machines do. Newer git words the `git pull`
    hint differently; the text follows the git of the recorded machines."""
    _write_files(tmp_path)
    _bash(
        "git init -q -b main && git add -A && git commit -qm 'Initial commit' && "
        "git checkout -q --orphan upstream && git commit -qm 'Bug Patch' && "
        "git remote add origin https://example.invalid/demo.git && "
        "git update-ref refs/remotes/origin/main HEAD && git checkout -q main && "
        "git config branch.main.remote origin && git config branch.main.merge refs/heads/main",
        tmp_path,
    )
    edit = "sed -i 's/SIZE = 1/SIZE = 2/' pkg/app.py"
    _bash(edit, tmp_path)
    real = (
        _bash("git status", tmp_path)
        .stdout.decode()
        .replace(
            "if you want to integrate the remote branch with yours",
            "to merge the remote branch into yours",
        )
    )
    messages = [
        {"role": "assistant", "content": f"```bash\n{edit}\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
    ]
    overlay = build_overlay(messages, sorted(FILES), FILES.get)
    meta = GitMeta(sha="0" * 40, tracking=SCAFFOLDS[("mini-coder-rs", "returncode")].tracking)
    _, exact, _ = _service()._build_repo_block(
        tmp_path, overlay.listing(), "git status", overlay, "returncode", meta
    )
    assert exact == real.removesuffix("\n")
