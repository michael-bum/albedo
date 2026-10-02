from __future__ import annotations

import functools
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from albedo_config import RepoContextSettings
from repo_context_service.command_search import ParseFailure
from repo_context_service.core import RepoContextService
from repo_context_service.git_sim import (
    GitMeta,
    apply_hunks,
    blob_hash,
    is_git_command,
    ledger_block,
)
from repo_context_service.git_sim.patches import observed_patches
from repo_context_service.overlay import build_overlay

BASE = {
    "src/app.py": "def main():\n    return 1\n",
    "README.md": "# demo\n",
}
LISTING = sorted(BASE)
META = GitMeta(sha="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0", owner="acme", repo="demo")


def _read_base(rel: str) -> str | None:
    return BASE.get(rel)


def _turn(command: str, body: str = "", returncode: int = 0) -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": f"```bash\n{command}\n```"},
        {
            "role": "user",
            "content": f"<returncode>{returncode}</returncode>\n<output>\n{body}\n</output>",
        },
    ]


@functools.cache
def _service() -> RepoContextService:
    return RepoContextService(RepoContextSettings(_env_file=None, cache_dir=tempfile.mkdtemp()))


def _run(command: str, messages: list[dict[str, str]] | None = None, meta: GitMeta = META):
    """The command's exact answer through the service's stage runner, or a ParseFailure when it
    is left to the simulator."""
    overlay = build_overlay(messages or [], LISTING, _read_base)
    run = _service()._run_chain
    computed, _, _ = run(Path("/nonexistent"), command, overlay, "returncode", meta)
    if computed is None:
        return ParseFailure("declined"), overlay
    _, exact, returncode = computed
    return SimpleNamespace(output=exact, returncode=returncode, empty=exact == ""), overlay


EDIT = _turn("cat > src/app.py <<'EOF'\ndef main():\n    return 2\nEOF")
CREATE = _turn("cat > notes.txt <<'EOF'\nhello\nEOF")


def test_git_add_is_silent_with_returncode_zero():
    result, _ = _run("cd /testbed && git add -A", EDIT)
    assert result.output == ""
    assert result.returncode == 0
    assert result.empty


def test_status_reports_the_default_branch_and_the_unstaged_edit():
    result, _ = _run("git status", EDIT + CREATE)
    assert result.output.split("\n") == [
        "On branch main",
        "Changes not staged for commit:",
        '  (use "git add <file>..." to update what will be committed)',
        '  (use "git restore <file>..." to discard changes in working directory)',
        "\tmodified:   src/app.py",
        "",
        "Untracked files:",
        '  (use "git add <file>..." to include in what will be committed)',
        "\tnotes.txt",
        "",
        'no changes added to commit (use "git add" and/or "git commit -a")',
    ]


def test_status_on_a_rebench_image_reports_a_detached_head():
    result, _ = _run("git status", EDIT, meta=GitMeta(sha=META.sha, detached=True))
    assert result.output.startswith("Not currently on any branch.\n")


def test_a_staged_tree_drops_the_summary_line_and_keeps_the_trailing_blank():
    result, _ = _run("git status", EDIT + _turn("git add -A"))
    assert result.output.endswith("\tmodified:   src/app.py\n")


def test_diff_renders_real_blob_hashes_and_git_hunk_headers():
    result, _ = _run("git diff", EDIT)
    old = blob_hash(BASE["src/app.py"])[:7]
    new = blob_hash("def main():\n    return 2\n")[:7]
    assert result.output.split("\n") == [
        "diff --git a/src/app.py b/src/app.py",
        f"index {old}..{new} 100644",
        "--- a/src/app.py",
        "+++ b/src/app.py",
        "@@ -1,2 +1,2 @@",
        " def main():",
        "-    return 1",
        "+    return 2",
    ]


def test_staging_moves_the_change_from_diff_to_diff_cached():
    staged = EDIT + _turn("git add -A")
    plain, _ = _run("git diff", staged)
    cached, _ = _run("git diff --cached", staged)
    assert plain.empty
    assert "-    return 1" in cached.output


def test_a_chain_executes_stage_by_stage_with_the_staging_applied():
    result, _ = _run("cd /testbed && git add -A && git diff --cached", EDIT)
    assert result.output.startswith("diff --git a/src/app.py b/src/app.py")
    assert result.returncode == 0


def test_checkout_of_a_pathspec_reports_the_updated_path_count():
    result, _ = _run("git checkout src/app.py", EDIT)
    assert result.output == "Updated 1 path from the index"


def test_a_double_dash_checkout_is_completely_silent():
    result, _ = _run("git checkout -- src/app.py", EDIT)
    assert result.output == ""
    assert result.returncode == 0


def test_stash_leaves_a_clean_tree_and_the_ledger_records_it():
    messages = EDIT + _turn("git add -A") + _turn("git stash", "Saved working directory")
    result, overlay = _run("git status", messages)
    assert result.output == "On branch main\nnothing to commit, working tree clean"
    assert "git stash" in ledger_block(overlay.git)
    assert "git add -A" in ledger_block(overlay.git)


def test_popping_an_empty_stash_fails_the_way_git_does():
    result, _ = _run("git stash pop", EDIT)
    assert result.output == "No stash entries found."
    assert result.returncode == 1


def test_the_abbreviation_length_is_learned_from_an_earlier_observation():
    seen = _turn("git diff", "index 1234567890..abcdef1234 100644")
    result, _ = _run("git diff", seen + EDIT)
    old = blob_hash(BASE["src/app.py"])[:10]
    new = blob_hash("def main():\n    return 2\n")[:10]
    assert result.output.split("\n")[1] == f"index {old}..{new} 100644"


def test_an_unmodelled_git_command_poisons_the_state_instead_of_guessing():
    result, _ = _run("git status", _turn("git commit -m wip", "[main abc1234] wip"))
    assert isinstance(result, ParseFailure)


def test_a_chain_stage_we_cannot_execute_refuses_the_whole_command():
    result, _ = _run("python reproduce.py && git add -A && git diff --cached", EDIT)
    assert isinstance(result, ParseFailure)


def test_a_search_stage_after_a_git_stage_sees_the_git_change():
    result, _ = _run("git checkout -- src/app.py && grep -rn 'return 1' src", EDIT)
    assert result.output == "src/app.py:2:    return 1"
    assert result.returncode == 0


def test_the_ledger_is_rendered_for_git_commands_only():
    assert is_git_command("git status")
    assert is_git_command("cd /testbed && git add app.py")
    assert not is_git_command("cat README.md")
    assert not is_git_command("grep -rn digit .")


APP_DIFF = (
    "diff --git a/src/app.py b/src/app.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def main():\n"
    "-    return 1\n"
    "+    return 2\n"
)
APP_HUNKS = [(1, [" def main():", "-    return 1", "+    return 2"])]
OPENHANDS_TRAILER = (
    "[The command completed with exit code 0.]\n"
    "[Current working directory: /workspace/demo]\n"
    "[Command finished with exit code 0]"
)


@pytest.mark.parametrize(
    "observation",
    [
        APP_DIFF,  # bare output, git's own trailing newline
        APP_DIFF + "\n",  # a blank line after the diff
        APP_DIFF + OPENHANDS_TRAILER,  # OpenHands bash output closes with the exit-code trailer
        APP_DIFF + "\n" + OPENHANDS_TRAILER,
        f"<returncode>0</returncode>\n<output>\n{APP_DIFF}\n</output>",
        f"OBSERVATION:\n{APP_DIFF}",
    ],
)
def test_observed_diff_is_kept_whatever_follows_the_last_hunk(observation):
    patches = observed_patches(observation)
    assert patches == {"src/app.py": APP_HUNKS}
    assert apply_hunks(BASE["src/app.py"], patches["src/app.py"]) == "def main():\n    return 2\n"


def test_observed_diff_keeps_a_blank_context_line_inside_a_hunk():
    diff = (
        "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1,3 +1,3 @@\n # demo\n\n-old\n+new\n" + OPENHANDS_TRAILER
    )
    assert observed_patches(diff) == {"README.md": [(1, [" # demo", " ", "-old", "+new"])]}


def test_observed_diff_keeps_every_file_when_a_trailer_follows():
    readme = (
        "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        "@@ -1 +1 @@\n-# demo\n+# Demo\n"
    )
    patches = observed_patches(APP_DIFF + readme + OPENHANDS_TRAILER)
    assert patches == {"src/app.py": APP_HUNKS, "README.md": [(1, ["-# demo", "+# Demo"])]}


def test_transcript_git_diff_teaches_the_overlay_the_edit():
    """A `git diff` the real harness ran (OpenHands trailer included) is how the overlay learns
    an edit it did not see being made; later reads must show the edited file."""
    messages = [
        {"role": "assistant", "content": "```bash\ngit diff\n```"},
        {"role": "user", "content": APP_DIFF + OPENHANDS_TRAILER},
    ]
    overlay = build_overlay(messages, LISTING, _read_base)
    assert overlay.read("src/app.py") == "def main():\n    return 2\n"


HISTORY = {
    "commits": [
        {"sha": META.sha, "subject": "Fix the parser"},
        {"sha": "b" * 40, "subject": "Start"},
    ],
    "complete": True,
}


def test_a_log_names_the_detached_head_only_when_it_writes_to_a_terminal():
    """Git decorates refs on a terminal: `(HEAD)` beside the commit a detached HEAD is at; a
    piped log is plain. Where the scaffold's terminal, or the refs of a branch, are not known,
    the log is left to the simulator rather than printed without them."""
    meta = replace(META, detached=True, decorate=True, history=lambda path: HISTORY)
    assert (
        _run("git log --oneline", meta=meta)[0].output
        == "a1b2c3d (HEAD) Fix the parser\nbbbbbbb Start"
    )
    assert _run("git log --oneline | head -1", meta=meta)[0].output == "a1b2c3d Fix the parser"
    for unknown in (replace(meta, decorate=None), replace(meta, detached=False)):
        assert isinstance(_run("git log --oneline", meta=unknown)[0], ParseFailure)


def test_a_short_hash_is_never_printed_before_its_length_is_known():
    """A short hash grows with the repository; until an observation shows one, a oneline log
    or a diff's `index` line is not answered, and a status (which has none) still is."""
    meta = replace(META, abbrev=None, detached=True, history=lambda path: HISTORY)
    edited = _turn("sed -i 's/return 1/return 2/' src/app.py")
    for command in ("git log --oneline", "git diff"):
        assert isinstance(_run(command, edited, meta)[0], ParseFailure), command
    assert "modified:   src/app.py" in _run("git status", edited, meta)[0].output
    shown = edited + _turn("git log --oneline -1", "a1b2c3d4 Fix the parser")
    index_line = _run("git diff", shown, meta)[0].output.split("\n")[1]
    assert index_line.startswith("index ")
    assert len(index_line.split()[1].split("..")[0]) == 8


def test_a_squashed_history_is_the_one_commit_an_observation_named():
    """A history squashed into one local commit is not the one the upstream API knows: a log
    is answered only once an observation has shown that commit."""
    meta = replace(META, squashed=True, history=lambda path: HISTORY)
    assert isinstance(_run("git log --oneline", meta=meta)[0], ParseFailure)
    seen = _turn("git log --oneline", "f15fb67 Initial commit")
    assert _run("git log --oneline -5", seen, meta)[0].output == "f15fb67 Initial commit"


def test_git_branch_at_a_detached_head_is_no_branch_and_nothing_else_is_guessed():
    """At a detached HEAD (the recorded openhands checkouts) git lists `* (no branch)`; a
    checkout on a branch may hold others, and one the session made is not the only one."""
    detached = replace(META, detached=True)
    assert _run("git branch", meta=detached)[0].output == "* (no branch)"
    assert _run("git branch -a", meta=detached)[0].output == "* (no branch)"
    assert isinstance(_run("git branch", meta=META)[0], ParseFailure)
    made = _turn("git checkout -b fix")
    assert isinstance(_run("git branch", made, detached)[0], ParseFailure)


def test_a_log_whose_hash_length_is_unknown_is_still_shown_to_the_simulator():
    """Declining a oneline log whose short-hash length is not known must not leave the simulator
    without the history: it would invent commits the repository does not have."""
    from repo_context_service.git_sim import git_evidence

    meta = replace(META, abbrev=None, detached=True, decorate=False, history=lambda path: HISTORY)
    overlay = build_overlay([], LISTING, _read_base)
    evidence = git_evidence("git log --oneline -2", overlay, _read_base, LISTING, meta)
    assert "a1b2c3d Fix the parser" in evidence and "bbbbbbb Start" in evidence
    assert "git prints more in a larger repository" in evidence


def test_a_stash_under_an_unknown_index_still_restores_tracked_files_from_head():
    """`cd src && git add .` may or may not have staged the edit, so the index is unknown; the
    stash still resets every tracked file to HEAD, while a `checkout --` from that index cannot
    say what text it restores."""
    poisoned = EDIT + _turn("cd src && git add .")
    result, overlay = _run("cat src/app.py", poisoned + _turn("git stash"))
    assert overlay.git.unknown
    assert result.output == BASE["src/app.py"].removesuffix("\n")
    result, _ = _run("cat src/app.py", poisoned + _turn("git reset --hard"))
    assert result.output == BASE["src/app.py"].removesuffix("\n")
    result, _ = _run("cat src/app.py", poisoned + _turn("git checkout -- src/app.py"))
    assert isinstance(result, ParseFailure)


def test_a_failed_commit_does_not_keep_the_edit_through_a_hard_reset():
    messages = EDIT + _turn("git commit -qm work", "no changes added to commit", 1)
    result, _ = _run("cat src/app.py", messages + _turn("git reset --hard"))
    assert result.output == BASE["src/app.py"].removesuffix("\n")
    # after a commit that went through, HEAD moved and the restored text is not the snapshot's
    result, _ = _run(
        "cat src/app.py", EDIT + _turn("git commit -qam work") + _turn("git reset --hard")
    )
    assert isinstance(result, ParseFailure)


def test_git_mv_into_a_missing_directory_moves_nothing():
    messages = _turn("mkdir out\ngit mv README.md docs/b.cfg\nrm -rf out")
    result, overlay = _run("cat README.md", messages)
    assert result.output == "# demo"
    assert not overlay.git.staged_deleted and "docs/b.cfg" not in overlay.git.index
    result, _ = _run("cat docs/b.cfg", messages)
    assert result.returncode == 1


def test_git_rm_removes_the_directory_its_last_file_left():
    result, _ = _run("ls", _turn("git rm -q src/app.py"))
    assert result.output == "README.md"
    result, _ = _run("ls src", _turn("git rm -q src/app.py"))
    assert result.returncode == 2


def test_a_reset_after_git_rm_unstages_the_deletion():
    result, _ = _run("git status --short", _turn("git rm -q README.md") + _turn("git reset -q"))
    assert result.output == " D README.md"
    result, _ = _run(
        "git status --short", _turn("git rm -q README.md") + _turn("git restore --staged README.md")
    )
    assert result.output == " D README.md"
    result, _ = _run(
        "git status --short", _turn("git rm -q README.md") + _turn("git reset -q --hard")
    )
    assert result.output == ""


def test_a_renamed_file_edited_afterwards_is_one_rm_row_in_short_status():
    messages = _turn("git mv README.md notes.md") + _turn("echo more >> notes.md")
    result, _ = _run("git status --short", messages)
    assert result.output == "RM README.md -> notes.md"
