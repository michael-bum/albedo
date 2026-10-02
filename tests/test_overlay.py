from __future__ import annotations

from albedo_eval_service.shared.observation_format import OPENHANDS_TRUNCATION_NOTICE
from repo_context_service.overlay import build_overlay

PATH = "pkg/big.py"
TRUE = "\n".join(f"line_{i:04d} = {i}  # real source content here" for i in range(900)) + "\n"
SMALL = "import os\nimport sys\n"


def _overlay(command: str, observation: str, base: str = TRUE):
    messages = [
        {"role": "assistant", "content": f"```bash\n{command}\n```"},
        {"role": "user", "content": observation},
    ]
    return build_overlay(messages, [PATH], lambda rel: base)


def _overlay_from(assistant_texts: list[str]):
    messages = [{"role": "assistant", "content": text} for text in assistant_texts]
    return build_overlay(messages, [PATH], lambda rel: TRUE)


def _search_replace(path: str, old: str, new: str) -> str:
    return f"Editing `{path}`:\n\n```\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n```"


def _openhands_clip(text: str) -> str:
    return f"{text[:15000]}\n{OPENHANDS_TRUNCATION_NOTICE}\n{text[-15000:]}"


def _returncode_clip(text: str) -> str:
    return (
        "<warning>\nThe output of your last command was too long.\n</warning><output_head>\n"
        f"{text[:5000]}\n</output_head>\n<elided_chars>\n{len(text) - 10000} characters elided\n"
        f"</elided_chars>\n<output_tail>\n{text[-5000:]}\n</output_tail>"
    )


def test_a_clipped_read_is_never_adopted():
    for observation in (
        _openhands_clip(TRUE),
        _returncode_clip(TRUE),
        f"{TRUE[:2000]}\n<response clipped>",
    ):
        assert _overlay("cat pkg/big.py", observation).read(PATH) is None


def test_a_file_the_agent_wrote_enters_the_listing_with_its_content():
    body = "#!/usr/bin/env python3\nprint('hi')"
    overlay = _overlay_from([f"```bash\ncat <<'EOF' > test_x.py\n{body}\nEOF\n```"])
    assert overlay.created == {"test_x.py"}
    assert overlay.read("test_x.py") == body + "\n"
    assert "test_x.py" in overlay.listing()


def test_a_full_rewrite_clears_dirt():
    overlay = _overlay_from(
        [
            f"```bash\nsed -i 's/a/b/' {PATH}\n```",
            f"```bash\ncat <<'EOF' > {PATH}\nfresh\nEOF\n```",
        ]
    )
    assert not overlay.is_dirty(PATH)
    assert overlay.read(PATH) == "fresh\n"


def test_a_heredoc_does_not_mask_a_later_edit_in_the_same_turn():
    overlay = _overlay_from(
        [
            f"```bash\ncat <<'EOF' > {PATH}\nSEEDED\nEOF\n```",
            f"```bash\ncat <<'EOF' > fresh.py\nX\nEOF\nsed -i 's/SEEDED/EDITED/' {PATH}\n```",
        ]
    )
    assert overlay.read("fresh.py") == "X\n"
    assert overlay.read(PATH) == "EDITED\n"


SOURCE = "def go():\n    return tuple(x for x in o)\nSIZE = 1\nTAIL = 2\n"


def _sed(command: str, base: str = SOURCE):
    messages = [{"role": "assistant", "content": f"```bash\n{command}\n```"}]
    return build_overlay(messages, [PATH], lambda rel: base)


def test_an_in_place_sed_is_applied_rather_than_forgotten():
    plain = _sed(f"sed -i 's/SIZE = 1/SIZE = 99/' {PATH}")
    assert not plain.is_dirty(PATH)
    assert plain.read(PATH) == SOURCE.replace("SIZE = 1", "SIZE = 99")

    addressed = _sed(f"cd /testbed && sed -i '2s|tuple(x for x in o)|[x for x in o]|' {PATH}")
    assert addressed.read(PATH) == SOURCE.replace("tuple(x for x in o)", "[x for x in o]")

    deleted = _sed(f"sed -i '3d' {PATH}")
    assert deleted.read(PATH) == "def go():\n    return tuple(x for x in o)\nTAIL = 2\n"


def test_an_in_place_sed_we_cannot_model_still_marks_the_file_dirty():
    for command in (
        f"sed -i '/SIZE/d' {PATH}",  # pattern address
        f"sed -i 's/SIZE = 1/&& extra/' {PATH}",  # backreference in the replacement
    ):
        overlay = _sed(command)
        assert overlay.is_dirty(PATH), command
        assert overlay.read(PATH) is None, command


def test_an_in_place_sed_is_applied_in_a_chain_a_pipe_and_before_a_missing_operand():
    edited = SOURCE.replace("SIZE = 1", "SIZE = 2")
    for command in (
        f"sed -i 's/SIZE = 1/SIZE = 2/' {PATH} && python -m pytest tests/",
        f"cd /testbed && sed -i -e 's/SIZE = 1/SIZE = 2/' {PATH} | head -1",
        # GNU sed edits each operand in turn, and only then fails on the missing one
        f"sed -i 's/SIZE = 1/SIZE = 2/' {PATH} other/mod.py",
    ):
        assert _sed(command).text(PATH) == edited, command
    # a form the model does not reproduce leaves the file present with unknown text
    unmodelled = _sed(f"sed -i '/SIZE/s/1/2/' {PATH}")
    assert unmodelled.is_dirty(PATH) and unmodelled.kind(PATH) == "file"
    # the script is not a file operand: `s/a/pkg/big.py/` names no file to forget
    assert not _sed(f"sed -n 's/a/{PATH}/p' notes.txt").is_dirty(PATH)


def test_several_expressions_in_one_sed_are_applied_in_order():
    both = SOURCE.replace("SIZE = 1", "SIZE = 2").replace("TAIL = 2", "TAIL = 3")
    for command in (
        f"sed -i -e 's/SIZE = 1/SIZE = 2/' -e 's/TAIL = 2/TAIL = 3/' {PATH}",
        f"sed -i 's/SIZE = 1/SIZE = 2/;s/TAIL = 2/TAIL = 3/' {PATH}",
    ):
        overlay = _sed(command)
        assert not overlay.is_dirty(PATH), command
        assert overlay.read(PATH) == both, command


def test_a_sed_that_matches_nothing_leaves_the_file_known_and_unchanged():
    overlay = _sed(f"sed -i 's/absent from the file/x/' {PATH}")
    assert not overlay.is_dirty(PATH)
    assert overlay.read(PATH) == SOURCE


def test_sed_can_append_insert_and_change_a_line():
    appended = _sed(f"sed -i '2a\\    return None' {PATH}")
    assert appended.read(PATH) == SOURCE.replace(
        "    return tuple(x for x in o)\n", "    return tuple(x for x in o)\n    return None\n"
    )

    inserted = _sed(f"sed -i '1i\\import os' {PATH}")
    assert inserted.read(PATH) == "import os\n" + SOURCE

    changed = _sed(f"sed -i '3c\\SIZE = 7' {PATH}")
    assert changed.read(PATH) == SOURCE.replace("SIZE = 1", "SIZE = 7")


def test_a_write_that_lands_outside_the_repo_keeps_the_file_it_reads_grounded():
    overlay = _sed(f"head -n 2 {PATH} > /tmp/part1.py")
    assert not overlay.is_dirty(PATH)
    assert overlay.opaque == []

    redirected = _sed(f"grep SIZE {PATH} > report.txt")
    assert not redirected.is_dirty(PATH)


def test_a_git_read_redirected_out_of_the_repo_keeps_its_source_grounded():
    outward = _sed(f"git diff {PATH} > /tmp/d.txt")
    assert not outward.is_dirty(PATH)
    assert outward.opaque == []

    # git reads the committed text, not the file the shell just truncated: it is restored
    inward = _sed(f"git show HEAD:{PATH} > {PATH}")
    assert inward.read(PATH) == SOURCE


def test_git_checkout_takes_back_a_sed_the_overlay_had_applied():
    done = "<returncode>0</returncode>\n<output>\n</output>"
    messages = [
        {"role": "assistant", "content": f"```bash\nsed -i 's/SIZE = 1/SIZE = 9/' {PATH}\n```"},
        {"role": "user", "content": done},
        {"role": "assistant", "content": f"```bash\ngit checkout -- {PATH}\n```"},
        {"role": "user", "content": done},
    ]
    overlay = build_overlay(messages, [PATH], lambda rel: SOURCE)
    assert overlay.read(PATH) == SOURCE
    assert not overlay.is_dirty(PATH)


def _git(*commands):
    done = "<returncode>0</returncode>\n<output>\n</output>"
    messages = []
    for command in commands:
        messages.append({"role": "assistant", "content": f"```bash\n{command}\n```"})
        messages.append({"role": "user", "content": done})
    return build_overlay(messages, [PATH], lambda rel: SOURCE)


def test_a_git_command_we_cannot_model_retires_every_key():
    clean = _git(f"cat {PATH}").state("BLOCK", {PATH})
    for command in ("git rm big.py", "git clean -fd", "git apply fix.patch", "git merge feature"):
        overlay = _git(command)
        assert overlay.git.unknown, command
        assert overlay.opaque == [(None, command)], command
        assert overlay.state("BLOCK", {PATH}) != clean, command


def test_an_unmodelled_git_command_is_recorded_once_not_on_every_later_step():
    overlay = _git("git rm big.py", f"cat {PATH}", "git status")
    assert overlay.opaque == [(None, "git rm big.py")]


def test_a_heredoc_behind_a_cd_lands_in_the_directory_the_chain_moved_to():
    overlay = _overlay_from(
        [
            "```bash\ncd /tmp && cat > repro.py <<'EOF'\nprint(1)\nEOF\n```",
            "```bash\ncd /testbed && cat > inside.py <<'EOF'\nprint(2)\nEOF\n```",
        ]
    )
    assert overlay.read("/tmp/repro.py") == "print(1)\n"
    assert overlay.read("repro.py") is None
    assert overlay.read("inside.py") == "print(2)\n"


def test_a_heredoc_behind_a_relative_cd_is_left_untracked():
    overlay = _overlay_from(["```bash\ncd subdir && cat > x.py <<'EOF'\nprint(1)\nEOF\n```"])
    assert overlay.read("x.py") is None
    assert overlay.read("subdir/x.py") is None


def test_a_heredoc_written_to_a_dot_slash_path_is_the_file_a_later_read_resolves():
    overlay = _overlay_from([f"```bash\ncat <<'EOF' > ./{PATH}\nNEW\nEOF\n```"])
    assert overlay.read(PATH) == "NEW\n"
    assert PATH not in overlay.created


def test_a_copy_from_outside_leaves_its_target_unknown_and_a_move_carries_the_text():
    assert _overlay_from([f"```bash\ncp /tmp/fixed.py {PATH}\n```"]).is_dirty(PATH)
    moved = _overlay_from([f"```bash\nmv {PATH} pkg/moved.py\n```"])
    assert moved.kind(PATH) is None
    assert moved.text("pkg/moved.py") == TRUE
    backup = _overlay_from([f"```bash\ncp {PATH} {PATH}.bak\n```"])
    assert backup.text(PATH) == backup.text(f"{PATH}.bak") == TRUE


def test_a_search_replace_edit_is_applied_to_the_file():
    old, new = "line_0003 = 3  # real source content here", "line_0003 = 33"
    overlay = _overlay_from([_search_replace(PATH, old, new)])
    assert overlay.read(PATH) == TRUE.replace(old, new)


def test_an_editor_view_is_adopted_without_its_closing_line_number():
    view = (
        f"Here's the result of running `cat -n` on /workspace/repo/{PATH}:\n"
        "     1\timport os\n     2\timport sys\n     3"
    )
    assert _overlay(f"cat -n {PATH}", view).text(PATH) == SMALL


def _returncode(code: int, output: str = "") -> str:
    return f"<returncode>{code}</returncode>\n<output>\n{output}</output>"


def test_a_simulated_read_after_an_unfollowed_edit_is_never_adopted():
    """Adopting what a simulated `cat` showed would make one wrong answer the file's text for
    the rest of the session; only reads recorded on a real machine settle an unknown file."""
    unfollowed = "perl -pi -e 's/line_/row_/' pkg/big.py"
    messages = [
        {"role": "assistant", "content": f"```bash\n{unfollowed}\n```"},
        {"role": "user", "content": _returncode(0)},
        {"role": "assistant", "content": "```bash\ncat pkg/big.py\n```"},
        {"role": "user", "content": _returncode(0, SMALL)},
    ]
    assert build_overlay(messages, [PATH], lambda rel: TRUE, recorded=2).text(PATH) == SMALL
    assert build_overlay(messages, [PATH], lambda rel: TRUE, recorded=1).text(PATH) is None


def test_shell_code_is_not_replayed_and_leaves_what_it_may_change_unknown():
    """Shell code run by `bash -c` or a script the session wrote is not replayed: a script that
    calls itself must not recurse, and one that edits a file must not leave its old text."""
    looping = _overlay_from(
        ["```bash\nprintf 'bash loop.sh\\n' > loop.sh\n```", "```bash\nbash loop.sh\n```"]
    )
    assert looping.text(PATH) == TRUE
    edited = _overlay(f"bash -c \"sed -i 's/line_/row_/' {PATH}\"", _returncode(0))
    assert edited.text(PATH) is None


def test_a_command_of_only_a_comment_is_replayed_without_failing():
    """A turn whose command is only a comment runs nothing; failing on it would leave every
    later turn of the session without grounding."""
    overlay = _overlay(
        "# look at the tests first", "<returncode>0</returncode>\n<output>\n</output>"
    )
    assert overlay.text(PATH) == TRUE


def test_a_read_whose_first_line_is_indented_is_learned_with_its_indentation():
    """Adopting a read without its first line's indentation would make every later exact
    answer about the file wrong, in every scaffold's format."""
    indented = "    return helper(x)\nprint(1)\n"
    blank_first = "\n    x = 1\n"
    messages = [
        {"role": "assistant", "content": f"```bash\nperl -pi -e 's/a/b/' {PATH}\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
        {"role": "assistant", "content": f"```bash\ncat {PATH}\n```"},
        {
            "role": "user",
            "content": f"<returncode>0</returncode>\n<output>\n{blank_first}</output>",
        },
    ]
    assert build_overlay(messages, [PATH], lambda rel: TRUE).text(PATH) == blank_first
    for observation in (
        f"<returncode>0</returncode>\n<output>\n{indented}</output>",
        f"{indented}\n[The command completed with exit code 0.]\n"
        "[Command finished with exit code 0]",
    ):
        messages = [
            {"role": "assistant", "content": f"```bash\nperl -pi -e 's/a/b/' {PATH}\n```"},
            {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
            {"role": "assistant", "content": f"```bash\ncat {PATH}\n```"},
            {"role": "user", "content": observation},
        ]
        assert build_overlay(messages, [PATH], lambda rel: TRUE).text(PATH) == indented


def test_a_file_the_editor_created_holds_the_text_without_the_heredoc_line_break():
    """A heredoc the openhands editor reports creating wrote the text exactly as given: a view
    of the file numbers no empty line after its last one."""
    messages = [
        {"role": "assistant", "content": "```bash\ncat > new.py <<'EOF'\nx = 1\nEOF\n```"},
        {"role": "user", "content": "File created successfully at: new.py"},
    ]
    assert build_overlay(messages, [PATH], lambda rel: TRUE).text("new.py") == "x = 1"
    plain = [
        messages[0],
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
    ]
    assert build_overlay(plain, [PATH], lambda rel: TRUE).text("new.py") == "x = 1\n"
