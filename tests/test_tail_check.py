from __future__ import annotations

import asyncio

from albedo_eval_service.simulator.prompt_simulator import simulation_system_prompt
from sanity_service.tail_check import (
    DUP_CMD_THRESHOLD,
    MAX_RUN_THRESHOLD,
    loop_stats,
    looping_reason,
)


def _turns(commands: list[str]) -> list[str]:
    return [f"THOUGHT: next step.\n\n```bash\n{command}\n```" for command in commands]


# submission 0a9a8532 sample 0, verbatim order: 15 of its 28 commands are the submit command,
# six of them consecutively at the end, while the model lands four distinct `sed -i` edits.
_SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_ANSWER"
_SUBMIT_SPAM = [
    "nl -ba conan/cli/commands/config.py | sed -n '70,115p'",
    "sed -i -e 's/^from conan.cli.formatters import default_json_formatter$/from x import y/' c.py",
    "nl -ba conan/cli/commands/config.py | sed -n '1,12p;75,82p'",
    "cd /workspaces/conan && python -m conan.cli.cli config_list",
    "cat <<'PYEOF' > /tmp/test_config_list_filter.py import fnmatch PYEOF",
    _SUBMIT,
    "sed -i '80,82c\\ subparser.add_argument('pattern', nargs=\"?\")' config.py",
    "nl -ba conan/cli/commands/config.py | sed -n '76,84p'",
    _SUBMIT,
    "cd /workspaces/conan && python -m pytest test/functional/cli/test_cli_config_commands.py",
    _SUBMIT,
    _SUBMIT,
    "sed -i \"88s/subparser.add_argument('pattern')/x/\" config.py",
    _SUBMIT,
    _SUBMIT,
    'sed -i \'89s/.*/ """Show all builtin configurations"""/\' config.py',
    _SUBMIT,
    _SUBMIT,
    "nl -ba conan/cli/commands/config.py | sed -n '74,102p'",
    _SUBMIT,
    _SUBMIT,
    'grep -n -A 12 "def show" conan/api/subapi/config.py conan/internal/model/conf.py',
    *[_SUBMIT] * 6,
]

# submission d45cce5b sample 0, verbatim order: the prescribed workflow, nothing else
_VERIFY_CYCLE = [
    "git diff",
    "sed -n '145,155p' ably/realtime_conn.go",
    'grep -rn "recoverable" ably/ --include="*.go" | head -20',
    'go test ./ably/... -run "Conn" -v 2>&1 | head -50',
    'grep -n "func recoverable" ably/realtime_conn.go',
    "git status --short && git log --oneline -3",
    "git diff --name-only",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    "git diff",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    'grep -rn "recoverable" ably/ --include="*.go" | head -20',
    'go test ./ably/... -run "Conn" -v 2>&1 | head -50',
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "git diff",
    "sed -n '145,155p' ably/realtime_conn.go",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "git status --short && git log --oneline -3",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    'grep -n "recoverable" ably/realtime_conn.go',
    'go test ./ably/... -run "Conn" -v 2>&1 | head -50',
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "git diff",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
]


def test_our_own_submit_demands_do_not_count_as_the_model_looping():
    stats = loop_stats(_turns(_SUBMIT_SPAM))
    assert stats["n_cmds"] == len(_SUBMIT_SPAM) - _SUBMIT_SPAM.count(_SUBMIT)
    assert stats["dup_cmd_ratio"] == 0.0
    assert stats["max_cmd_run"] == 1
    assert not looping_reason(_turns(_SUBMIT_SPAM))


def test_the_prescribed_verify_cycle_is_not_a_loop():
    assert not looping_reason(_turns(_VERIFY_CYCLE))
    assert loop_stats(_turns(_VERIFY_CYCLE))["dup_cmd_ratio"] < DUP_CMD_THRESHOLD


def test_a_stuck_model_still_fails():
    stuck = ["sed -n '2273,2310p' pandas/io/stata.py"] * 21
    reason = looping_reason(_turns(stuck))
    assert "looping" in reason
    assert loop_stats(_turns(stuck))["max_cmd_run"] == 21


def test_the_consecutive_rule_is_untouched():
    varied = [f"cat file_{i}.py" for i in range(20)]
    assert not looping_reason(_turns(varied))
    assert looping_reason(_turns(varied + ["git diff"] * MAX_RUN_THRESHOLD))


def test_the_ratio_rule_still_catches_a_heavy_non_consecutive_loop():
    alternating = ["cat a.py", "cat b.py"] * 10
    stats = loop_stats(_turns(alternating))
    assert stats["max_cmd_run"] == 1
    assert stats["dup_cmd_ratio"] >= DUP_CMD_THRESHOLD
    assert "duplicate command ratio" in looping_reason(_turns(alternating))


def test_the_simulator_is_told_not_to_pre_apply_a_pending_request():
    prompt = simulation_system_prompt("openhands")
    assert "Never pre-apply a pending request" in prompt
    assert "BEFORE that" in prompt


def test_a_submit_the_harness_would_ignore_is_still_counted():
    from albedo_eval_service.shared.submit_protocol import asked_submit as _asked_submit

    assert _asked_submit("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    assert _asked_submit("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")
    assert _asked_submit("echo SUBMIT_TASK_33360A0E && git add -A && git diff --cached")
    assert not _asked_submit(
        "git add -A && git diff --cached && echo FINALIZE_AND_SUBMIT_TASK_OUTPUT"
    )
    assert not _asked_submit("git diff && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

    marker_last = ["git diff && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"] * MAX_RUN_THRESHOLD
    assert looping_reason(_turns(marker_last)), "a spammed unregisterable submit is still a loop"


def test_a_sample_that_only_ever_submits_is_left_to_the_submit_checks():
    from types import SimpleNamespace

    from sanity_service.chain import empty_submit_count

    marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    only_submits = [f"echo {marker}"] * 12
    assert loop_stats(_turns(only_submits)) == {
        "n_cmds": 0,
        "dup_cmd_ratio": 0.0,
        "max_cmd_run": 0,
    }
    assert not looping_reason(_turns(only_submits))

    state = SimpleNamespace(
        turns=[
            {"role": "assistant", "content": turn, "score_target": True}
            for turn in _turns(only_submits)
        ],
        submits=[],
        submit_marker=marker,
        submit_clause=f"echo {marker}",
    )
    assert empty_submit_count(state, marker) >= 2


def _state_with(turns: list[dict]):
    from types import SimpleNamespace

    return SimpleNamespace(
        sample_id="s", prompt="task", turns=turns, error="", heuristic_reason="", submit_clause=""
    )


def _scored(command: str) -> dict:
    return {"role": "assistant", "content": _turns([command])[0], "score_target": True}


def _obs(text: str) -> dict:
    return {"role": "user", "content": text, "environment_observation": True}


def _asked(text: str) -> dict:
    return {"role": "user", "content": text, "injected": True}


_EMPTY = "[The command finished with exit code 0.]\n[Command finished with exit code 0]"


def test_paired_turns_attach_the_result_and_the_requester_message():
    from sanity_service.tail_check import paired_turns

    turns = [
        {"role": "user", "content": "task"},
        _scored("cat a.py"),
        _obs("1\tx = 1"),
        _scored("echo DONE"),
        _asked("Looks good, also add a docstring."),
        _scored("sed -i '1i # doc' a.py"),
    ]
    assert paired_turns(turns) == [
        (_turns(["cat a.py"])[0], "1\tx = 1", None),
        (_turns(["echo DONE"])[0], None, "Looks good, also add a docstring."),
        (_turns(["sed -i '1i # doc' a.py"])[0], None, None),
    ]


def test_the_judge_is_shown_results_requester_messages_and_the_submit_command():
    from sanity_service.tail_check import TAIL_CUTOFF, _tail_user

    paired = [(f"turn {i}", None, None) for i in range(TAIL_CUTOFF)]
    paired.append(("THOUGHT: read\n```bash\ncat a.py\n```", _EMPTY, None))
    paired.append(("```bash\necho DONE && cat patch.txt\n```", None, "Please submit again now."))
    rendered = _tail_user("task", paired, submit_clause="echo DONE && cat patch.txt")
    assert f"LATE TURN {TAIL_CUTOFF + 1}:" in rendered
    assert f"RESULT {TAIL_CUTOFF + 1}" in rendered and _EMPTY in rendered
    assert f"REQUESTER {TAIL_CUTOFF + 2}" in rendered and "Please submit again now." in rendered
    assert "echo DONE && cat patch.txt" in rendered
    assert "turn 3" not in rendered, "turns before the cutoff stay hidden"


def test_a_memo_frozen_re_ask_no_longer_trips_the_mechanical_gate():
    from sanity_service.tail_check import run_tail_check

    read = "sed -n '278,295p' moto/eks/models.py"
    turns = [
        {"role": "user", "content": "task"},
        _scored("sed -i '1a x' moto/eks/models.py"),
        _obs(_EMPTY),
    ]
    for _ in range(13):
        turns += [_scored(read), _obs(_EMPTY)]
    state = _state_with(turns)
    verdicts = asyncio.run(run_tail_check([state]))
    assert not state.heuristic_reason, state.heuristic_reason
    assert verdicts and verdicts[0].reason.startswith("finished in") or not verdicts[0].checked


def test_a_real_loop_against_a_responsive_shell_still_trips_it():
    from sanity_service.tail_check import run_tail_check

    read = "cat -n modin/pandas/utils.py | sed -n '139,160p'"
    content = "   139\tdef cast_function_modin2pandas(func):\n[Command finished with exit code 0]"
    turns = [{"role": "user", "content": "task"}]
    for _ in range(15):
        turns += [_scored(read), _obs(content)]
    state = _state_with(turns)
    asyncio.run(run_tail_check([state]))
    assert "looping" in state.heuristic_reason
