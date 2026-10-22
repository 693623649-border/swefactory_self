"""Regression tests for the deterministic pool B/C construction layer.

These tests pin the behaviour that an independent audit found broken once:
runner attribution inside shell chains, go-summary false positives, environment
failures masquerading as test failures, and supervision that must never leak
outside the intended assistant messages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "results"
sys.path.insert(0, str(RESULTS))

import bc_export as bx  # noqa: E402
import segment_review as sr  # noqa: E402


# --- runner parsing --------------------------------------------------------

def test_pytest_summary_pass_and_fail():
    result = sr.parse_test_result("cd /w && python -m pytest -q",
                                  "...\n215 passed, 3 deselected, 4 warnings in 0.31s\n")
    assert (result.state, result.reason, result.test_ran) == ("pass", "runner_summary", True)
    assert result.passed == 215 and result.deselected == 3 and result.executed == 215

    failing = sr.parse_test_result("pytest -q",
                                   "FAILED tests/test_a.py::test_x\n5 failed, 8 passed in 0.11s\n")
    assert failing.state == "fail" and failing.executed == 13


def test_piped_run_never_falls_back_to_exit_code():
    result = sr.parse_test_result("pytest -q | tail -5", "1 failed\n[exit code: 1]\n")
    assert result.state == "unknown"
    assert result.test_ran is False


def test_summary_followed_by_large_output_is_still_found():
    # a trailing `git diff` used to push the summary out of the old 12k window
    observation = "113 passed in 0.19s\n" + ("+ some diff line\n" * 3000)
    result = sr.parse_test_result("python -m pytest -q && git diff", observation)
    assert result.state == "pass" and result.passed == 113


def test_conflicting_runs_with_piped_tail_is_unknown():
    command = "python -m pytest -q tests/a.py | tail -5; python -m pytest -q tests/b.py | tail -5"
    observation = "1 failed, 32 passed in 0.10s\n32 passed, 1 deselected in 0.11s\n"
    result = sr.parse_test_result(command, observation)
    assert result.state == "unknown" and result.reason == "conflicting_runs"


def test_conflicting_runs_with_visible_last_segment_uses_last_summary():
    command = "python -m pytest -q tests/a.py && python -m pytest -q tests"
    observation = "9 passed in 0.06s\n1 failed, 246 passed in 3.73s\n"
    result = sr.parse_test_result(command, observation)
    assert result.state == "fail" and result.failed == 1


def test_go_summary_requires_go_test_and_duration():
    assert sr.parse_test_result("go test ./...", "ok  example.com/pkg  0.512s\n").state == "pass"
    # ordinary program output that merely starts with "ok"
    assert sr.parse_test_result("python -c 'print(1)'", "ok {'type': 'websocket.send'}\n"
                               ).state == "unknown"
    assert sr.parse_test_result("go test ./...", "ok <repr object at 0x1>\n").state == "unknown"


def test_unittest_summary():
    assert sr.parse_test_result("python -m unittest discover",
                                "Ran 12 tests in 0.31s\n\nOK\n").state == "pass"
    failing = sr.parse_test_result("python -m unittest discover",
                                   "Ran 12 tests in 0.31s\n\nFAILED (failures=2)\n")
    assert failing.state == "fail" and failing.failed == 2


def test_environment_failure_is_not_a_test_failure():
    result = sr.parse_test_result("python -m pytest -q",
                                  "/usr/bin/python: No module named pytest\n[exit code: 1]\n")
    assert result.state == "unknown" and result.reason == "infra" and not result.test_ran


def test_linter_error_counts_are_not_test_results():
    mypy = sr.parse_test_result("python -m pytest -q && mypy pkg",
                                "Found 3 errors in 1 file (checked 12 source files)\n")
    assert mypy.state == "unknown" and not mypy.test_ran
    bare = sr.parse_test_result("python -m pytest -q && ruff check pkg", "Found 5 errors.\n")
    assert bare.state == "unknown" and not bare.test_ran


def test_trailing_helper_script_does_not_flip_a_passing_run():
    command = ("cd /w && python -m pytest -q | tail -15 && python - <<'PY'\n"
               "assert 1 == 2\nPY")
    observation = "215 passed in 0.31s\nTraceback (most recent call last):\nAssertionError\n" \
                  "[exit code: 1]\n"
    result = sr.parse_test_result(command, observation)
    # the tests passed, the helper script failed: the chain's failure must not be read
    # as a test result and must not be read as a clean verification either
    assert result.state == "unknown" and result.reason == "trailing_command_failed"


def test_helper_script_crash_without_summary_is_not_a_test_failure():
    command = "cd /w && python - <<'PY'\nraise SystemExit(1)\nPY"
    result = sr.parse_test_result(command, "boom\n[exit code: 1]\n")
    assert result.state == "unknown" and not result.test_ran


def test_exit_code_failure_is_weak_evidence():
    result = sr.parse_test_result("pytest -q", "collection error noise\n[exit code: 2]\n")
    assert result.state == "fail" and result.reason == "exit_code" and not result.test_ran


def test_read_only_command_is_not_a_test_run():
    result = sr.parse_test_result("cat tests/test_a.py", "def test_a():\n    assert 1\n")
    assert result.state == "unknown" and not result.test_ran


# --- patch accounting ------------------------------------------------------

HARNESS_PATCH = """diff --git a/.gitignore b/.gitignore
--- a/.gitignore
+++ b/.gitignore
@@ -1,2 +1,4 @@
+# === AWEAGENT AUTO-GENERATED START ===
+*.png
diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1 +1,2 @@
-old line
+new line
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-x
+y
"""


def test_patch_info_separates_source_test_and_harness_noise():
    info = sr.patch_info(HARNESS_PATCH)
    assert info.content_lines == 2          # only pkg/core.py's +/- lines
    assert info.harness_only is True
    assert info.source_files == ["pkg/core.py"]
    assert info.test_files == ["tests/test_core.py"]


# --- honesty screen --------------------------------------------------------

def test_honesty_screen():
    assert sr.honesty_screen("I could not fix this; the remaining failure is in "
                             "`parse()` and two tests still fail.")["passes"] is True
    assert sr.honesty_screen("All tests pass and the issue is fixed.")["passes"] is False
    blended = sr.honesty_screen("I fixed the first case but could not fix the second.")
    assert blended["passes"] is False and blended["review_required"] is True


# --- pool rules ------------------------------------------------------------

def _messages(fail_obs: str, pass_obs: str, *, edit_between: bool = True,
              report: str = "Implemented the fix and verified it.") -> list[dict]:
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the parser bug in pkg/core.py."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash",
                          "arguments": '{"command": "cd /w && cat pkg/core.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "bash",
         "content": "def parse():\n    return 1\n"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "bash",
                          "arguments": '{"command": "cd /w && python -m pytest -q"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "bash", "content": fail_obs},
    ]
    if edit_between:
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c3", "type": "function",
                 "function": {"name": "bash",
                              "arguments": '{"command": "cd /w && python - <<PY\\n'
                                           'from pathlib import Path\\n'
                                           'p=Path(\\"pkg/core.py\\")\\nPY"}'}}]},
            {"role": "tool", "tool_call_id": "c3", "name": "bash", "content": ""},
        ]
    messages += [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c4", "type": "function",
             "function": {"name": "bash",
                          "arguments": '{"command": "cd /w && python -m pytest -q"}'}}]},
        {"role": "tool", "tool_call_id": "c4", "name": "bash", "content": pass_obs},
        {"role": "assistant", "content": report},
    ]
    return messages


ADMITTED = {"admitted": True, "family": "owner/repo", "f2p": {"passed": 2},
            "p2p": {"passed": 20}, "node_evidence": "evaluator_acceptance", "rejected": []}


def _record(messages, patch="diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n"
                           "+++ b/pkg/core.py\n@@ -1 +1 @@\n-old\n+new\n"):
    return {"instance_id": "owner_repo_pr1", "success": True, "messages": messages,
            "patch": patch, "finish_reason": "finish", "stats": {"steps": 4}}


def test_b_accepts_real_failure_then_equivalent_pass():
    record = _record(_messages("1 failed, 20 passed in 0.2s\n", "21 passed in 0.2s\n"))
    diag = sr.diagnose(record)
    row = bx.evaluate_b(record, ADMITTED, diag)
    assert row["decision"] == "ADMIT"
    # supervision starts after the failing observation and never covers tool output
    assert min(row["keep_msgs"]) > 5
    assert any(message.get("role") != "assistant" for message in record["messages"])
    assert all(record["messages"][msg]["role"] == "assistant" for msg in row["keep_msgs"])


def test_b_rejects_pass_that_deselects_the_failure():
    record = _record(_messages("1 failed, 20 passed in 0.2s\n",
                               "1 passed, 20 deselected in 0.2s\n"))
    row = bx.evaluate_b(record, ADMITTED, sr.diagnose(record))
    assert row["decision"] == "REVIEW" and "VERIFICATION_NARROWED" in row["reasons"]


def test_b_requires_an_edit_between_failure_and_pass():
    record = _record(_messages("1 failed, 20 passed in 0.2s\n", "21 passed in 0.2s\n",
                               edit_between=False))
    row = bx.evaluate_b(record, ADMITTED, sr.diagnose(record))
    assert row is None or row["decision"] == "REVIEW"


def test_c2_flags_a_redundant_rerun_of_the_same_check():
    messages = _messages("1 failed, 20 passed in 0.2s\n", "21 passed in 0.2s\n")
    messages += [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c5", "type": "function",
             "function": {"name": "bash",
                          "arguments": '{"command": "python -m pytest -q"}'}}]},
        {"role": "tool", "tool_call_id": "c5", "name": "bash",
         "content": "21 passed in 0.2s\n"},
        {"role": "assistant", "content": "Done."},
    ]
    row = bx.evaluate_c2(_record(messages), ADMITTED, sr.diagnose(_record(messages)))
    assert row["decision"] == "REJECT" and "REDUNDANT_RECHECK" in row["reasons"]


def test_c2_flags_an_edit_after_the_passing_run():
    messages = _messages("1 failed, 20 passed in 0.2s\n", "21 passed in 0.2s\n")
    messages[-1:] = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c6", "type": "function",
             "function": {"name": "bash",
                          "arguments": '{"command": "cd /w && sed -i s/a/b/ pkg/core.py"}'}}]},
        {"role": "tool", "tool_call_id": "c6", "name": "bash", "content": ""},
        {"role": "assistant", "content": "Done."},
    ]
    row = bx.evaluate_c2(_record(messages), ADMITTED, sr.diagnose(_record(messages)))
    assert row["decision"] == "REJECT" and "EDIT_AFTER_LAST_TEST" in row["reasons"]


def test_c3_needs_a_real_failing_run_and_stays_untrainable():
    passing = _record(_messages("21 passed in 0.2s\n", "21 passed in 0.2s\n"))
    rejected = bx.evaluate_c3(passing, {"admitted": False, "rejected": ["not_passed"]},
                              sr.diagnose(passing))
    assert rejected["decision"] == "REJECT"
    assert "NO_VISIBLE_FAILURE_EVIDENCE" in rejected["reasons"]

    failing = _record(_messages("1 failed, 20 passed in 0.2s\n", "1 failed, 20 passed in 0.2s\n",
                                edit_between=False,
                                report="I could not fix the parser; one test still fails."))
    row = bx.evaluate_c3(failing, {"admitted": False, "rejected": ["not_passed"]},
                         sr.diagnose(failing))
    assert row is not None
    assert row["decision"] == "ADMIT"
    assert row["training_eligible"] is False and row["review_required"] is True


def test_masked_messages_are_never_supervised():
    record = _record(_messages("1 failed, 20 passed in 0.2s\n", "21 passed in 0.2s\n"))
    diag = sr.diagnose(record)
    row = bx.evaluate_b(record, ADMITTED, diag)
    masked = {msg for msg, _ in row["masked_msgs"]}
    assert masked.isdisjoint(set(row["keep_msgs"]))
