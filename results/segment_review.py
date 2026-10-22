"""Deterministic evidence layer for pool B / pool C construction.

This module does **no** teacher calling and **no** container replay: it turns the
already recorded factory trajectories into the evidence a pool exporter needs —
tool events, test-verification states, patch statistics, mechanical repetition and
grounding — so that every label can be recomputed from the raw JSONL.

Design constraints (see docs/工程问题/2026-10-23-BC池教师执行逻辑与实现顺序.md):
  * tool observations are never supervised; only whole assistant messages can be;
  * a "test passed" label is only emitted when the observation carries a parsable
    runner summary; an exit code alone is not accepted for piped commands;
  * every label carries reason codes and evidence message indices so a reviewer can
    re-derive it without guessing.

The unit of supervision is one assistant message (the factory records native tool
calls, so the call and its argument text live in that message).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# The DSH-compatible tool surface only appends the marker on non-zero exits.
EXIT_MARK = re.compile(r"\[exit code: (-?\d+)\]\s*$")
NO_OUTPUT = re.compile(r"\[no output\]")

TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[\w./-]+\.py$|conftest\.py$|_test\.py$")
# Broad "a test was involved" detector; used only to find candidate events.
TESTY = re.compile(
    r"\bpytest\b|\bpython[0-9.]*\s+-m\s+(pytest|unittest)\b|\btox\b|\bnox\b|"
    r"\bctest\b|\bmake\b[^\n]*\b(check|test)\b|\bcargo\s+test\b|\bgo\s+test\b|"
    r"\bnpm\s+(run\s+)?test\b|\./[\w./-]*(test|tests|runtests)\b|"
    r"\bunittest\b|\bpy\.test\b|\btrial\b|\bjest\b|\bmocha\b")
# Commands that would make a leading path mention a false "mutation".
READ_ONLY_HEAD = re.compile(
    r"^(cat|grep|rg|head|tail|sed\s+-n|less|more|ls|find|wc|git\s+(status|diff|log|show)|"
    r"echo|which|python[0-9.]*\s+-c)\b")
MUTATION = re.compile(
    r"\bsed\s+-i|\bpatch\b|\bgit\s+apply\b|\btee\b|\bcat\s*>|\bpython[0-9.]*\s*-\s*<<|"
    r"\bcp\s|\bmv\s|\brm\s|\btouch\s|\bmkdir\s|\bperl\s+-[pi]|(?<![0-9&])>\s*[^\s&|>]+")
NULL_REDIRECT = re.compile(r"[0-9]?>>?\s*/dev/null|2>&1")
PIPE = re.compile(r"(?<!\|)\|(?!\|)")
HARNESS_GITIGNORE = re.compile(r"AWEAGENT AUTO-GENERATED")

# --- runner summaries -------------------------------------------------------
# Two failure modes this layer must not have (found by an independent audit of the
# first version):
#   * a single shell chain may start several runners ("pytest a | tail -5; pytest b"),
#     so a summary line belongs to *some* run, not necessarily the last one;
#   * "ok <repr>" / "FAIL expected call" are ordinary program output, not go test.
# Therefore: families are parsed only when the command names them, every summary line
# in the tail is collected, and disagreement between runs yields ``unknown``.
COUNT = {
    "passed": re.compile(r"\b(\d+)\s+passed\b"),
    "failed": re.compile(r"\b(\d+)\s+failed\b"),
    "errors": re.compile(r"\b(\d+)\s+errors?\b"),
    "skipped": re.compile(r"\b(\d+)\s+skipped\b"),
    "xfailed": re.compile(r"\b(\d+)\s+xfailed\b"),
    "deselected": re.compile(r"\b(\d+)\s+deselected\b"),
}
DURATION = re.compile(r"\bin \d+(?:\.\d+)?s\b")
UNITTEST_OK = re.compile(r"^OK(\s*\(|\s*$)")
UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in")
UNITTEST_FAILED = re.compile(r"^FAILED \((?P<body>[^)]*)\)")
CTEST = re.compile(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)")
CARGO = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed")
GO_RESULT = re.compile(r"^(ok|FAIL)\s+\S+\s+\d+(?:\.\d+)?s\b")
MOCHA = re.compile(r"(\d+) (passing|failing)")

# A family only unlocks its summary parser when the command actually names it.
FAMILIES = {
    "pytest": re.compile(r"\bpytest\b|\bpy\.test\b"),
    "unittest": re.compile(r"\bunittest\b|python[0-9.]*\s+-m\s+unittest\b"),
    "ctest": re.compile(r"\bctest\b"),
    "cargo": re.compile(r"\bcargo\s+test\b"),
    "go": re.compile(r"\bgo\s+test\b"),
    "mocha": re.compile(r"\bnpm\s+(run\s+)?test\b|\bmocha\b|\bjest\b"),
}
# Running the project's test script counts as a runner even though the summary comes
# from whatever the script invokes.
RUNNER_HINT = re.compile(
    r"\bpytest\b|\bpytest\b|\bpy\.test\b|\bunittest\b|\btox\b|\bnox\b|\bctest\b|"
    r"\bcargo\s+test\b|\bgo\s+test\b|\bnpm\s+(run\s+)?test\b|\bmocha\b|\bjest\b|"
    r"\bmake\s+(test|check|tests)\b|[\w./-]*(?:runtests|run_tests|test_runner)[\w./-]*")
INFRA_MARKERS = re.compile(
    r"No module named|ModuleNotFoundError|ImportError|unrecognized arguments|"
    r"command not found|file or directory not found|No such file or directory|"
    r"No tests ran|no tests ran|error: pathspec|could not import", re.I)

SUMMARY_WINDOW = 100000     # tail scanned for runner summaries; build logs are huge
REPEAT_WINDOW = 4000        # observation identity is compared on a bounded prefix


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text or "")


def observed_exit(observation: str) -> int | None:
    """Exit code the harness appended, or None when the command exited zero."""
    match = EXIT_MARK.search(observation or "")
    return int(match.group(1)) if match else None


def _int(pattern: re.Pattern, text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


@dataclass
class TestResult:
    state: str                 # pass | fail | unknown
    reason: str                # runner_summary | exit_code | no_summary | conflicting_runs | infra
    passed: int = 0
    failed: int = 0
    errors: int = 0
    runner: str = ""
    evidence: str = ""         # the summary line that decided the state
    test_ran: bool = False     # a runner actually produced a result we could read
    skipped: int = 0
    deselected: int = 0

    @property
    def executed(self) -> int:
        """Tests that actually ran (deselected ones were filtered out, not executed)."""
        return self.passed + self.failed + self.errors + self.skipped

    def as_dict(self) -> dict:
        return {"state": self.state, "reason": self.reason, "passed": self.passed,
                "failed": self.failed, "errors": self.errors, "runner": self.runner,
                "evidence": self.evidence[:160], "test_ran": self.test_ran,
                "skipped": self.skipped, "deselected": self.deselected,
                "executed": self.executed}


LINTER_NOISE = re.compile(
    r"in \d+ files?\b|source files?|no issues found|Success: no issues|"
    r"\bFound \d+ errors?\b", re.I)
CODE_SHAPE = re.compile(r"\b(def|class|import|return|assert|lambda|self\.)\b")
SUMMARY_MAX_LEN = 200


def _looks_like_summary_line(line: str) -> bool:
    """Reject tool output that merely contains a number next to "passed"/"errors"."""
    if len(line) > SUMMARY_MAX_LEN:
        return False
    if LINTER_NOISE.search(line) or CODE_SHAPE.search(line):
        return False
    return True


def _one_line_summary(line: str, families: set[str]) -> TestResult | None:
    """Parse a single summary line; return None when it is not one.

    ``families`` gates the parsers: a line that merely starts with "ok" is only a
    go-test result when the command actually ran ``go test``. A summary line must also
    look like a runner summary, not like mypy ("Found 3 errors in 1 file") or a line of
    source code that happens to contain "1 passed in".
    """
    line = line.strip()
    if not line or not _looks_like_summary_line(line):
        return None
    if "pytest" in families:
        counts = {name: _int(pattern, line) for name, pattern in COUNT.items()}
        if any(value is not None for value in counts.values()):
            # "errors" alone is how linters talk; a runner summary needs a pass/marker
            # count or a duration to be recognisable
            has_context = bool(DURATION.search(line) or counts["passed"]
                               or counts["deselected"] or counts["skipped"] or counts["xfailed"])
            if has_context:
                passed = counts["passed"] or 0
                failed = counts["failed"] or 0
                errors = counts["errors"] or 0
                state = "fail" if (failed or errors) else ("pass" if passed else "unknown")
                return TestResult(state, "runner_summary", passed, failed, errors, "pytest",
                                  line, test_ran=True,
                                  skipped=counts["skipped"] or 0,
                                  deselected=counts["deselected"] or 0)
    if "ctest" in families:
        ctest = CTEST.search(line)
        if ctest:
            total, failed = int(ctest.group(3)), int(ctest.group(2))
            return TestResult("fail" if failed else "pass", "runner_summary", total - failed,
                              failed, 0, "ctest", line, test_ran=True)
    if "cargo" in families:
        cargo = CARGO.search(line)
        if cargo:
            return TestResult("pass" if cargo.group(1) == "ok" else "fail", "runner_summary",
                              int(cargo.group(2)), int(cargo.group(3)), 0, "cargo", line,
                              test_ran=True)
    if "mocha" in families:
        mocha = MOCHA.search(line)
        if mocha:
            value, kind = int(mocha.group(1)), mocha.group(2)
            return TestResult("fail" if kind == "failing" else "unknown", "runner_summary",
                              0, value if kind == "failing" else 0, 0, "mocha", line,
                              test_ran=True)
    if "go" in families:
        go = GO_RESULT.match(line)
        if go:
            return TestResult("fail" if go.group(1) == "FAIL" else "pass", "runner_summary",
                              0, 1 if go.group(1) == "FAIL" else 0, 0, "go", line,
                              test_ran=True)
    return None


def command_families(command: str) -> set[str]:
    """Runner families the command actually names (gates the summary parsers)."""
    return {name for name, pattern in FAMILIES.items() if pattern.search(command or "")}


def command_segments(command: str) -> list[str]:
    """Split a shell chain into commands, keeping heredoc bodies inside one segment."""
    raw = [part.strip() for part in re.split(r"&&|\|\||;|\n", strip_null(command))
           if part.strip()]
    merged: list[str] = []
    heredoc = None
    for part in raw:
        if heredoc is not None:
            merged[-1] = merged[-1] + "\n" + part
            if part == heredoc:
                heredoc = None
            continue
        merged.append(part)
        marker = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)", part)
        if marker:
            heredoc = marker.group(1)
    return merged


def normalize_command(command: str) -> str:
    """Drop workspace prefixes and whitespace so two runs of one check compare equal."""
    text = re.sub(r"^\s*(?:cd\s+\S+\s*(?:&&|;)\s*)+", "", strip_null(command)).strip()
    return re.sub(r"\s+", " ", text)


def runner_segments(command: str) -> list[tuple[str, set[str]]]:
    """(segment text, families it names) for every segment that starts a runner."""
    out = []
    for segment in command_segments(command):
        families = {name for name, pattern in FAMILIES.items() if pattern.search(segment)}
        if families:
            out.append((segment, families))
    return out


def _scan_summaries(text: str, families: set[str]) -> list[TestResult]:
    """Collect every runner summary in the observation, in order.

    The tail window is the normal path (test output ends with its summary); when a big
    trailing diff pushed the summary out of it, the whole text is scanned as a fallback
    instead of silently degrading to "unknown".
    """
    if not families:
        return []
    found = _scan_lines(text[-SUMMARY_WINDOW:].splitlines(), families)
    if not found and len(text) > SUMMARY_WINDOW:
        found = _scan_lines(text.splitlines(), families)
    return found


def _scan_lines(lines: list[str], families: set[str]) -> list[TestResult]:
    found: list[TestResult] = []
    position = len(lines) - 1
    while position >= 0:
        line = lines[position].strip()
        if "unittest" in families and UNITTEST_OK.match(line):
            ran = 0
            for back in lines[max(0, position - 4):position + 1]:
                ran_match = UNITTEST_RAN.match(back.strip())
                if ran_match:
                    ran = int(ran_match.group(1))
            found.append(TestResult("pass", "runner_summary", ran, 0, 0, "unittest", line,
                                    test_ran=True))
        elif "unittest" in families and UNITTEST_FAILED.match(line):
            body = UNITTEST_FAILED.match(line).group("body")
            bad = sum(int(n) for n in re.findall(r"=(\d+)", body)) or 1
            found.append(TestResult("fail", "runner_summary", 0, bad, 0, "unittest", line,
                                    test_ran=True))
        else:
            parsed = _one_line_summary(line, families)
            if parsed is not None:
                found.append(parsed)
        position -= 1
    found.reverse()
    return found


def parse_test_result(command: str, observation: str) -> TestResult:
    """Decide pass/fail for one tool call from its observable text.

    Attribution rules, in order:
      1. the command must name a runner (or start a test script) -> else unknown;
      2. collect every summary line in the observation tail;
      3. if the last runner segment is *not* piped, its output is complete, so the
         last summary belongs to it -> that is the result;
      4. if the last runner segment is piped, a summary cannot be attributed, so the
         result is only accepted when every summary in the output agrees;
      5. without a summary, a non-piped non-zero exit code is a failure unless an
         environment marker explains it (missing module, bad arguments, no tests).
    A piped run never falls back to the exit code, because the exit status belongs to
    the last command of the pipe.
    """
    runners = runner_segments(command or "")
    segments = command_segments(command or "")
    families = set().union(*[item[1] for item in runners]) if runners else set()
    if not families:
        if not RUNNER_HINT.search(command or ""):
            return TestResult("unknown", "no_summary", 0, 0, 0, "", "", test_ran=False)
        families = {"pytest", "unittest"}      # test script; parse only pytest shapes
    text = strip_ansi(observation or "")
    piped = bool(PIPE.search(strip_null(command)))
    last_piped = bool(runners) and bool(PIPE.search(runners[-1][0]))
    exit_code = observed_exit(text)
    # did the shell chain continue past the last runner? a failure there belongs to the
    # trailing command, so "the tests passed" would be the wrong reading of the call
    last_runner_is_last = bool(runners) and segments and runners[-1][0] == segments[-1]

    summaries = _scan_summaries(text, families)
    states = {item.state for item in summaries if item.state != "unknown"}
    # `a || b` runs b only when a failed, so "the last summary passed" is not the same
    # as "the tests passed"; with more than one runner in a conditional chain we abstain
    conditional = len(runners) > 1 and "||" in (command or "")
    # two different runner families in one chain can overwrite each other's verdict
    family_states = {}
    for family in families:
        per_family = _scan_summaries(text, {family})
        if per_family:
            family_states[family] = {item.state for item in per_family
                                     if item.state != "unknown"}
    mixed = len({frozenset(value) for value in family_states.values() if value}) > 1
    if conditional and len(states) > 1:
        return TestResult("unknown", "conditional_chain", 0, 0, 0, "mixed",
                          summaries[-1].evidence, test_ran=True)
    if mixed:
        return TestResult("unknown", "mixed_runners", 0, 0, 0, "mixed",
                          summaries[-1].evidence if summaries else "", test_ran=True)
    if len(states) == 1:
        trailing_failure = exit_code is not None and bool(runners) and not last_runner_is_last
        if trailing_failure:
            chosen = next(item for item in reversed(summaries) if item.state in states)
            return TestResult("unknown", "trailing_command_failed", chosen.passed,
                              chosen.failed, chosen.errors, chosen.runner,
                              f"{chosen.evidence} | [exit code: {exit_code}]",
                              test_ran=True, skipped=chosen.skipped,
                              deselected=chosen.deselected)
        for item in reversed(summaries):
            if item.state in states:
                return item
    if len(states) > 1:
        if runners and not last_piped and exit_code is None:
            return summaries[-1]               # the last run is fully visible
        return TestResult("unknown", "conflicting_runs", 0, 0, 0, "mixed",
                          summaries[-1].evidence, test_ran=True)

    infra = INFRA_MARKERS.search(text[-4000:]) is not None
    if infra:
        return TestResult("unknown", "infra", 0, 0, 0, "", "environment failure", False)
    # an exit code only describes the test when the failing command *is* the runner;
    # otherwise it reports a crash in some helper script ("python - <<PY ...")
    if exit_code is not None and not piped and last_runner_is_last:
        return TestResult("fail", "exit_code", 0, 0, 0, "unknown", f"exit code {exit_code}",
                          test_ran=False)
    if exit_code is not None and not last_runner_is_last:
        return TestResult("unknown", "trailing_command_failed", 0, 0, 0, "",
                          f"exit code {exit_code} came from a non-test command", False)
    return TestResult("unknown", "no_summary", 0, 0, 0, "", "", test_ran=False)


def strip_null(command: str) -> str:
    return NULL_REDIRECT.sub(" ", command or "")


def is_mutation(command: str) -> bool:
    """True when the command can write to the workspace."""
    cleaned = strip_null(command)
    for part in re.split(r"&&|\|\||;", cleaned):
        part = part.strip()
        if not part or READ_ONLY_HEAD.match(part):
            continue
        if MUTATION.search(part):
            return True
    return False


def is_testy(command: str) -> bool:
    return bool(TESTY.search(command or ""))


def normalize_args(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


@dataclass
class ToolEvent:
    msg: int                  # index of the assistant message that emitted the call
    call_id: str
    name: str
    args: dict
    obs_msg: int | None
    observation: str
    kind: str                 # edit | mutate | test | read | other
    command: str = ""
    exit_code: int | None = None
    test: TestResult | None = None
    signature: str = ""
    mutates: bool = False     # the command can write to the workspace
    runs_test: bool = False   # the command mentions a test runner

    def as_dict(self) -> dict:
        return {"msg": self.msg, "call_id": self.call_id, "name": self.name,
                "kind": self.kind, "obs_msg": self.obs_msg,
                "command": self.command[:300], "exit_code": self.exit_code,
                "test": self.test.as_dict() if self.test else None,
                "signature": self.signature,
                "mutates": self.mutates, "runs_test": self.runs_test}


def events(messages) -> list[ToolEvent]:
    """Flatten the trajectory into tool events, keeping message indices intact."""
    by_call = {m.get("tool_call_id"): i for i, m in enumerate(messages)
               if m.get("role") == "tool"}
    out: list[ToolEvent] = []
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name") or call.get("name") or ""
            args = normalize_args(function.get("arguments"))
            obs_msg = by_call.get(call.get("id"))
            observation = strip_ansi((messages[obs_msg].get("content") or "")
                                     if obs_msg is not None else "")
            command = str(args.get("command", "")) if name == "bash" else ""
            kind = "other"
            test = None
            mutates = False
            runs_test = False
            if name == "edit":
                kind = "edit"
            elif name == "read":
                kind = "read"
            elif name == "bash":
                mutates = is_mutation(command)
                runs_test = is_testy(command)
                if runs_test:
                    test = parse_test_result(command, observation)
                kind = "test" if runs_test else ("mutate" if mutates else "read")
                if kind == "read" and not command:
                    kind = "other"
            signature = sha256(
                json.dumps([name, sorted(args.items())], ensure_ascii=False,
                           default=str).encode()).hexdigest()[:16]
            out.append(ToolEvent(index, call.get("id") or "", name, args, obs_msg, observation,
                                 kind, command, observed_exit(observation), test, signature,
                                 mutates, runs_test))
    return out


@dataclass
class PatchInfo:
    files: list[str] = field(default_factory=list)        # every path in the diff
    source_files: list[str] = field(default_factory=list)  # non-test, non-harness
    test_files: list[str] = field(default_factory=list)
    content_lines: int = 0        # +/- lines in source files, harness blocks excluded
    test_lines: int = 0           # +/- lines in test files
    harness_only: bool = False

    def as_dict(self) -> dict:
        return {"files": self.files, "source_files": self.source_files,
                "test_files": self.test_files, "content_lines": self.content_lines,
                "test_lines": self.test_lines, "harness_only": self.harness_only}


def patch_info(patch: str) -> PatchInfo:
    """Split a unified diff into source edits, test edits and harness noise."""
    info = PatchInfo()
    in_harness = False
    current_is_test = False
    for line in (patch or "").splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1].strip()
            info.files.append(path)
            current_is_test = bool(TEST_PATH.search(path))
            if current_is_test:
                info.test_files.append(path)
            in_harness = False
            continue
        if line.startswith(("+++", "---", "@@", "index ", "new file", "deleted file",
                            "similarity", "rename ")):
            continue
        if HARNESS_GITIGNORE.search(line):
            in_harness = True
            continue
        if line.startswith(("+", "-")) and line.strip() not in ("+", "-"):
            if in_harness:
                info.harness_only = True
                continue
            if current_is_test:
                info.test_lines += 1
            else:
                info.content_lines += 1
    info.source_files = [f for f in info.files
                         if f not in info.test_files and not f.endswith(".gitignore")]
    return info


def observation_digest(observation: str) -> str:
    normalized = re.sub(r"\s+", " ", observation[:REPEAT_WINDOW]).strip()
    return sha256(normalized.encode()).hexdigest()[:16]


def grounded_files(evs: list[ToolEvent], files: list[str], before_msg: int) -> list[str]:
    """Edited paths that were already mentioned by an earlier command or observation."""
    grounded = []
    for path in files:
        needle = path.lower()
        for event in evs:
            if event.msg >= before_msg:
                break
            haystack = (event.command + "\n" + event.observation[:REPEAT_WINDOW]).lower()
            if needle and needle in haystack:
                grounded.append(path)
                break
    return grounded


@dataclass
class Diagnosis:
    record_id: str
    messages: list
    evs: list[ToolEvent]
    patch: PatchInfo
    first_mut_msg: int | None
    first_mut_step: int | None
    last_mut_msg: int | None
    first_edit_grounded: bool
    test_events: list[ToolEvent]
    last_pass_msg: int | None
    failure_then_pass: list[tuple[int, int, int]]   # (fail obs msg, pass obs msg, correction edit msg)
    duplicate_msg: dict[int, str]                   # masked message -> reason code
    final_msg: int | None
    final_text: str

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "first_mut_msg": self.first_mut_msg, "first_mut_step": self.first_mut_step,
            "last_mut_msg": self.last_mut_msg, "first_edit_grounded": self.first_edit_grounded,
            "tests": [e.as_dict() for e in self.test_events],
            "last_pass_msg": self.last_pass_msg,
            "failure_then_pass": [list(x) for x in self.failure_then_pass],
            "masked": self.duplicate_msg, "final_msg": self.final_msg,
            "patch": self.patch.as_dict(),
        }


def diagnose(record: dict) -> Diagnosis:
    """Compute every deterministic fact the pool rules are allowed to use."""
    messages = record.get("messages") or []
    evs = events(messages)
    patch = patch_info(record.get("patch") or "")
    assistant_msgs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]

    mutations = [e for e in evs if e.name == "edit" or e.mutates]
    first_mut = mutations[0] if mutations else None
    last_mut = mutations[-1] if mutations else None
    first_step = None
    if first_mut is not None:
        first_step = sum(1 for i in assistant_msgs if i <= first_mut.msg)
    grounded = bool(first_mut) and bool(grounded_files(evs, patch.source_files, first_mut.msg))

    tests = [e for e in evs if e.test is not None and e.test.test_ran]
    passes = [e for e in tests if e.test.state == "pass"]
    last_pass = passes[-1].obs_msg if passes else None

    # failure -> correction -> verified pass, all inside this single session
    pairs = []
    seen_pairs: set[tuple[int, int]] = set()
    for test in tests:
        if test.test.state != "fail" or test.obs_msg is None:
            continue
        next_pass = next((e for e in passes if e.obs_msg and e.obs_msg > test.obs_msg), None)
        if next_pass is None:
            continue
        between = [e for e in mutations if test.msg < e.msg < next_pass.obs_msg]
        if not between:
            continue
        key = (test.obs_msg, next_pass.obs_msg)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pairs.append((test.obs_msg, next_pass.obs_msg, between[0].msg))

    # mechanical repetition: same call signature and same observation as an earlier call
    masked: dict[int, str] = {}
    first_seen: dict[str, str] = {}
    for event in evs:
        key = event.signature
        digest = observation_digest(event.observation)
        previous = first_seen.get(key)
        if previous is not None and previous == digest:
            masked[event.msg] = "MECHANICAL_REPEAT"
        elif previous is None:
            first_seen[key] = digest
    # verbatim repeated assistant text with no tool call adds no new information
    seen_text: dict[str, int] = {}
    for index in assistant_msgs:
        message = messages[index]
        text = (message.get("content") or "").strip()
        if message.get("tool_calls") or len(text) < 80:
            continue
        key = sha256(re.sub(r"\s+", " ", text).encode()).hexdigest()
        if key in seen_text:
            masked[index] = "REPEAT_TEXT"
        else:
            seen_text[key] = index

    final_index = None
    final_text = ""
    for index in reversed(assistant_msgs):
        if (messages[index].get("content") or "").strip():
            final_index = index
            final_text = messages[index].get("content") or ""
            break

    return Diagnosis(str(record.get("instance_id")), messages, evs, patch,
                      first_mut.msg if first_mut else None, first_step,
                      last_mut.msg if last_mut else None, grounded,
                      tests, last_pass, pairs, masked, final_index, final_text)


# --- report honesty screen (C3) --------------------------------------------
SUCCESS_CLAIM = re.compile(
    r"\ball tests? pass(ed)?\b|\btests? (?:now )?pass\b|\beverything (?:now )?passes\b|"
    r"\bfixed\b|\bresolved\b|\bimplemented successfully\b|\bcompleted successfully\b|"
    r"\bworks correctly\b|\ball green\b|\bsuccessfully (?:fixed|implemented|resolved)\b|"
    r"\bthe (?:issue|bug|problem) is (?:now )?(?:fixed|resolved)\b", re.I)
LIMITATION = re.compile(
    r"\bunable\b|\bcould not\b|\bcouldn't\b|\bnot able\b|\bnot possible\b|\bfailed to\b|"
    r"\bdid not\b|\bdidn't\b|\bcannot\b|\bcan't\b|\bnot fixed\b|\bnot resolved\b|"
    r"\bstill fail(s|ing)?\b|\bremains?\b|\bremaining\b|\bincomplete\b|\bnot complete\b|"
    r"\bnot implemented\b|\bunresolved\b|\bno (?:working |successful )?(?:change|patch|fix)\b|"
    r"\bwas not able\b|\bblocked\b|\bout of (?:budget|steps)\b", re.I)
NEGATION = re.compile(r"\b(no|not|n't|without|never)\b", re.I)


def honesty_screen(text: str) -> dict:
    """Keyword screen for an honest "I did not finish" report.

    This is a **screen**, not a semantic verdict: it can reject a report but it cannot
    prove honesty. Admitted C3 rows therefore carry ``review_required`` so a semantic
    reviewer still has to confirm them before they are used for training.
    """
    body = (text or "")[:6000]
    claims = [m.group(0) for m in SUCCESS_CLAIM.finditer(body)]
    limits = [m.group(0) for m in LIMITATION.finditer(body)]
    unnegated_claims = []
    for match in SUCCESS_CLAIM.finditer(body):
        prefix = body[max(0, match.start() - 60):match.start()]
        if NEGATION.search(prefix):
            continue
        unnegated_claims.append(match.group(0))
    return {"success_claims": unnegated_claims, "claim_hits": claims,
            "limitation_hits": limits,
            "passes": bool(limits) and not unnegated_claims,
            "review_required": bool(unnegated_claims) or not limits}


def message_targets(diagnosis: Diagnosis, keep: set[int]) -> dict:
    """Turn a keep-set into the exact per-message supervision plan (v1: message level)."""
    targets = {}
    for index in sorted(keep):
        message = diagnosis.messages[index]
        if message.get("role") != "assistant":
            continue
        if index in diagnosis.duplicate_msg:
            targets[index] = "MASK"
            continue
        targets[index] = "KEEP"
    return targets
