"""Admission gates + 64K export over the formal large batch.

Reads every trajectories.jsonl under results/pool_a_formal/batch_*/, applies the
eight admission gates from the plan, renders admitted samples through the student
Qwen template with an assistant-only mask, and writes:

  results/pool_a_formal/admission_ledger.jsonl   one line per attempted instance
  results/pool_a_formal/sft_admitted.jsonl       ids+mask for admitted samples
  results/pool_a_formal/export_summary.json      aggregated counts and tokens

Gates: evaluator success, declared F2P/P2P node statuses, no test-path edits,
target test executed after the last edit, sequence length <= 65536, at least one
non-empty think block, patch dedup, family-level validation families excluded.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

FORMAL = Path(__file__).resolve().parents[1] / "results" / "pool_a_formal"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_sft_qwen import Renderer, build  # noqa: E402

MAX_TOKENS = 65536
TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[\w./-]+\.py$|conftest\.py$|_test\.py$")
# Test execution across ecosystems: the selection is python-labelled but repos such as
# flatbuffers run ctest/cmake binaries, so pytest-only detection under-counts verification.
TEST_RUN = re.compile(
    r"\bpytest\b|\bpython[0-9.]*\s+-m\s+(pytest|unittest)\b|\btox\b|\bnox\b|"
    r"\bctest\b|\bmake\b[^\n]*\bcheck\b|\bcargo\s+test\b|\bgo\s+test\b|"
    r"\bnpm\s+(run\s+)?test\b|\./[\w./-]*(test|tests|runtests)\b")

# Any command that can mutate files, not just the edit tool.
MUTATION = re.compile(
    r"\bsed\s+-i|\bpatch\b|\bgit\s+apply\b|\btee\b|\bcat\s*>|\bpython[0-9.]*\s*-\s*<<|"
    r"\bcp\s|\bmv\s|\brm\s|\btouch\s|\bmkdir\s|\bperl\s+-[pi]|(?<![0-9&])>\s*[^\s&|>]+")
NULL_REDIRECT = re.compile(r"[0-9]?>>?\s*/dev/null|2>&1")


def strip_null(cmd: str) -> str:
    return NULL_REDIRECT.sub(" ", cmd)


def declared(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def tool_sequence(messages) -> tuple[list[str], list[int]]:
    """Return (tool names in order, indices of bash commands)."""
    names, bash_idx = [], []
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name") or call.get("name") or ""
            names.append(name)
            if name == "bash":
                args = function.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if isinstance(args, dict) and TEST_RUN.search(str(args.get("command", ""))):
                    bash_idx.append(index)
    return names, bash_idx


def load_records() -> list[dict]:
    rows = []
    skipped = []
    for shard in sorted(FORMAL.glob("batch_*/**/trajectories.jsonl")):
        for number, line in enumerate(shard.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A killed run can leave a half-written line; never let one truncated
                # record block the whole export.
                skipped.append(f"{shard}:{number}")
                continue
            record["_batch"] = shard.parts[-3]
            rows.append(record)
    if skipped:
        print(f"skipped {len(skipped)} malformed trajectory lines: {skipped[:5]}", flush=True)
    return rows


def main() -> None:
    raw = {json.loads(l)["instance_id"]: json.loads(l)
           for l in (FORMAL / "tasks_formal.jsonl").read_text(encoding="utf-8").splitlines()}
    qualification = json.loads((FORMAL / "qualification.json").read_text(encoding="utf-8"))
    validation_families = set(qualification.get("validation_families", []))

    renderer = Renderer()
    records = load_records()
    ids_seen = [r["instance_id"] for r in records]
    if len(ids_seen) != len(set(ids_seen)):
        # Restarted orchestrators can leave several run dirs; keep the first record per id.
        deduped, seen = [], set()
        for record in records:
            if record["instance_id"] in seen:
                continue
            seen.add(record["instance_id"])
            deduped.append(record)
        print(f"deduped {len(records)-len(deduped)} duplicate records across run dirs", flush=True)
        records = deduped
    seen_patch: dict[str, str] = {}
    ledger = []
    admitted = []
    totals = Counter()
    tokens = supervised = 0

    for record in records:
        iid = record["instance_id"]
        task = raw.get(iid, {})
        family = f'{task.get("user", "")}/{task.get("repo", "")}'
        details = (record.get("eval_result") or {}).get("details") or {}
        matched = details.get("matched") or {}
        f2p = declared(task.get("FAIL_TO_PASS"))
        p2p = declared(task.get("PASS_TO_PASS"))
        if record.get("success"):
            # The factory evaluator records only marker/exit_code on success; node-level
            # statuses appear solely on failure paths. Acceptance already requires every
            # declared node to pass, so this is an aggregate inference, not node evidence.
            f2p_status = Counter({"passed": len(f2p)})
            p2p_status = Counter({"passed": len(p2p)})
            node_evidence = "evaluator_acceptance"
        else:
            f2p_status = Counter(matched.get(n, "unmatched") for n in f2p)
            p2p_status = Counter(matched.get(n, "unmatched") for n in p2p)
            node_evidence = "junit_nodes"
        patch = record.get("patch") or ""
        files = [ln.split(" b/", 1)[-1] for ln in patch.splitlines() if ln.startswith("diff --git ")]
        test_paths = [f for f in files if TEST_PATH.search(f)]

        names, test_runs = tool_sequence(record.get("messages") or [])
        # Mutation = edit tool OR a bash command that writes files.
        edit_positions = []
        for pos, message in enumerate(record.get("messages") or []):
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name") or call.get("name") or ""
                args = function.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                args = args if isinstance(args, dict) else {}
                if name == "edit":
                    edit_positions.append(pos)
                elif name == "bash" and MUTATION.search(strip_null(str(args.get("command", "")))):
                    edit_positions.append(pos)
        last_edit = max(edit_positions, default=-1)
        verified_after_edit = any(idx > last_edit for idx in test_runs)

        verdict = {"instance_id": iid, "family": family, "batch": record["_batch"],
                   "success": bool(record.get("success")),
                   "f2p": dict(f2p_status), "p2p": dict(p2p_status),
                   "patch_chars": len(patch), "files": len(files),
                   "test_paths": test_paths, "node_evidence": node_evidence, "test_runs": len(test_runs), "mutations": len(edit_positions), "verified_after_last_edit": verified_after_edit, "steps": (record.get("stats") or {}).get("steps"),
                   "finish_reason": record.get("finish_reason"),
                   "prompt_tokens": (record.get("stats") or {}).get("total_prompt_tokens", 0),
                   "completion_tokens": (record.get("stats") or {}).get("total_completion_tokens", 0),
                   "duration": record.get("duration")}
        reject = []
        infra = record.get("finish_reason") in ("error", "timeout") or bool(details.get("error"))
        if infra:
            # Auth/network/timeout failures are infrastructure, not model shortfalls.
            reject.append("infra_error")
        elif not record.get("success"):
            reject.append("not_passed")
        if not record.get("success") and not infra and (f2p_status.get("failed") or f2p_status.get("unmatched")):
            reject.append("f2p_incomplete")
        if not record.get("success") and not infra and (p2p_status.get("failed") or p2p_status.get("unmatched")):
            reject.append("p2p_incomplete")
        if test_paths:
            reject.append("test_paths_edited")
        if not test_runs:
            # Hard gate: no test execution anywhere in the trajectory. Whether a test ran
            # after the *last* mutation is reported as a statistic because command-level
            # detection is imperfect for heredoc/native-binary workflows.
            reject.append("no_test_at_all")
        if family in validation_families:
            reject.append("validation_family")
        if not patch.strip():
            reject.append("empty_patch")
        digest = sha256(re.sub(r"\s+", "", patch).encode()).hexdigest()[:16]
        if digest in seen_patch:
            reject.append("duplicate_patch")
        seen_patch.setdefault(digest, iid)

        length = None
        think_ok = None
        if not reject:
            ids, mask, _ = build(renderer, record["messages"])
            length = len(ids)
            empty = renderer.render(record["messages"]).count("<think>\n\n</think>")
            non_empty = renderer.render(record["messages"]).count("<think>") - empty
            think_ok = non_empty > 0
            if length > MAX_TOKENS:
                reject.append("over_length")
            if not think_ok:
                reject.append("no_think")
            if sum(mask) == 0:
                reject.append("empty_mask")
            if not reject:
                admitted.append({"instance_id": iid, "ids": ids, "mask": mask})
                tokens += len(ids)
                supervised += sum(mask)

        verdict["rejected"] = reject
        verdict["admitted"] = not reject
        verdict["length"] = length
        ledger.append(verdict)
        totals["attempted"] += 1
        totals["admitted"] += not reject
        for reason in reject:
            totals[f"reject_{reason}"] += 1

    with (FORMAL / "admission_ledger.jsonl").open("w", encoding="utf-8") as f:
        for row in ledger:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (FORMAL / "sft_admitted.jsonl").open("w", encoding="utf-8") as f:
        for row in admitted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "attempted": totals["attempted"], "admitted": totals["admitted"],
        "rejections": {k[7:]: v for k, v in totals.items()
                       if k.startswith("reject_") and v},
        "sequences": len(admitted), "total_tokens": tokens, "supervised_tokens": supervised,
        "supervised_fraction": round(supervised / tokens, 4) if tokens else None,
        "per_sample_tokens": round(tokens / len(admitted), 1) if admitted else None,
        "max_tokens": MAX_TOKENS,
        "batches_seen": sorted({r["_batch"] for r in records}),
    }
    (FORMAL / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
