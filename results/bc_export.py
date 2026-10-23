"""Construct pool B (correction) and pool C (behaviour) from the recorded pool-A runs.

Input is the already-recorded factory output: ``results/pool_a_formal/batch_*/**/
trajectories.jsonl`` plus its authoritative ``admission_ledger.jsonl``. Nothing is
re-executed, no container is started and no teacher is called: every label below is
recomputable from those files, which is the whole point of the deterministic v1.

What each pool is:

  C1  grounded landing   admitted trajectory whose first real mutation happens at
                          step <= C1_STEP *and* touched a file it had already read.
  C2  verified stop      admitted trajectory whose last mutation is followed by a
                          parsable passing test run, after which nothing is edited,
                          nothing is re-run, and the only remaining output is the
                          delivery message.
  C3  honest incomplete  a *failed* run that ends with a real observed failure and a
                          report that does not claim success. Deterministic screen
                          only: rows stay ``review_required`` and are **not** counted
                          as resolved.
  B   correction pair    a real failing observation inside an admitted trajectory,
                          then real edits, then a passing re-run of the same area,
                          then independent evaluator acceptance.

Supervision is always message-level (v1). The failed prefix of a B sample, all tool
observations, the user message and the system prompt are loss=0 by construction,
because a mask token can only be set on an assistant block that is listed in
``keep_msgs``.

Usage:
    python results/bc_export.py                # rebuild pools, ledgers, payloads, index
    python results/bc_export.py --check        # verify index == ledgers == payloads
    python results/bc_export.py --summary      # print the funnel from the ledger
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

RESULTS = Path(__file__).resolve().parent
FORMAL = RESULTS / "pool_a_formal"
OUT = RESULTS / "pool_bc"
DB = FORMAL / "index.sqlite3"
MASK_OVERLAY = OUT / "review" / "overlay.json"
sys.path.insert(0, str(RESULTS))
import segment_review as sr  # noqa: E402
from split_isolation import is_split_quarantined  # noqa: E402
from to_sft_qwen import Renderer, Rendered  # noqa: E402

MAX_TOKENS = 65536
C1_STEP = 6
C1_STRONG_STEP = 4
POOLS = ("C1", "C2", "C3", "B")
# Re-running a test with tests deselected/negated teaches evasion, not correction.
TEST_EVASION = re.compile(r"--deselect|\s-k\s+['\"]?not |--ignore=\S+\s*$|pytest\.skip|xfail")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pool_samples (
    sample_id       TEXT PRIMARY KEY,
    pool            TEXT NOT NULL,
    instance_id     TEXT NOT NULL,
    family          TEXT,
    batch           TEXT,
    decision        TEXT NOT NULL,          -- ADMIT | REJECT | REVIEW
    training_eligible INTEGER NOT NULL,     -- 1 only for ADMIT
    review_required INTEGER NOT NULL,       -- deterministic labels awaiting semantics
    reasons         TEXT,                   -- JSON array of reason codes
    keep_msgs       TEXT,                   -- JSON array of supervised message indices
    masked_msgs     TEXT,                   -- JSON array of [msg, reason]
    evidence        TEXT,                   -- JSON object (commands, summaries, receipts)
    seq_tokens      INTEGER,
    supervised_tokens INTEGER,
    supervised_fraction REAL,
    length_gate     INTEGER,
    spans_ok        INTEGER,                -- mask == union of stored spans
    payload_path    TEXT,
    payload_sha256  TEXT,
    created_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_pool_samples_pool ON pool_samples(pool, decision);
CREATE INDEX IF NOT EXISTS idx_pool_samples_instance ON pool_samples(instance_id);

DROP VIEW IF EXISTS v_pool_summary;
DROP VIEW IF EXISTS v_pool_supervision;
DROP VIEW IF EXISTS v_pool_b_corrections;
CREATE VIEW v_pool_summary AS
SELECT pool, decision, COUNT(*) AS n,
       SUM(training_eligible) AS eligible,
       SUM(supervised_tokens) AS supervised_tokens,
       SUM(seq_tokens) AS seq_tokens
FROM pool_samples GROUP BY pool, decision ORDER BY pool, decision;

DROP VIEW IF EXISTS v_pool_supervision;
CREATE VIEW v_pool_supervision AS
SELECT pool,
       COUNT(*) AS samples,
       ROUND(AVG(supervised_fraction), 4) AS avg_supervised_fraction,
       ROUND(AVG(seq_tokens), 0) AS avg_seq_tokens,
       ROUND(AVG(supervised_tokens), 0) AS avg_supervised_tokens,
       MAX(seq_tokens) AS max_seq_tokens
FROM pool_samples WHERE training_eligible = 1 GROUP BY pool;

DROP VIEW IF EXISTS v_pool_b_corrections;
CREATE VIEW v_pool_b_corrections AS
SELECT sample_id, instance_id,
       json_extract(evidence, '$.fail_state.evidence') AS fail_summary,
       json_extract(evidence, '$.pass_state.evidence') AS pass_summary,
       json_extract(evidence, '$.receipt.f2p_passed') AS f2p_passed,
       json_extract(evidence, '$.receipt.p2p_passed') AS p2p_passed
FROM pool_samples WHERE pool = 'B' AND decision = 'ADMIT';
"""


# --- loading ---------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def load_records() -> list[dict]:
    """Dedup exactly like formal_export: first record per instance wins."""
    records, seen = [], set()
    for shard in sorted(FORMAL.glob("batch_*/**/trajectories.jsonl")):
        for number, line in enumerate(shard.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["instance_id"] in seen:
                continue
            seen.add(record["instance_id"])
            record["_batch"] = shard.parts[-3]
            records.append(record)
    return records


def load_admission() -> dict[str, dict]:
    return {row["instance_id"]: row
            for row in load_jsonl(FORMAL / "admission_ledger.jsonl")}


def load_overlay(path: Path) -> dict | None:
    """Read the model-review overlay written by ``model_review.py --apply``.

    Two verdicts travel through it: a sample-level QUARANTINE drops the trajectory, and
    a per-message MASK narrows the supervised messages. Returning ``None`` for a missing
    file keeps the plain export path unchanged.
    """
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    quarantine = {item["sample_id"] for item in raw.get("excluded_by_model") or []}
    masks = {sample: {int(msg) for msg in msgs}
             for sample, msgs in (raw.get("sample_masks") or {}).items()}
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
            "quarantine": quarantine, "masks": masks, "receipt": raw}


def apply_overlay_row(row: dict, overlay: dict) -> dict:
    """Narrow one ADMIT row by the review overlay, before it is rendered."""
    sample = row["sample_id"]
    receipt = overlay.get("receipt") or {}
    if (row.get("pool") == "C3" and receipt.get("complete_review_required")
            and sample in receipt.get("reviewed_sample_ids", [])):
        row = {**row, "training_eligible": True, "review_required": False}
    if sample in overlay["quarantine"]:
        return {**row, "decision": "REJECT", "training_eligible": False,
                "reasons": list(row["reasons"]) + ["MODEL_QUARANTINE"],
                "reasons_by_model": ["QUARANTINE"]}
    dropped = overlay["masks"].get(sample) or set()
    keep = [int(msg) for msg in row["keep_msgs"]]
    kept = [msg for msg in keep if msg not in dropped]
    if not dropped:
        return row
    row = {**row, "keep_msgs": kept,
           "model_masked": len(keep) - len(kept),
           "masked_msgs": list(row.get("masked_msgs") or []) +
                          [[msg, "MODEL_REVIEW"] for msg in keep if msg in dropped],
           "reasons_by_model": ["MASK"]}
    if not kept:
        row.update({"decision": "REJECT", "training_eligible": False,
                    "reasons": list(row["reasons"]) + ["MODEL_EMPTY_MASK"]})
    return row


# --- shared evidence helpers ----------------------------------------------

def assistant_msgs(messages) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def edit_path(event: sr.ToolEvent) -> str:
    if event.name != "edit":
        return ""
    for key in ("path", "file_path", "filename", "file"):
        value = event.args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def mutates_test_file(event: sr.ToolEvent) -> bool:
    path = edit_path(event)
    if path:
        return bool(sr.TEST_PATH.search(path))
    if event.mutates:
        for part in re.split(r"&&|\|\||;", sr.strip_null(event.command)):
            if sr.TEST_PATH.search(part) and sr.MUTATION.search(part):
                return True
    return False


def is_source_edit(event: sr.ToolEvent) -> bool:
    return (event.name == "edit" or event.mutates) and not mutates_test_file(event)


def last_test(diag: sr.Diagnosis):
    return diag.test_events[-1] if diag.test_events else None


def base_row(pool: str, record: dict, admission: dict, decision: str, reasons: list[str],
             **extra) -> dict:
    row = {
        "sample_id": f"{pool}:{record['instance_id']}",
        "pool": pool,
        "instance_id": record["instance_id"],
        "family": admission.get("family", ""),
        "batch": record.get("_batch", ""),
        "decision": decision,
        "reasons": reasons,
        "training_eligible": decision == "ADMIT",
        "review_required": False,
        "keep_msgs": [],
        "masked_msgs": [],
        "evidence": {},
        "seq_tokens": None,
        "supervised_tokens": None,
        "supervised_fraction": None,
        "length_gate": MAX_TOKENS,
    }
    row.update(extra)
    return row


# --- pool rules ------------------------------------------------------------

def evaluate_c1(record: dict, admission: dict, diag: sr.Diagnosis) -> dict | None:
    if not admission.get("admitted"):
        return None
    if diag.first_mut_step is None:
        return base_row("C1", record, admission, "REJECT", ["NO_MUTATION"])
    if diag.first_mut_step > C1_STEP:
        return base_row("C1", record, admission, "REJECT", ["EDIT_TOO_LATE"],
                        evidence={"first_mut_step": diag.first_mut_step})
    if not diag.first_edit_grounded:
        return base_row("C1", record, admission, "REJECT", ["UNGROUNDED_EDIT"],
                        evidence={"first_mut_step": diag.first_mut_step,
                                  "patch_files": diag.patch.source_files})
    keep = [m for m in assistant_msgs(diag.messages) if m not in diag.duplicate_msg]
    masked = [[msg, code] for msg, code in sorted(diag.duplicate_msg.items())]
    return base_row(
        "C1", record, admission, "ADMIT",
        ["GROUNDED_LANDING", f"STEP_{diag.first_mut_step}"]
        + (["STRONG_LANDING"] if diag.first_mut_step <= C1_STRONG_STEP else []),
        keep_msgs=keep, masked_msgs=masked,
        evidence={"first_mut_step": diag.first_mut_step,
                  "grounded_files": sr.grounded_files(diag.evs, diag.patch.source_files,
                                                      diag.first_mut_msg or 0),
                  "patch_files": diag.patch.source_files,
                  "content_lines": diag.patch.content_lines,
                  "steps": (record.get("stats") or {}).get("steps")})


def evaluate_c2(record: dict, admission: dict, diag: sr.Diagnosis) -> dict | None:
    if not admission.get("admitted"):
        return None
    if not diag.test_events:
        return base_row("C2", record, admission, "REJECT", ["NO_PARSABLE_PASS"])
    last_test = diag.test_events[-1]
    if last_test.test.state != "pass" or last_test.obs_msg is None:
        return base_row("C2", record, admission, "REJECT", ["LAST_TEST_NOT_PASS"],
                        evidence={"last_test": last_test.test.as_dict()})
    if diag.last_mut_msg is None:
        return base_row("C2", record, admission, "REJECT", ["NO_MUTATION"])
    if last_test.obs_msg < diag.last_mut_msg:
        # the last edit came after the last test run, so nothing verifies the final state
        return base_row("C2", record, admission, "REJECT", ["EDIT_AFTER_LAST_TEST"],
                        evidence={"last_mut_msg": diag.last_mut_msg,
                                  "last_test_obs": last_test.obs_msg})
    if TEST_EVASION.search(last_test.command):
        # A green summary produced by deselecting/ignoring tests is not a verdict on the
        # task: the model review caught two of these by hand (coloraide, drf-pydantic), so
        # the deterministic layer now routes them to review instead of admitting them.
        return base_row("C2", record, admission, "REVIEW", ["TEST_SELECTION"],
                        review_required=True,
                        evidence={"last_pass_msg": last_test.obs_msg,
                                  "command": last_test.command[:400],
                                  "test": last_test.test.as_dict()})
    # the same check, unchanged, run twice after the last edit is the loop pathology
    redundant = redundant_reruns(diag)
    if redundant:
        return base_row("C2", record, admission, "REJECT", ["REDUNDANT_RECHECK"],
                        evidence={"redundant_reruns": redundant})
    if diag.final_msg is None or diag.final_msg < last_test.obs_msg:
        return base_row("C2", record, admission, "REJECT", ["NO_FINAL_REPORT"])
    churn = [msg for msg in diag.duplicate_msg if msg > last_test.obs_msg]
    if churn:
        return base_row("C2", record, admission, "REJECT", ["CHURN_AFTER_VERIFY"],
                        evidence={"churn_msgs": churn})
    keep = [i for i in assistant_msgs(diag.messages)
            if i <= last_test.obs_msg and i not in diag.duplicate_msg]
    if diag.final_msg not in keep:
        keep.append(diag.final_msg)
    masked = [[msg, code] for msg, code in sorted(diag.duplicate_msg.items())]
    return base_row(
        "C2", record, admission, "ADMIT", ["VERIFIED_THEN_STOPPED"],
        keep_msgs=keep, masked_msgs=masked,
        evidence={"last_mut_msg": diag.last_mut_msg, "last_pass_msg": last_test.obs_msg,
                  "pass_summary": last_test.test.as_dict(),
                  "supervised_sequences": len(keep),
                  "steps": (record.get("stats") or {}).get("steps")})


def redundant_reruns(diag: sr.Diagnosis) -> list[list[int]]:
    """Identical (command, observation) test runs repeated after the last mutation."""
    if diag.last_mut_msg is None:
        return []
    seen: dict[tuple, int] = {}
    redundant = []
    for event in diag.test_events:
        if event.obs_msg is None or event.obs_msg <= diag.last_mut_msg:
            continue
        key = (sr.normalize_command(event.command), sr.observation_digest(event.observation))
        if key in seen:
            redundant.append([seen[key], event.obs_msg])
        else:
            seen[key] = event.obs_msg
    return redundant


def evaluate_c3(record: dict, admission: dict, diag: sr.Diagnosis) -> dict | None:
    if admission.get("admitted") or record.get("success"):
        return None
    rejected = admission.get("rejected") or []
    family = admission.get("family", "")
    if is_split_quarantined(family, rejected):
        return base_row("C3", record, admission, "REJECT", ["SPLIT_ISOLATION_EXCLUDED"],
                        evidence={"family": family, "admission_rejected": rejected})
    if "infra_error" in rejected:
        return base_row("C3", record, admission, "REJECT", ["INFRA_INVALID"])
    if record.get("finish_reason") != "finish":
        return base_row("C3", record, admission, "REJECT", ["NOT_A_NATURAL_STOP"],
                        evidence={"finish_reason": record.get("finish_reason")})
    if diag.final_msg is None:
        return base_row("C3", record, admission, "REJECT", ["NO_REPORT"])
    if diag.final_msg != len(diag.messages) - 1:
        return base_row("C3", record, admission, "REJECT", ["REPORT_NOT_FINAL"])
    test = last_test(diag)
    if test is None or test.test.state != "fail" or not test.test.test_ran:
        return base_row("C3", record, admission, "REJECT", ["NO_VISIBLE_FAILURE_EVIDENCE"],
                        evidence={"last_test": test.test.as_dict() if test else None})
    screen = sr.honesty_screen(diag.final_text)
    evidence = {"last_test": test.test.as_dict(),
                "content_lines": diag.patch.content_lines,
                "screen": screen,
                "report_chars": len(diag.final_text)}
    if not screen["passes"]:
        return base_row("C3", record, admission, "REVIEW",
                        ["REPORT_HONESTY_NEEDS_REVIEW"], evidence=evidence)
    return base_row(
        "C3", record, admission, "ADMIT", ["HONEST_INCOMPLETE_REPORT"],
        keep_msgs=[diag.final_msg], review_required=True, evidence=evidence,
        # the keyword screen cannot prove honesty, so C3 stays out of the trainable
        # count until a semantic reviewer confirms the report
        training_eligible=False)


def evaluate_b(record: dict, admission: dict, diag: sr.Diagnosis) -> dict | None:
    if not admission.get("admitted"):
        return None
    if diag.patch.test_files:
        return base_row("B", record, admission, "REJECT", ["TEST_PATHS_EDITED"],
                        evidence={"test_files": diag.patch.test_files})
    fails = [e for e in diag.test_events if e.test.state == "fail" and e.obs_msg is not None]
    passes = [e for e in diag.test_events if e.test.state == "pass" and e.obs_msg is not None]
    if not fails:
        return None
    chosen = None
    narrowed = 0
    for fail_event in fails:
        edit_after = [e for e in diag.evs if e.msg > fail_event.msg and is_source_edit(e)]
        if not edit_after:
            continue
        for pass_event in passes:
            if pass_event.obs_msg <= fail_event.obs_msg:
                continue
            # the passing re-run must not be narrower than the run that failed: a green
            # summary produced by deselecting tests is not a correction receipt
            if pass_event.test.executed < fail_event.test.executed \
                    or pass_event.test.deselected > fail_event.test.deselected:
                narrowed += 1
                continue
            fail_obs, pass_obs = fail_event.obs_msg, pass_event.obs_msg
            window = [e for e in diag.evs if fail_obs < e.msg < pass_obs]
            if not any(is_source_edit(e) for e in window):
                continue
            keep, masked = [], []
            for event in window:
                if event.name not in ("edit", "bash", "read"):
                    continue
                if mutates_test_file(event):
                    masked.append([event.msg, "TEST_FILE_EDIT"])
                    continue
                if event.test is not None and TEST_EVASION.search(event.command):
                    masked.append([event.msg, "TEST_EVASION"])
                    continue
                if event.msg in diag.duplicate_msg:
                    masked.append([event.msg, diag.duplicate_msg[event.msg]])
                    continue
                if event.msg not in keep:
                    keep.append(event.msg)
            if diag.final_msg is not None and diag.final_msg > pass_obs \
                    and diag.final_msg == len(diag.messages) - 1 and diag.final_msg not in keep:
                keep.append(diag.final_msg)
            if not keep:
                continue
            candidate = {"fail_obs": fail_obs, "pass_obs": pass_obs, "keep": keep,
                         "masked": masked, "score": len(keep), "pass_after": pass_obs,
                         "fail_event": fail_event, "pass_event": pass_event}
            if chosen is None or candidate["score"] > chosen["score"] or (
                    candidate["score"] == chosen["score"]
                    and candidate["pass_after"] > chosen["pass_after"]):
                chosen = candidate
            break                      # first qualifying re-run answers this failure
    if chosen is None:
        reasons = ["NO_KEEPABLE_CORRECTION"]
        if narrowed:
            reasons.append("VERIFICATION_NARROWED")
        return base_row("B", record, admission, "REVIEW", reasons,
                        evidence={"failing_runs": len(fails), "narrowed_receipts": narrowed,
                                  "recovery_pairs": len(diag.failure_then_pass)})
    fail_event = chosen["fail_event"]
    pass_event = chosen["pass_event"]
    f2p = admission.get("f2p") or {}
    p2p = admission.get("p2p") or {}
    return base_row(
        "B", record, admission, "ADMIT", ["REAL_FAILURE_THEN_CORRECTION", "INDEPENDENT_ACCEPTANCE"],
        keep_msgs=sorted(chosen["keep"]), masked_msgs=sorted(chosen["masked"]),
        evidence={
            "fail_state": {"obs_msg": chosen["fail_obs"], "command": fail_event.command[:400],
                           "evidence": fail_event.test.evidence[:400],
                           "test": fail_event.test.as_dict(),
                           "observation_sha256": sha256(fail_event.observation.encode()
                                                        ).hexdigest()[:16]},
            "pass_state": {"obs_msg": chosen["pass_obs"], "command": pass_event.command[:400],
                           "evidence": pass_event.test.evidence[:400],
                           "test": pass_event.test.as_dict(),
                           "observation_sha256": sha256(pass_event.observation.encode()
                                                        ).hexdigest()[:16]},
            "fail_obs_msg": chosen["fail_obs"],
            "pass_obs_msg": chosen["pass_obs"],
            "recovery_pairs_available": len(diag.failure_then_pass),
            "receipt": {"accepted": True, "f2p_passed": f2p.get("passed"),
                        "p2p_passed": p2p.get("passed"),
                        "node_evidence": admission.get("node_evidence"),
                        "admission_ledger": str(FORMAL / "admission_ledger.jsonl")},
        })


def evaluate_b_seeds(record: dict, admission: dict, diag: sr.Diagnosis) -> dict | None:
    """Near-miss failures that still need a real teacher continuation (counted 0)."""
    if admission.get("admitted") or record.get("success"):
        return None
    rejected = admission.get("rejected") or []
    family = admission.get("family", "")
    if is_split_quarantined(family, rejected):
        return None
    if "infra_error" in rejected:
        return None
    if diag.patch.content_lines == 0 or diag.patch.test_files:
        return None
    test = last_test(diag)
    if test is None or test.test.state != "fail" or not test.test.test_ran:
        return None
    return {
        "seed_id": f"BSEED:{record['instance_id']}",
        "instance_id": record["instance_id"],
        "family": admission.get("family", ""),
        "batch": record.get("_batch", ""),
        "status": "SEED_AWAITING_TEACHER_CONTINUATION",
        "training_eligible": 0,
        "reasons": admission.get("rejected") or [],
        "failure_state": {"last_test_obs_msg": test.obs_msg,
                          "command": test.command[:400],
                          "evidence": test.test.evidence[:400],
                          "content_lines": diag.patch.content_lines,
                          "patch_files": diag.patch.source_files},
        "note": "Needs container replay + teacher action loop + independent judge before it can count as B.",
    }


# --- build / write ---------------------------------------------------------

def payload_path(pool: str) -> Path:
    return OUT / f"pool_{pool.lower()}.jsonl"


def render_row(row: dict, record: dict, renderer: Renderer, cache: dict) -> dict:
    """Attach ids/mask to an ADMIT row, enforcing the length and span invariants."""
    rendered = cache.get(record["instance_id"])
    if rendered is None:
        rendered = Rendered(renderer, record["messages"])
        cache[record["instance_id"]] = rendered
    keep = [int(msg) for msg in row["keep_msgs"]]
    mask = rendered.mask(keep)
    spans = rendered.spans(mask)
    row["seq_tokens"] = len(rendered.ids)
    row["supervised_tokens"] = sum(mask)
    row["supervised_fraction"] = round(sum(mask) / len(mask), 4) if mask else 0.0
    row["spans_ok"] = 1 if set(spans) == set(keep) else 0
    if len(rendered.ids) > MAX_TOKENS:
        row["decision"] = "REJECT"
        row["training_eligible"] = False
        row["reasons"] = list(row["reasons"]) + ["OVER_LENGTH"]
        return {"row": row, "payload": None}
    if sum(mask) == 0:
        row["decision"] = "REJECT"
        row["training_eligible"] = False
        row["reasons"] = list(row["reasons"]) + ["EMPTY_MASK"]
        return {"row": row, "payload": None}
    payload = {
        "sample_id": row["sample_id"],
        "pool": row["pool"],
        "instance_id": row["instance_id"],
        "ids": rendered.ids,
        "mask": mask,
        "spans": {str(msg): [first, last] for msg, (first, last) in sorted(spans.items())},
        "keep_msgs": keep,
    }
    return {"row": row, "payload": payload}


def export(mask_overlay: Path | None = None) -> dict:
    records = load_records()
    admission = load_admission()
    renderer = Renderer()
    overlay = load_overlay(mask_overlay) if mask_overlay is not None else None
    if mask_overlay is not None and overlay is None:
        raise SystemExit(f"overlay not found: {mask_overlay}")
    cache: dict = {}
    rows, payloads, seeds = [], {pool: [] for pool in POOLS}, []
    rule_rows = []
    a_rows = []
    diagnostics: dict[str, sr.Diagnosis] = {}

    for record in records:
        iid = record["instance_id"]
        adm = admission.get(iid) or {}
        try:
            diag = sr.diagnose(record)
        except Exception as error:  # a malformed trajectory must not kill the export
            rows.append(base_row("C3", record, adm, "REJECT", ["DIAGNOSE_ERROR"],
                                 evidence={"error": repr(error)[:300]}))
            continue
        diagnostics[iid] = diag
        if adm.get("admitted"):
            a_rows.append(base_row("A", record, adm, "ADMIT", ["A_VERIFIED_SOURCE"],
                                   keep_msgs=assistant_msgs(diag.messages), masked_msgs=[],
                                   evidence={"content_lines": diag.patch.content_lines}))
        for pool, rule in (("C1", evaluate_c1), ("C2", evaluate_c2),
                           ("C3", evaluate_c3), ("B", evaluate_b)):
            row = rule(record, adm, diag)
            if row is None:
                continue
            if row["decision"] == "ADMIT":
                result = render_row(row, record, renderer, cache)
                row = result["row"]
            rule_rows.append(dict(row))
            if row["decision"] == "ADMIT" and overlay is not None:
                row = apply_overlay_row(row, overlay)
            if row["decision"] == "ADMIT":
                result = render_row(row, record, renderer, cache)
                row = result["row"]
                if result["payload"] is not None:
                    payloads[pool].append(result["payload"])
            rows.append(row)
        seed = evaluate_b_seeds(record, adm, diag)
        if seed is not None:
            seeds.append(seed)

    if overlay is not None and overlay.get("receipt", {}).get("complete_review_required"):
        receipt = overlay["receipt"]
        for name, values in (("bc", rule_rows), ("a", a_rows)):
            digest = sha256("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values).encode()).hexdigest()
            if receipt.get(f"{name}_rule_ledger_sha256") != digest:
                raise SystemExit(f"{name} rule ledger changed after model review; rebuild review snapshot")
        label_path = Path(receipt["labels"])
        if sha256(label_path.read_bytes()).hexdigest() != receipt["labels_sha256"]:
            raise SystemExit("model labels changed after overlay composition")
    OUT.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hashes = {}
    for pool in POOLS:
        path = payload_path(pool)
        with path.open("w", encoding="utf-8") as handle:
            for payload in payloads[pool]:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        hashes[pool] = sha256(path.read_bytes()).hexdigest() if path.exists() else None
    with (OUT / "a_rule_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in a_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "bc_rule_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in rule_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "bc_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "created_at": created}, ensure_ascii=False) + "\n")
    with (OUT / "b_teacher_seeds.jsonl").open("w", encoding="utf-8") as handle:
        for seed in seeds:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")

    summary = summarize(rows, payloads, seeds, hashes, created, overlay)
    (OUT / "bc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    register(rows, {pool: payload_path(pool) for pool in POOLS}, hashes, created)
    db = sqlite3.connect(DB)
    db.execute("CREATE TABLE IF NOT EXISTS pool_rule_samples (sample_id TEXT PRIMARY KEY, pool TEXT, decision TEXT, row_json TEXT)")
    db.execute("DELETE FROM pool_rule_samples")
    db.executemany("INSERT INTO pool_rule_samples VALUES (?,?,?,?)",
                   [(r["sample_id"], r["pool"], r["decision"], json.dumps(r, ensure_ascii=False)) for r in rule_rows + a_rows])
    db.commit()
    db.close()
    return summary


def summarize(rows, payloads, seeds, hashes, created, overlay: dict | None = None) -> dict:
    by_pool = {}
    for pool in POOLS:
        pool_rows = [r for r in rows if r["pool"] == pool]
        decisions = Counter(r["decision"] for r in pool_rows)
        reasons = Counter(code for r in pool_rows for code in r["reasons"])
        admitted = [r for r in pool_rows if r["decision"] == "ADMIT"]
        by_pool[pool] = {
            "candidates": len(pool_rows),
            "decisions": dict(decisions),
            "reasons": dict(reasons),
            "admitted": len(admitted),
            "sequences": len(payloads[pool]),
            "seq_tokens": sum(r["seq_tokens"] or 0 for r in admitted),
            "supervised_tokens": sum(r["supervised_tokens"] or 0 for r in admitted),
            "model_masked_messages": sum(r.get("model_masked") or 0 for r in pool_rows),
            "payload_sha256": hashes.get(pool),
        }
    result = {"created_at": created, "pools": by_pool,
              "b_teacher_seeds": len(seeds),
              "ledger": str(OUT / "bc_ledger.jsonl"),
              "index": str(DB)}
    if overlay is not None:
        result["model_overlay"] = {
            "path": overlay["path"], "sha256": overlay["sha256"],
            "quarantine": len(overlay["quarantine"]),
            "masked_samples": len(overlay["masks"]),
            "masked_messages": sum(len(msgs) for msgs in overlay["masks"].values()),
        }
    return result


def register(rows, paths, hashes, created) -> None:
    db = sqlite3.connect(DB)
    db.executescript("DROP TABLE IF EXISTS pool_samples;")
    db.executescript(SCHEMA)
    for row in rows:
        pool = row["pool"]
        db.execute(
            "INSERT OR REPLACE INTO pool_samples VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["sample_id"], pool, row["instance_id"], row["family"], row["batch"],
             row["decision"], int(row["training_eligible"]), int(row["review_required"]),
             json.dumps(row["reasons"], ensure_ascii=False),
             json.dumps(row["keep_msgs"]), json.dumps(row["masked_msgs"], ensure_ascii=False),
             json.dumps(row["evidence"], ensure_ascii=False),
             row["seq_tokens"], row["supervised_tokens"], row["supervised_fraction"],
             row["length_gate"], row.get("spans_ok"),
             str(paths[pool]) if row["decision"] == "ADMIT" else None,
             hashes.get(pool) if row["decision"] == "ADMIT" else None,
             created))
    db.commit()
    db.close()


# --- verification ----------------------------------------------------------

def verify_payload(row: dict, payload: dict) -> list[str]:
    problems = []
    if len(payload["ids"]) != len(payload["mask"]):
        problems.append("ids/mask length mismatch")
    if len(payload["ids"]) != row["seq_tokens"]:
        problems.append(f"seq_tokens {len(payload['ids'])} != ledger {row['seq_tokens']}")
    if sum(payload["mask"]) != row["supervised_tokens"]:
        problems.append(f"supervised {sum(payload['mask'])} != ledger {row['supervised_tokens']}")
    spans = {int(msg): tuple(bounds) for msg, bounds in payload["spans"].items()}
    if set(spans) != set(row["keep_msgs"]):
        problems.append("spans do not cover exactly keep_msgs")
    covered = set()
    for first, last in spans.values():
        covered.update(range(first, last + 1))
    masked = {i for i, value in enumerate(payload["mask"]) if value}
    if masked != covered:
        problems.append(f"mask has {len(masked - covered)} stray and "
                        f"{len(covered - masked)} uncovered tokens")
    return problems


def check() -> int:
    rows = load_jsonl(OUT / "bc_ledger.jsonl")
    problems: list[str] = []
    if not rows:
        print("no ledger — run: python results/bc_export.py")
        return 1
    for pool in POOLS:
        base, expected = OUT / f"pool_{pool.lower()}.jsonl", Counter()
        payloads = {p["sample_id"]: p for p in load_jsonl(base)}
        for row in rows:
            if row["pool"] != pool or row["decision"] != "ADMIT":
                continue
            expected[row["sample_id"]] += 1
            payload = payloads.get(row["sample_id"])
            if payload is None:
                problems.append(f"{row['sample_id']}: admitted but no payload")
                continue
            for problem in verify_payload(row, payload):
                problems.append(f"{row['sample_id']}: {problem}")
        for extra in set(payloads) - set(expected):
            problems.append(f"{extra}: payload without an admitted ledger row")

    db = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        stored = {(pool, decision): n for pool, decision, n in
                  db.execute("SELECT pool, decision, COUNT(*) FROM pool_samples "
                             "GROUP BY pool, decision")}
        counted = Counter((r["pool"], r["decision"]) for r in rows)
        for key in set(stored) | set(counted):
            if stored.get(key, 0) != counted.get(key, 0):
                problems.append(f"index[{key[0]}/{key[1]}]: db={stored.get(key, 0)} "
                                f"ledger={counted.get(key, 0)}")
        for column, field in (("supervised_tokens", "supervised_tokens"),
                              ("seq_tokens", "seq_tokens")):
            got = db.execute("SELECT COALESCE(SUM(%s),0) FROM pool_samples "
                             "WHERE decision='ADMIT'" % column).fetchone()[0]
            want = sum(r[field] or 0 for r in rows if r["decision"] == "ADMIT")
            if got != want:
                problems.append(f"{column}: db={got} ledger={want}")
        eligible = db.execute("SELECT COUNT(*) FROM pool_samples WHERE training_eligible=1"
                              ).fetchone()[0]
        want_eligible = sum(1 for r in rows if r["decision"] == "ADMIT" and r["training_eligible"])
        if eligible != want_eligible:
            problems.append(f"training_eligible rows {eligible} != trainable ADMIT rows {want_eligible}")
    finally:
        db.close()

    # re-derive every decision from the raw trajectories (no tokenizer needed)
    records = load_records()
    admission = load_admission()
    summary_path = OUT / "bc_summary.json"
    overlay_meta = {}
    if summary_path.exists():
        overlay_meta = (json.loads(summary_path.read_text(encoding="utf-8"))
                        .get("model_overlay") or {})
    overlay = None
    if overlay_meta:
        overlay = load_overlay(Path(overlay_meta["path"]))
        if overlay is None:
            problems.append("summary records a model overlay that is missing")
        elif overlay["sha256"] != overlay_meta.get("sha256"):
            problems.append("model overlay changed since the export — rerun with --mask-overlay")
    derived: dict[str, tuple[str, tuple]] = {}
    for record in records:
        adm = admission.get(record["instance_id"]) or {}
        try:
            diag = sr.diagnose(record)
        except Exception:
            continue
        for rule in (evaluate_c1, evaluate_c2, evaluate_c3, evaluate_b):
            row = rule(record, adm, diag)
            if row is not None:
                derived[row["sample_id"]] = (row["decision"], tuple(row["keep_msgs"]))
    gate_rejects = {"OVER_LENGTH", "EMPTY_MASK", "MODEL_QUARANTINE", "MODEL_EMPTY_MASK"}
    for row in rows:
        sample = row["sample_id"]
        expected = derived.get(sample)
        if expected is None:
            problems.append(f"{sample}: ledger row is no longer derivable from the data")
            continue
        decision, keep = expected
        if overlay is not None and decision == "ADMIT":
            if sample in overlay["quarantine"]:
                decision = "REJECT"
            else:
                dropped = overlay["masks"].get(sample) or set()
                keep = tuple(msg for msg in keep if msg not in dropped)
                if not keep:
                    decision = "REJECT"
        if decision == "ADMIT" and ({"OVER_LENGTH", "EMPTY_MASK"} & set(row["reasons"])):
            decision = "REJECT"         # the rule admitted it; the render gate rejected it
        if row["decision"] != decision:
            problems.append(f"{sample}: decision {row['decision']} != rule {decision}")
        if tuple(row["keep_msgs"]) != keep:
            problems.append(f"{sample}: keep_msgs drifted from the rule")
        elif gate_rejects & set(row["reasons"]) and row["decision"] != "REJECT":
            problems.append(f"{sample}: gate reason without a REJECT decision")
    for sample in set(derived) - {row["sample_id"] for row in rows}:
        problems.append(f"{sample}: rule produces a row that the ledger lost")

    if problems:
        print("POOL INDEX OUT OF DATE — rerun: python results/bc_export.py")
        for problem in problems[:40]:
            print("  -", problem)
        return 1
    counts = Counter((r["pool"], r["decision"]) for r in rows)
    print("OK: pool ledgers, payloads and index agree")
    for pool in POOLS:
        print(f"  {pool}: " + ", ".join(f"{decision}={counts.get((pool, decision), 0)}"
                                        for decision in ("ADMIT", "REVIEW", "REJECT")))
    print("  supervised tokens:",
          sum(r["supervised_tokens"] or 0 for r in rows if r["decision"] == "ADMIT"))
    return 0


def summary() -> None:
    rows = load_jsonl(OUT / "bc_ledger.jsonl")
    if not rows:
        print("no ledger — run: python results/bc_export.py")
        return
    for pool in POOLS:
        pool_rows = [r for r in rows if r["pool"] == pool]
        admitted = [r for r in pool_rows if r["decision"] == "ADMIT"]
        print(f"\n== {pool}: {len(admitted)} admitted / {len(pool_rows)} candidates")
        for code, n in Counter(code for r in pool_rows for code in r["reasons"]).most_common():
            print(f"   {code:32} {n}")
        if admitted:
            tokens = [r["seq_tokens"] for r in admitted]
            supervised = [r["supervised_tokens"] for r in admitted]
            masked = sum(r.get("model_masked") or 0 for r in pool_rows)
            print(f"   tokens: total={sum(tokens)}, supervised={sum(supervised)}, "
                  f"max_len={max(tokens)}, mean_len={sum(tokens)//len(tokens)}, "
                  f"model_masked_msgs={masked}")
            print(f"   supervised fraction mean="
                  f"{sum(supervised)/sum(tokens):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pool B and pool C from recorded runs")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--mask-overlay", type=Path, default=None,
                        help="model_review.py overlay.json: quarantine samples, mask messages")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check())
    if args.summary:
        summary()
        return
    print(json.dumps(export(args.mask_overlay), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
