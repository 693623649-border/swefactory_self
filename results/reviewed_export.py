"""Export and manifest release for reviewed Pool A and combined training pools.

Bridges the model-review overlay with verified Pool A:
  * Reads the admission ledger for Pool A (a_rule_ledger.jsonl)
  * Validates completeness and freshness of review/overlay.json
  * Renders token payloads (ids, mask, spans) for eligible A trajectories
  * Excludes poisoned trajectories (QUARANTINE) and masks poisoned messages (MASK)
  * Emits pool_bc/pool_a_reviewed.jsonl, pool_bc/a_reviewed_ledger.jsonl,
    and pool_bc/release_manifest.json
  * Registers metadata into index.sqlite3 under pool_reviewed_a and pool_training_release

Usage:
    python results/reviewed_export.py                  # export and register
    python results/reviewed_export.py --check          # verify hashes, counts, masks, and db
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

RESULTS = Path(__file__).resolve().parent
FORMAL = RESULTS / "pool_a_formal"
POOL = RESULTS / "pool_bc"
DB = FORMAL / "index.sqlite3"

DEFAULT_OVERLAY = POOL / "review" / "overlay.json"
DEFAULT_A_RULE_LEDGER = POOL / "a_rule_ledger.jsonl"
DEFAULT_BC_RULE_LEDGER = POOL / "bc_rule_ledger.jsonl"
DEFAULT_BC_LEDGER = POOL / "bc_ledger.jsonl"
DEFAULT_BASELINE_A = FORMAL / "sft_admitted.jsonl"
DEFAULT_A_REVIEWED_PAYLOAD = POOL / "pool_a_reviewed.jsonl"
DEFAULT_A_REVIEWED_LEDGER = POOL / "a_reviewed_ledger.jsonl"
DEFAULT_MANIFEST = POOL / "release_manifest.json"

MAX_TOKENS = 65536
ALL_POOLS = ("A", "B", "C1", "C2", "C3")

sys.path.insert(0, str(RESULTS))

SCHEMA = """
CREATE TABLE IF NOT EXISTS pool_reviewed_a (
    sample_id           TEXT PRIMARY KEY,
    instance_id         TEXT NOT NULL,
    family              TEXT,
    batch               TEXT,
    decision            TEXT NOT NULL,          -- ADMIT | REJECT
    training_eligible   INTEGER NOT NULL,       -- 1 if ADMIT else 0
    reasons             TEXT,                   -- JSON array of reason codes
    keep_msgs           TEXT,                   -- JSON array of supervised message indices
    masked_msgs         TEXT,                   -- JSON array of [msg, reason]
    evidence            TEXT,                   -- JSON object
    seq_tokens          INTEGER,
    supervised_tokens   INTEGER,
    supervised_fraction REAL,
    length_gate         INTEGER,
    spans_ok            INTEGER,
    payload_path        TEXT,
    payload_sha256      TEXT,
    created_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_pool_reviewed_a_decision
    ON pool_reviewed_a(decision, training_eligible);
CREATE INDEX IF NOT EXISTS idx_pool_reviewed_a_instance
    ON pool_reviewed_a(instance_id);

CREATE TABLE IF NOT EXISTS pool_training_release (
    pool                TEXT PRIMARY KEY,
    trainable           INTEGER NOT NULL,
    file_path           TEXT NOT NULL,
    file_sha256         TEXT NOT NULL,
    raw_payload_count   INTEGER NOT NULL,
    eligible_count      INTEGER NOT NULL,
    total_seq_tokens    INTEGER NOT NULL,
    total_sup_tokens    INTEGER NOT NULL,
    metadata_json       TEXT,
    created_at          TEXT NOT NULL
);
"""


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file without splitting on unicode separators like U+2028."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256(path.read_bytes()).hexdigest()


def validate_overlay(
    overlay_path: Path,
    a_rule_path: Path,
    bc_rule_path: Path | None = None,
) -> dict:
    """Validate overlay completeness and reject stale overlays."""
    import bc_export
    if not overlay_path.exists():
        raise SystemExit(f"overlay not found: {overlay_path}")
    raw = json.loads(overlay_path.read_text(encoding="utf-8"))

    if not raw.get("complete_review_required"):
        raise SystemExit(f"incomplete overlay: complete_review_required is not set ({overlay_path})")

    labels_file = raw.get("labels")
    if not labels_file:
        raise SystemExit(f"overlay missing labels pointer: {overlay_path}")
    labels_path = Path(labels_file)
    if not labels_path.exists():
        raise SystemExit(f"overlay labels file not found: {labels_path}")
    cur_labels_sha = sha256(labels_path.read_bytes()).hexdigest()
    if cur_labels_sha != raw.get("labels_sha256"):
        raise SystemExit("model labels changed after overlay composition (stale labels_sha256)")

    if not a_rule_path.exists():
        raise SystemExit(f"a_rule_ledger not found: {a_rule_path}")
    a_rows = [json.loads(line) for line in a_rule_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    cur_a_sha = sha256(a_rule_path.read_bytes()).hexdigest()
    stable_a_sha = bc_export.rule_digest(a_rows)
    expected_a_sha = raw.get("a_rule_ledger_sha256")
    if not expected_a_sha:
        raise SystemExit(f"overlay missing required a_rule_ledger_sha256: {overlay_path}")
    if expected_a_sha not in {cur_a_sha, stable_a_sha}:
        raise SystemExit("a rule ledger changed after model review (stale a_rule_ledger_sha256)")

    if bc_rule_path is not None:
        if not bc_rule_path.exists():
            raise SystemExit(f"bc_rule_ledger not found: {bc_rule_path}")
        bc_rows = [json.loads(line) for line in bc_rule_path.read_text(encoding="utf-8").split("\n") if line.strip()]
        cur_bc_sha = sha256(bc_rule_path.read_bytes()).hexdigest()
        stable_bc_sha = bc_export.rule_digest(bc_rows)
        expected_bc_sha = raw.get("bc_rule_ledger_sha256") or raw.get("rule_ledger_sha256")
        if not expected_bc_sha:
            raise SystemExit(f"overlay missing required bc_rule_ledger_sha256: {overlay_path}")
        if expected_bc_sha not in {cur_bc_sha, stable_bc_sha}:
            raise SystemExit("bc rule ledger changed after model review (stale bc_rule_ledger_sha256)")

    quarantine = {item["sample_id"] for item in raw.get("excluded_by_model") or []}
    source_quarantine = set(raw.get("source_quarantine") or [])
    masks = {sample: {int(msg) for msg in msgs}
             for sample, msgs in (raw.get("sample_masks") or {}).items()}
    source_masks = {iid: {int(msg) for msg in msgs}
                    for iid, msgs in (raw.get("source_masks") or {}).items()}

    return {
        "path": str(overlay_path),
        "sha256": sha256(overlay_path.read_bytes()).hexdigest(),
        "quarantine": quarantine,
        "source_quarantine": source_quarantine,
        "masks": masks,
        "source_masks": source_masks,
        "receipt": raw,
    }


def apply_overlay_to_a_row(row: dict, overlay: dict) -> dict:
    """Apply sample quarantine and message masking from overlay to an A-pool row."""
    sample = row["sample_id"]
    iid = row.get("instance_id") or sample.split(":", 1)[-1]

    # Check sample-level quarantine (either sample_id or cross-view instance_id)
    if sample in overlay["quarantine"] or iid in overlay.get("source_quarantine", set()):
        return {
            **row,
            "decision": "REJECT",
            "training_eligible": False,
            "reasons": list(row.get("reasons") or []) + ["MODEL_QUARANTINE"],
            "reasons_by_model": ["QUARANTINE"],
        }

    # Collect dropped messages (both sample-specific and cross-view source_masks)
    dropped = set(overlay["masks"].get(sample) or set())
    dropped |= set(overlay.get("source_masks", {}).get(iid) or set())

    keep = [int(msg) for msg in row.get("keep_msgs") or []]
    kept = [msg for msg in keep if msg not in dropped]

    if not dropped:
        return row

    new_row = {
        **row,
        "keep_msgs": kept,
        "model_masked": len(keep) - len(kept),
        "masked_msgs": list(row.get("masked_msgs") or []) +
                       [[msg, "MODEL_REVIEW"] for msg in keep if msg in dropped],
        "reasons_by_model": ["MASK"],
    }
    if not kept:
        new_row.update({
            "decision": "REJECT",
            "training_eligible": False,
            "reasons": list(row.get("reasons") or []) + ["MODEL_EMPTY_MASK"],
        })
    return new_row


def _render_trajectory(renderer, messages: list[dict]):
    """Produce a Rendered-compatible object (having .ids, .mask, .spans) from renderer and messages.

    Compatible with:
      - Real to_sft_qwen.Renderer (returns str from render(); needs to_sft_qwen.Rendered)
      - Fake test mock renderers (whose render() directly returns a Rendered-like instance with ids and mask)
    """
    if hasattr(renderer, "prefix_text") and hasattr(renderer, "tokenizer"):
        from to_sft_qwen import Rendered
        return Rendered(renderer, messages)
    if hasattr(renderer, "render"):
        res = renderer.render(messages)
        if hasattr(res, "ids") and hasattr(res, "mask"):
            return res
    from to_sft_qwen import Rendered
    return Rendered(renderer, messages)


def render_a_row(row: dict, record: dict, renderer, cache: dict | None = None) -> dict:
    """Render token ids and mask for an ADMIT A-pool row, checking length and mask validity."""
    if cache is None:
        cache = {}
    iid = record["instance_id"]
    rendered = cache.get(iid)
    if rendered is None:
        rendered = _render_trajectory(renderer, record["messages"])
        cache[iid] = rendered

    keep = [int(msg) for msg in row["keep_msgs"]]
    # Reject stale rule ledgers rather than producing ids/mask that contradict them.
    from segment_review import tool_error_messages
    invalid = set(keep) & tool_error_messages(record["messages"]).keys()
    if invalid:
        raise ValueError(f"{iid}: supervised tool errors at {sorted(invalid)}; rebuild rule ledger")
    mask = rendered.mask(keep)
    spans = rendered.spans(mask)

    row["seq_tokens"] = len(rendered.ids)
    row["supervised_tokens"] = sum(mask)
    row["supervised_fraction"] = round(sum(mask) / len(mask), 4) if mask else 0.0
    row["spans_ok"] = 1 if set(spans) == set(keep) else 0

    if len(rendered.ids) > MAX_TOKENS:
        row["decision"] = "REJECT"
        row["training_eligible"] = False
        row["reasons"] = list(row.get("reasons") or []) + ["OVER_LENGTH"]
        return {"row": row, "payload": None}

    if sum(mask) == 0:
        row["decision"] = "REJECT"
        row["training_eligible"] = False
        row["reasons"] = list(row.get("reasons") or []) + ["EMPTY_MASK"]
        return {"row": row, "payload": None}

    # Strict payload: no hidden judge, evaluation results, or gold patches
    payload = {
        "sample_id": row["sample_id"],
        "pool": "A",
        "instance_id": row["instance_id"],
        "ids": rendered.ids,
        "mask": mask,
        "spans": {str(msg): [first, last] for msg, (first, last) in sorted(spans.items())},
        "keep_msgs": keep,
        "training_eligible": True,
    }
    return {"row": row, "payload": payload}


def build_manifest(
    output_dir: Path,
    reviewed_a_payload_path: Path,
    a_rows: list[dict],
    bc_ledger_path: Path,
    baseline_a_path: Path,
    created_at: str,
) -> dict:
    """Construct release manifest covering audited baseline, reviewed A/B/C pools, and view groups."""
    baseline_sha = file_sha256(baseline_a_path)
    baseline_rows = load_jsonl(baseline_a_path) if baseline_a_path.exists() else []

    training_pools = {}

    # Pool A (reviewed)
    a_admitted = [r for r in a_rows if r["decision"] == "ADMIT" and r.get("training_eligible") is True]
    a_sha = file_sha256(reviewed_a_payload_path)
    training_pools["A"] = {
        "pool": "A",
        "trainable": True,
        "path": str(reviewed_a_payload_path.resolve()),
        "sha256": a_sha,
        "raw_payload_count": len(a_admitted),
        "eligible_count": len(a_admitted),
        "seq_tokens": sum(r["seq_tokens"] or 0 for r in a_admitted),
        "supervised_tokens": sum(r["supervised_tokens"] or 0 for r in a_admitted),
    }

    # Pools B, C1, C2, C3 from output_dir and bc_ledger
    bc_rows = load_jsonl(bc_ledger_path) if bc_ledger_path.exists() else []
    by_pool_bc = defaultdict(list)
    for r in bc_rows:
        by_pool_bc[r["pool"]].append(r)

    for p in ("B", "C1", "C2", "C3"):
        p_path = output_dir / f"pool_{p.lower()}.jsonl"
        p_sha = file_sha256(p_path)
        payloads = load_jsonl(p_path) if p_path.exists() else []
        ledger_entries = by_pool_bc[p]

        if p == "C3":
            # C3 is eligible strictly if training_eligible == 1 / True in bc_ledger
            eligible_entries = [r for r in ledger_entries if r.get("decision") == "ADMIT" and r.get("training_eligible") is True]
            seq_tokens = sum(r["seq_tokens"] or 0 for r in eligible_entries)
            supervised_tokens = sum(r["supervised_tokens"] or 0 for r in eligible_entries)
            training_pools[p] = {
                "pool": p,
                "trainable": True,
                "path": str(p_path.resolve()) if p_path.exists() else None,
                "sha256": p_sha,
                "raw_payload_count": len(payloads),
                "eligible_count": len(eligible_entries),
                "seq_tokens": seq_tokens,
                "supervised_tokens": supervised_tokens,
                "note": "C3 filtered strictly by final bc_ledger.training_eligible; seeds are excluded.",
            }
        else:
            admitted = [r for r in ledger_entries if r.get("decision") == "ADMIT" and r.get("training_eligible") is True]
            seq_tokens = sum(r["seq_tokens"] or 0 for r in admitted)
            supervised_tokens = sum(r["supervised_tokens"] or 0 for r in admitted)
            training_pools[p] = {
                "pool": p,
                "trainable": True,
                "path": str(p_path.resolve()) if p_path.exists() else None,
                "sha256": p_sha,
                "raw_payload_count": len(payloads),
                "eligible_count": len(admitted),
                "seq_tokens": seq_tokens,
                "supervised_tokens": supervised_tokens,
            }

    # B seeds audit entry (seeds are untrainable)
    b_seeds_path = output_dir / "b_teacher_seeds.jsonl"
    seeds_count = len(load_jsonl(b_seeds_path)) if b_seeds_path.exists() else 0
    seeds_audit = {
        "pool": "B_SEEDS",
        "trainable": False,
        "path": str(b_seeds_path.resolve()) if b_seeds_path.exists() else None,
        "sha256": file_sha256(b_seeds_path),
        "count": seeds_count,
        "eligible_count": 0,
        "note": "Teacher continuation seeds are untrainable near-misses.",
    }

    # View groups across eligible pools
    view_groups = defaultdict(list)
    # Reviewed A
    for r in a_admitted:
        view_groups[r["instance_id"]].append({
            "pool": "A",
            "sample_id": r["sample_id"],
            "seq_tokens": r["seq_tokens"],
            "supervised_tokens": r["supervised_tokens"],
        })
    # B, C1, C2, C3
    for p in ("B", "C1", "C2", "C3"):
        for r in by_pool_bc[p]:
            if r.get("decision") == "ADMIT" and r.get("training_eligible") is True:
                view_groups[r["instance_id"]].append({
                    "pool": p,
                    "sample_id": r["sample_id"],
                    "seq_tokens": r.get("seq_tokens"),
                    "supervised_tokens": r.get("supervised_tokens"),
                })

    manifest = {
        "manifest_version": "v1-reviewed-release",
        "created_at": created_at,
        "notes": (
            "Multi-view groups must not be sampled as independent tasks. "
            "No custom mixture ratios or duplicate corpus added. "
            "C3 eligible samples are strictly governed by semantic review."
        ),
        "audit_baseline": {
            "pool": "A_BASELINE",
            "trainable": False,
            "description": "Original rule-based Pool A baseline for audit only, not for training",
            "path": str(baseline_a_path.resolve()) if baseline_a_path.exists() else None,
            "sha256": baseline_sha,
            "count": len(baseline_rows),
        },
        "training_pools": training_pools,
        "seeds_audit": seeds_audit,
        "summary": {
            "total_trainable_pools": len(training_pools),
            "total_eligible_samples": sum(v["eligible_count"] for v in training_pools.values()),
            "total_seq_tokens": sum(v["seq_tokens"] for v in training_pools.values()),
            "total_supervised_tokens": sum(v["supervised_tokens"] for v in training_pools.values()),
            "unique_instances": len(view_groups),
        },
        "view_groups": dict(sorted(view_groups.items())),
    }
    return manifest


def register_to_sqlite(
    db_path: Path,
    a_rows: list[dict],
    manifest: dict,
    payload_path: Path,
    payload_sha: str | None,
    created_at: str,
) -> None:
    """Register reviewed pool A and release manifest into SQLite without dropping other tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    try:
        db.executescript(SCHEMA)

        # 1. Update pool_reviewed_a
        db.execute("DELETE FROM pool_reviewed_a")
        for row in a_rows:
            is_admit = row["decision"] == "ADMIT" and row.get("training_eligible") is True
            db.execute(
                "INSERT INTO pool_reviewed_a VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["sample_id"],
                    row["instance_id"],
                    row.get("family", ""),
                    row.get("batch", ""),
                    row["decision"],
                    1 if is_admit else 0,
                    json.dumps(row.get("reasons") or [], ensure_ascii=False),
                    json.dumps(row.get("keep_msgs") or []),
                    json.dumps(row.get("masked_msgs") or [], ensure_ascii=False),
                    json.dumps(row.get("evidence") or {}, ensure_ascii=False),
                    row.get("seq_tokens"),
                    row.get("supervised_tokens"),
                    row.get("supervised_fraction"),
                    row.get("length_gate", MAX_TOKENS),
                    row.get("spans_ok", 0),
                    str(payload_path) if is_admit else None,
                    payload_sha if is_admit else None,
                    created_at,
                ),
            )

        # 2. Update pool_training_release
        db.execute("DELETE FROM pool_training_release")
        for p, info in manifest["training_pools"].items():
            db.execute(
                "INSERT INTO pool_training_release VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    p,
                    1 if info.get("trainable") else 0,
                    str(info.get("path") or ""),
                    str(info.get("sha256") or ""),
                    int(info.get("raw_payload_count") or 0),
                    int(info.get("eligible_count") or 0),
                    int(info.get("seq_tokens") or 0),
                    int(info.get("supervised_tokens") or 0),
                    json.dumps({k: v for k, v in info.items() if k not in (
                        "pool", "trainable", "path", "sha256", "raw_payload_count",
                        "eligible_count", "seq_tokens", "supervised_tokens"
                    )}, ensure_ascii=False),
                    created_at,
                ),
            )
        # Also register audit baseline
        base_info = manifest["audit_baseline"]
        db.execute(
            "INSERT INTO pool_training_release VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "A_BASELINE_AUDIT",
                0,
                str(base_info.get("path") or ""),
                str(base_info.get("sha256") or ""),
                int(base_info.get("count") or 0),
                0,
                0,
                0,
                json.dumps({"description": base_info.get("description")}, ensure_ascii=False),
                created_at,
            ),
        )

        db.commit()
    finally:
        db.close()


def export(
    overlay_path: Path = DEFAULT_OVERLAY,
    a_rule_path: Path = DEFAULT_A_RULE_LEDGER,
    bc_rule_path: Path = DEFAULT_BC_RULE_LEDGER,
    bc_ledger_path: Path = DEFAULT_BC_LEDGER,
    baseline_a_path: Path = DEFAULT_BASELINE_A,
    output_dir: Path = POOL,
    db_path: Path = DB,
    renderer=None,
    records: list[dict] | None = None,
) -> dict:
    """Run full review-aware export for Pool A and construct release manifest."""
    overlay = validate_overlay(overlay_path, a_rule_path, bc_rule_path)

    if renderer is None:
        from to_sft_qwen import Renderer
        renderer = Renderer()

    if records is None:
        import bc_export
        records = bc_export.load_records()

    records_map = {r["instance_id"]: r for r in records}

    a_rule_rows = load_jsonl(a_rule_path)
    reviewed_a_rows = []
    a_payloads = []
    cache: dict = {}

    for row in a_rule_rows:
        row = dict(row)
        iid = row["instance_id"]
        record = records_map.get(iid)
        if record is None:
            row["decision"] = "REJECT"
            row["training_eligible"] = False
            row["reasons"] = list(row.get("reasons") or []) + ["TRAJECTORY_MISSING"]
            reviewed_a_rows.append(row)
            continue

        row = apply_overlay_to_a_row(row, overlay)
        if row["decision"] == "ADMIT":
            result = render_a_row(row, record, renderer, cache)
            row = result["row"]
            if result["payload"] is not None:
                a_payloads.append(result["payload"])
        else:
            row["training_eligible"] = False

        reviewed_a_rows.append(row)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Output pool_bc/pool_a_reviewed.jsonl
    a_payload_path = output_dir / "pool_a_reviewed.jsonl"
    write_jsonl(a_payload_path, a_payloads)
    a_payload_sha = file_sha256(a_payload_path)

    # 2. Output pool_bc/a_reviewed_ledger.jsonl
    ledger_rows_with_meta = []
    for r in reviewed_a_rows:
        meta = {
            **r,
            "created_at": created_at,
            "payload_path": str(a_payload_path) if r["decision"] == "ADMIT" and r.get("training_eligible") else None,
            "payload_sha256": a_payload_sha if r["decision"] == "ADMIT" and r.get("training_eligible") else None,
        }
        ledger_rows_with_meta.append(meta)

    a_reviewed_ledger_path = output_dir / "a_reviewed_ledger.jsonl"
    write_jsonl(a_reviewed_ledger_path, ledger_rows_with_meta)

    # 3. Output pool_bc/release_manifest.json
    manifest = build_manifest(
        output_dir=output_dir,
        reviewed_a_payload_path=a_payload_path,
        a_rows=reviewed_a_rows,
        bc_ledger_path=bc_ledger_path,
        baseline_a_path=baseline_a_path,
        created_at=created_at,
    )
    manifest_path = output_dir / "release_manifest.json"
    write_json(manifest_path, manifest)

    # 4. Register to index.sqlite3
    register_to_sqlite(
        db_path=db_path,
        a_rows=reviewed_a_rows,
        manifest=manifest,
        payload_path=a_payload_path,
        payload_sha=a_payload_sha,
        created_at=created_at,
    )

    summary = {
        "created_at": created_at,
        "a_rule_candidates": len(a_rule_rows),
        "a_reviewed_admitted": len(a_payloads),
        "a_payload_sha256": a_payload_sha,
        "manifest": str(manifest_path),
        "db": str(db_path),
        "training_pools": {
            p: {
                "eligible": info["eligible_count"],
                "seq_tokens": info["seq_tokens"],
                "sup_tokens": info["supervised_tokens"],
            }
            for p, info in manifest["training_pools"].items()
        },
    }
    return summary


def check(
    overlay_path: Path = DEFAULT_OVERLAY,
    a_rule_path: Path = DEFAULT_A_RULE_LEDGER,
    bc_rule_path: Path = DEFAULT_BC_RULE_LEDGER,
    output_dir: Path = POOL,
    db_path: Path = DB,
    renderer=None,
    records: list[dict] | None = None,
) -> int:
    """Verify hashes, JSONL counts, token masks, spans, sqlite key fields, and rule+overlay semantic integrity."""
    problems = []

    # 1. Overlay check
    try:
        overlay = validate_overlay(overlay_path, a_rule_path, bc_rule_path)
    except Exception as exc:
        problems.append(f"overlay validation failed: {exc}")
        overlay = None

    manifest_path = output_dir / "release_manifest.json"
    if not manifest_path.exists():
        problems.append(f"manifest file missing: {manifest_path}")
        print("CHECK FAILED:\n" + "\n".join(problems))
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 2. Check manifest referenced payloads (hashes, physical existence, counts)
    for p, info in manifest.get("training_pools", {}).items():
        path_str = info.get("path")
        if not path_str:
            problems.append(f"pool {p} path not specified in manifest")
            continue
        p_path = Path(path_str)
        if not p_path.exists():
            problems.append(f"pool {p} payload file missing: {p_path}")
            continue
        actual_sha = sha256(p_path.read_bytes()).hexdigest()
        if actual_sha != info.get("sha256"):
            problems.append(f"pool {p} hash mismatch: file={actual_sha} manifest={info.get('sha256')}")
        actual_lines = len(load_jsonl(p_path))
        if actual_lines != info.get("raw_payload_count"):
            problems.append(f"pool {p} line count mismatch: file={actual_lines} manifest={info.get('raw_payload_count')}")

    # Baseline audit path check
    baseline_info = manifest.get("audit_baseline", {})
    if baseline_info.get("path"):
        b_path = Path(baseline_info["path"])
        if b_path.exists():
            b_sha = sha256(b_path.read_bytes()).hexdigest()
            if b_sha != baseline_info.get("sha256"):
                problems.append(f"baseline A hash mismatch: file={b_sha} manifest={baseline_info.get('sha256')}")

    # 3. Check SQLite tables alignment (counts and key fields per row)
    a_ledger_path = output_dir / "a_reviewed_ledger.jsonl"
    ledger_rows = load_jsonl(a_ledger_path)
    if not db_path.exists():
        problems.append(f"SQLite db missing: {db_path}")
    else:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        try:
            # 3.1 Check pool_reviewed_a table rows and fields against a_reviewed_ledger
            db_a_rows = {r["sample_id"]: dict(r) for r in db.execute("SELECT * FROM pool_reviewed_a").fetchall()}
            if len(db_a_rows) != len(ledger_rows):
                problems.append(f"pool_reviewed_a table count {len(db_a_rows)} != ledger lines {len(ledger_rows)}")

            for row in ledger_rows:
                sid = row.get("sample_id")
                db_row = db_a_rows.get(sid)
                if not db_row:
                    problems.append(f"pool_reviewed_a missing row in DB: {sid}")
                    continue

                if db_row["decision"] != row.get("decision"):
                    problems.append(f"pool_reviewed_a {sid}: decision db='{db_row['decision']}' != ledger='{row.get('decision')}'")

                is_admit = row.get("decision") == "ADMIT" and bool(row.get("training_eligible"))
                expected_eligible = 1 if is_admit else 0
                if db_row["training_eligible"] != expected_eligible:
                    problems.append(f"pool_reviewed_a {sid}: training_eligible db={db_row['training_eligible']} != expected={expected_eligible}")

                if db_row["seq_tokens"] != row.get("seq_tokens"):
                    problems.append(f"pool_reviewed_a {sid}: seq_tokens db={db_row['seq_tokens']} != ledger={row.get('seq_tokens')}")

                if db_row["supervised_tokens"] != row.get("supervised_tokens"):
                    problems.append(f"pool_reviewed_a {sid}: supervised_tokens db={db_row['supervised_tokens']} != ledger={row.get('supervised_tokens')}")

                expected_payload_sha = row.get("payload_sha256") if is_admit else None
                if db_row["payload_sha256"] != expected_payload_sha:
                    problems.append(f"pool_reviewed_a {sid}: payload_sha256 db='{db_row['payload_sha256']}' != expected='{expected_payload_sha}'")

                db_keep = json.loads(db_row["keep_msgs"] or "[]")
                ledger_keep = [int(m) for m in (row.get("keep_msgs") or [])]
                if db_keep != ledger_keep:
                    problems.append(f"pool_reviewed_a {sid}: keep_msgs db={db_keep} != ledger={ledger_keep}")

            # 3.2 Check pool_training_release table against manifest
            db_rel_rows = {r["pool"]: dict(r) for r in db.execute("SELECT * FROM pool_training_release").fetchall()}
            manifest_pools = manifest.get("training_pools", {})
            for p, info in manifest_pools.items():
                db_p = db_rel_rows.get(p)
                if not db_p:
                    problems.append(f"pool_training_release missing pool in DB: {p}")
                    continue

                exp_trainable = 1 if info.get("trainable") else 0
                if db_p["trainable"] != exp_trainable:
                    problems.append(f"pool_training_release {p}: trainable db={db_p['trainable']} != manifest={exp_trainable}")

                exp_sha = str(info.get("sha256") or "")
                if db_p["file_sha256"] != exp_sha:
                    problems.append(f"pool_training_release {p}: file_sha256 db='{db_p['file_sha256']}' != manifest='{exp_sha}'")

                exp_raw = int(info.get("raw_payload_count") or 0)
                if db_p["raw_payload_count"] != exp_raw:
                    problems.append(f"pool_training_release {p}: raw_payload_count db={db_p['raw_payload_count']} != manifest={exp_raw}")

                exp_eligible = int(info.get("eligible_count") or 0)
                if db_p["eligible_count"] != exp_eligible:
                    problems.append(f"pool_training_release {p}: eligible_count db={db_p['eligible_count']} != manifest={exp_eligible}")

                exp_seq = int(info.get("seq_tokens") or 0)
                if db_p["total_seq_tokens"] != exp_seq:
                    problems.append(f"pool_training_release {p}: total_seq_tokens db={db_p['total_seq_tokens']} != manifest={exp_seq}")

                exp_sup = int(info.get("supervised_tokens") or 0)
                if db_p["total_sup_tokens"] != exp_sup:
                    problems.append(f"pool_training_release {p}: total_sup_tokens db={db_p['total_sup_tokens']} != manifest={exp_sup}")

            # Baseline audit row in release table
            if "A_BASELINE_AUDIT" in db_rel_rows:
                db_base = db_rel_rows["A_BASELINE_AUDIT"]
                exp_base_sha = str(baseline_info.get("sha256") or "")
                if db_base["file_sha256"] != exp_base_sha:
                    problems.append(f"pool_training_release A_BASELINE_AUDIT file_sha256 db='{db_base['file_sha256']}' != manifest='{exp_base_sha}'")
                exp_base_cnt = int(baseline_info.get("count") or 0)
                if db_base["raw_payload_count"] != exp_base_cnt:
                    problems.append(f"pool_training_release A_BASELINE_AUDIT count db={db_base['raw_payload_count']} != manifest={exp_base_cnt}")
        finally:
            db.close()

    # 4. Check rule + overlay semantic recomputation for Pool A (prevents keep tampering)
    expected_a_map: dict[str, dict] = {}
    if overlay is not None and a_rule_path.exists():
        a_rule_rows = load_jsonl(a_rule_path)
        for r in a_rule_rows:
            overlaid = apply_overlay_to_a_row(dict(r), overlay)
            expected_a_map[overlaid["sample_id"]] = overlaid

        for row in ledger_rows:
            sid = row.get("sample_id")
            exp = expected_a_map.get(sid)
            if not exp:
                problems.append(f"{sid}: sample found in a_reviewed_ledger but missing in a_rule_ledger")
                continue
            exp_keep = [int(m) for m in (exp.get("keep_msgs") or [])]
            act_keep = [int(m) for m in (row.get("keep_msgs") or [])]
            if act_keep != exp_keep:
                problems.append(f"{sid}: ledger keep_msgs {act_keep} != rule+overlay recomputation {exp_keep}")
            if exp.get("decision") == "REJECT" and (row.get("decision") == "ADMIT" or row.get("training_eligible")):
                problems.append(f"{sid}: ledger admitted sample rejected by rule+overlay ({exp.get('reasons')})")

    # 5. Check Pool A payloads: ids/mask length, spans alignment, recompute ids/mask/spans
    a_payload_path = output_dir / "pool_a_reviewed.jsonl"
    if a_payload_path.exists():
        a_payloads = load_jsonl(a_payload_path)
        if a_payloads:
            if renderer is None:
                try:
                    from to_sft_qwen import Renderer
                    renderer = Renderer()
                except Exception as exc:
                    problems.append(f"failed to initialize Renderer for check: {exc}")
                    renderer = None

            if records is None:
                try:
                    import bc_export
                    records = bc_export.load_records()
                except Exception as exc:
                    problems.append(f"failed to load trajectory records: {exc}")
                    records = []
            records_map = {r["instance_id"]: r for r in records} if records else {}

            for payload in a_payloads:
                sample_id = payload.get("sample_id")
                iid = payload.get("instance_id")
                ids = payload.get("ids") or []
                mask = payload.get("mask") or []
                keep = payload.get("keep_msgs") or []
                spans = payload.get("spans") or {}

                if len(ids) != len(mask):
                    problems.append(f"{sample_id}: len(ids) {len(ids)} != len(mask) {len(mask)}")
                if sum(mask) == 0:
                    problems.append(f"{sample_id}: empty mask in admitted payload")
                if len(ids) > MAX_TOKENS:
                    problems.append(f"{sample_id}: token length {len(ids)} exceeds MAX_TOKENS")

                # Check spans match keep_msgs
                if set(spans.keys()) != {str(m) for m in keep}:
                    problems.append(f"{sample_id}: spans keys {set(spans.keys())} != keep_msgs {set(keep)}")

                # Check payload keep_msgs matches rule+overlay recomputation
                if expected_a_map:
                    exp = expected_a_map.get(sample_id)
                    if not exp:
                        problems.append(f"{sample_id}: payload exists but not found in a_rule_ledger")
                    elif exp.get("decision") != "ADMIT" or not exp.get("training_eligible"):
                        problems.append(f"{sample_id}: payload exists for sample rejected by rule+overlay")
                    else:
                        exp_keep = [int(m) for m in (exp.get("keep_msgs") or [])]
                        if [int(m) for m in keep] != exp_keep:
                            problems.append(f"{sample_id}: payload keep_msgs {keep} != rule+overlay recomputation {exp_keep}")

                # Recompute tokens/mask/spans from record (record missing cannot be silently skipped)
                rec = records_map.get(iid)
                if rec is None:
                    problems.append(f"{sample_id}: trajectory record missing for instance {iid}")
                elif renderer is not None:
                    try:
                        recomputed = _render_trajectory(renderer, rec["messages"])
                        rec_mask = recomputed.mask(keep)
                        rec_spans = recomputed.spans(rec_mask)
                        if recomputed.ids != ids:
                            problems.append(f"{sample_id}: recomputed token ids mismatch")
                        if rec_mask != mask:
                            problems.append(f"{sample_id}: recomputed token mask mismatch")
                        formatted_spans = {str(msg): [first, last] for msg, (first, last) in sorted(rec_spans.items())}
                        if formatted_spans != spans:
                            problems.append(f"{sample_id}: recomputed spans mismatch")
                    except Exception as exc:
                        problems.append(f"{sample_id}: failed to recompute mask: {exc}")

    if problems:
        print("CHECK FAILED:")
        for prob in problems[:30]:
            print(f"  * {prob}")
        if len(problems) > 30:
            print(f"  ... and {len(problems)-30} more problems")
        return 1

    print("CHECK PASSED: all payloads, masks, spans, ledgers, and database indexes are verified.")
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export and verify reviewed Pool A and release manifest.")
    parser.add_argument("--check", action="store_true", help="Verify integrity of released payloads, ledgers, and sqlite index")
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY, help="Path to overlay.json")
    parser.add_argument("--a-rule-ledger", type=Path, default=DEFAULT_A_RULE_LEDGER, help="Path to a_rule_ledger.jsonl")
    parser.add_argument("--bc-rule-ledger", type=Path, default=DEFAULT_BC_RULE_LEDGER, help="Path to bc_rule_ledger.jsonl")
    parser.add_argument("--bc-ledger", type=Path, default=DEFAULT_BC_LEDGER, help="Path to bc_ledger.jsonl")
    parser.add_argument("--baseline-a", type=Path, default=DEFAULT_BASELINE_A, help="Path to baseline sft_admitted.jsonl")
    parser.add_argument("--output-dir", type=Path, default=POOL, help="Output directory for reviewed pools and manifest")
    parser.add_argument("--db", type=Path, default=DB, help="Path to SQLite index db")
    args = parser.parse_args(argv)

    if args.check:
        code = check(
            overlay_path=args.overlay,
            a_rule_path=args.a_rule_ledger,
            bc_rule_path=args.bc_rule_ledger,
            output_dir=args.output_dir,
            db_path=args.db,
        )
        sys.exit(code)

    summary = export(
        overlay_path=args.overlay,
        a_rule_path=args.a_rule_ledger,
        bc_rule_path=args.bc_rule_ledger,
        bc_ledger_path=args.bc_ledger,
        baseline_a_path=args.baseline_a,
        output_dir=args.output_dir,
        db_path=args.db,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
