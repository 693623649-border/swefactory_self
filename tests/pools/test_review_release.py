"""Unit tests for reviewed Pool A export and release manifest verification.

Verifies:
  1. Cross-view message mask: message masked in C1 propagates to A via overlay.
  2. Cross-view quarantine: QUARANTINE in one view rejects the whole trajectory across views.
  3. Packet format: assistant reasoning preserved, tool observation clipped with middle omitted,
     and evaluation/judge results strictly excluded.
  4. Overlay freshness & completeness: stale hashes or incomplete reviews are rejected.
  5. Empty mask rejection: no pseudo-data when all supervised messages are masked.
  6. End-to-end export & check: mocked export, sqlite indexing, manifest creation, and integrity verification.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
sys.path.insert(0, str(RESULTS_DIR))

import model_review
import reviewed_export


class MockRendered:
    """Mock of to_sft_qwen.Rendered providing token ids, mask, and spans."""

    def __init__(self, messages: list[dict]):
        self.messages = messages
        # Generate predictable tokens per message: [1000 + i] * 10
        self.ids = []
        self._msg_spans = {}
        cur = 0
        for i, _ in enumerate(messages):
            length = 10
            start = cur
            end = cur + length
            cur = end
            self.ids.extend([1000 + i] * length)
            self._msg_spans[i] = (start, end)

    def mask(self, keep=None) -> list[int]:
        keep_set = set(keep) if keep is not None else set()
        mask_out = [0] * len(self.ids)
        for i, (start, end) in self._msg_spans.items():
            if i in keep_set:
                for idx in range(start, end):
                    mask_out[idx] = 1
        return mask_out

    def spans(self, mask=None) -> dict:
        result = {}
        for i, (start, end) in self._msg_spans.items():
            if mask is not None and any(mask[start:end]):
                result[i] = (start, end)
        return result


class MockRenderer:
    """Mock renderer returning MockRendered instances."""

    def render(self, messages: list[dict]) -> MockRendered:
        return MockRendered(messages)


def test_cross_view_mask_c1_to_a():
    """Verify that a message flagged as MASK in C1 propagates to A via overlay source_masks."""
    overlay = {
        "quarantine": set(),
        "source_quarantine": set(),
        "masks": {},
        "source_masks": {
            "django__django-11000": {2},  # Message 2 masked during C1 review
        },
    }
    a_row = {
        "sample_id": "A:django__django-11000",
        "instance_id": "django__django-11000",
        "pool": "A",
        "decision": "ADMIT",
        "training_eligible": True,
        "keep_msgs": [2, 4],
        "reasons": [],
    }

    result = reviewed_export.apply_overlay_to_a_row(a_row, overlay)

    assert result["decision"] == "ADMIT"
    assert result["training_eligible"] is True
    assert result["keep_msgs"] == [4]  # Message 2 dropped
    assert result["model_masked"] == 1
    assert [2, "MODEL_REVIEW"] in result["masked_msgs"]
    assert "MASK" in result["reasons_by_model"]


def test_cross_view_quarantine_rejects_a():
    """Verify that QUARANTINE in one view (C1/B) rejects the entire instance across all views in A."""
    overlay = {
        "quarantine": {"C1:sympy__sympy-15000"},
        "source_quarantine": {"sympy__sympy-15000"},  # Cross-view quarantine
        "masks": {},
        "source_masks": {},
    }
    a_row = {
        "sample_id": "A:sympy__sympy-15000",
        "instance_id": "sympy__sympy-15000",
        "pool": "A",
        "decision": "ADMIT",
        "training_eligible": True,
        "keep_msgs": [1, 3],
        "reasons": [],
    }

    result = reviewed_export.apply_overlay_to_a_row(a_row, overlay)

    assert result["decision"] == "REJECT"
    assert result["training_eligible"] is False
    assert "MODEL_QUARANTINE" in result["reasons"]
    assert "QUARANTINE" in result["reasons_by_model"]


def test_packet_assistant_reasoning_and_tool_clip_and_no_eval():
    """Verify packet_for preserves full assistant reasoning, clips long tool output, and excludes eval."""
    long_obs = "OBS_HEAD_" + ("x" * 3000) + "_OBS_TAIL"
    record = {
        "instance_id": "test__pkg-1",
        "problem_statement": "Issue description here",
        "messages": [
            {"role": "user", "content": "Please fix bug."},
            {
                "role": "assistant",
                "content": "I am thinking carefully about the root cause.",
                "reasoning_raw": "Deep reasoning: line 42 has off-by-one bug.",
                "tool_calls": [{"id": "tc1", "name": "bash", "arguments": "pytest"}],
            },
            {
                "role": "tool",
                "name": "bash",
                "tool_call_id": "tc1",
                "content": long_obs,
            },
            {
                "role": "assistant",
                "content": "I fixed it.",
            },
        ],
        # Hidden eval / gold materials that must NEVER leak
        "eval_result": {"f2p_passed": True, "p2p_passed": True},
        "gold_patch": "diff --git a/test b/test\n+fix",
        "node_evidence": {"receipt": "secret"},
    }
    ledger_row = {
        "sample_id": "C1:test__pkg-1",
        "instance_id": "test__pkg-1",
        "pool": "C1",
        "decision": "ADMIT",
        "keep_msgs": [1, 3],
    }

    packet = model_review.packet_for("C1:test__pkg-1", ledger_row, record)

    # 1. Check packet structure
    assert packet["sample_id"] == "C1:test__pkg-1"
    assert packet["instance_id"] == "test__pkg-1"

    # 2. Assistant reasoning preserved
    assistant_entry = next(w for w in packet["window"] if w["msg"] == 1)
    assert assistant_entry["reasoning_raw"] == "Deep reasoning: line 42 has off-by-one bug."

    # 3. Tool observation clipped with middle omitted indicator
    tool_entry = next(w for w in packet["window"] if w["msg"] == 2)
    assert tool_entry["content_truncated"] is True
    assert "[tool observation middle omitted]" in tool_entry["content"]
    assert tool_entry["content"].startswith("OBS_HEAD_")
    assert tool_entry["content"].endswith("_OBS_TAIL")

    # 4. Strict absence of evaluation / test receipts / gold patches anywhere in packet
    dumped = json.dumps(packet)
    assert "f2p_passed" not in dumped
    assert "p2p_passed" not in dumped
    assert "gold_patch" not in dumped
    assert "node_evidence" not in dumped


def test_overlay_stale_rejection(tmp_path: Path):
    """Verify validate_overlay strictly rejects missing, incomplete, or stale overlays."""
    a_rule_file = tmp_path / "a_rule.jsonl"
    a_rule_file.write_text('{"sample_id": "A:1"}\n', encoding="utf-8")
    a_sha = sha256(a_rule_file.read_bytes()).hexdigest()

    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text('{"sample_id": "1", "verdict": "KEEP"}\n', encoding="utf-8")
    labels_sha = sha256(labels_file.read_bytes()).hexdigest()

    # Valid base overlay
    valid_overlay = {
        "complete_review_required": True,
        "labels": str(labels_file),
        "labels_sha256": labels_sha,
        "a_rule_ledger_sha256": a_sha,
        "excluded_by_model": [],
        "sample_masks": {},
    }

    overlay_path = tmp_path / "overlay.json"

    # Case 1: Incomplete overlay
    bad_overlay = dict(valid_overlay, complete_review_required=False)
    overlay_path.write_text(json.dumps(bad_overlay), encoding="utf-8")
    with pytest.raises(SystemExit, match="incomplete overlay"):
        reviewed_export.validate_overlay(overlay_path, a_rule_file)

    # Case 2: Stale labels_sha256
    bad_overlay = dict(valid_overlay, labels_sha256="0000000000000000000000000000000000000000000000000000000000000000")
    overlay_path.write_text(json.dumps(bad_overlay), encoding="utf-8")
    with pytest.raises(SystemExit, match="stale labels_sha256"):
        reviewed_export.validate_overlay(overlay_path, a_rule_file)

    # Case 3: Stale a_rule_ledger_sha256
    bad_overlay = dict(valid_overlay, a_rule_ledger_sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
    overlay_path.write_text(json.dumps(bad_overlay), encoding="utf-8")
    with pytest.raises(SystemExit, match="stale a_rule_ledger_sha256"):
        reviewed_export.validate_overlay(overlay_path, a_rule_file)

    # Case 4: Valid overlay passes
    overlay_path.write_text(json.dumps(valid_overlay), encoding="utf-8")
    loaded = reviewed_export.validate_overlay(overlay_path, a_rule_file)
    assert loaded["sha256"] == sha256(overlay_path.read_bytes()).hexdigest()


def test_empty_mask_rejection():
    """Verify that an admitted A row whose supervised messages are all masked is rejected."""
    overlay = {
        "quarantine": set(),
        "source_quarantine": set(),
        "masks": {"A:inst_all_masked": {1}},
        "source_masks": {},
    }
    a_row = {
        "sample_id": "A:inst_all_masked",
        "instance_id": "inst_all_masked",
        "pool": "A",
        "decision": "ADMIT",
        "training_eligible": True,
        "keep_msgs": [1],
        "reasons": [],
    }

    result = reviewed_export.apply_overlay_to_a_row(a_row, overlay)
    assert result["decision"] == "REJECT"
    assert result["training_eligible"] is False
    assert "MODEL_EMPTY_MASK" in result["reasons"]
    assert result["keep_msgs"] == []


def test_end_to_end_export_and_check_mocked(tmp_path: Path):
    """End-to-end test of export and check using mocks and temporary workspace."""
    pool_dir = tmp_path / "pool_bc"
    formal_dir = tmp_path / "pool_a_formal"
    pool_dir.mkdir(parents=True)
    formal_dir.mkdir(parents=True)

    # 1. Setup mock records
    records = [
        {
            "instance_id": "inst_admit_1",
            "messages": [
                {"role": "user", "content": "solve task"},
                {"role": "assistant", "content": "plan"},
                {"role": "assistant", "content": "solution code"},
            ],
        },
        {
            "instance_id": "inst_quarantined",
            "messages": [
                {"role": "user", "content": "solve task 2"},
                {"role": "assistant", "content": "bad reasoning"},
            ],
        },
    ]

    # 2. Setup A rule ledger
    a_rule_path = pool_dir / "a_rule_ledger.jsonl"
    a_rule_rows = [
        {
            "sample_id": "A:inst_admit_1",
            "instance_id": "inst_admit_1",
            "pool": "A",
            "decision": "ADMIT",
            "training_eligible": True,
            "keep_msgs": [1, 2],
            "reasons": [],
        },
        {
            "sample_id": "A:inst_quarantined",
            "instance_id": "inst_quarantined",
            "pool": "A",
            "decision": "ADMIT",
            "training_eligible": True,
            "keep_msgs": [1],
            "reasons": [],
        },
    ]
    reviewed_export.write_jsonl(a_rule_path, a_rule_rows)
    a_rule_sha = sha256(a_rule_path.read_bytes()).hexdigest()

    # 3. Setup labels and overlay
    labels_path = pool_dir / "labels.jsonl"
    labels_path.write_text('{"sample_id": "C1:inst_quarantined", "verdict": "QUARANTINE"}\n', encoding="utf-8")
    labels_sha = sha256(labels_path.read_bytes()).hexdigest()

    overlay_path = pool_dir / "overlay.json"
    overlay_data = {
        "complete_review_required": True,
        "labels": str(labels_path),
        "labels_sha256": labels_sha,
        "a_rule_ledger_sha256": a_rule_sha,
        "excluded_by_model": [
            {"sample_id": "C1:inst_quarantined", "verdict": "QUARANTINE"},
            {"sample_id": "A:inst_quarantined", "verdict": "QUARANTINE"},
        ],
        "source_quarantine": ["inst_quarantined"],
        "sample_masks": {},
        "source_masks": {},
    }
    reviewed_export.write_json(overlay_path, overlay_data)

    # 4. Setup mock baseline and pools B, C1, C2, C3
    baseline_a_path = formal_dir / "sft_admitted.jsonl"
    baseline_a_path.write_text('{"sample_id": "A_BASE:1"}\n', encoding="utf-8")

    # Pool B, C1, C2 payloads and ledger
    for p in ("b", "c1", "c2"):
        payload_file = pool_dir / f"pool_{p}.jsonl"
        payload_file.write_text(f'{{"sample_id": "{p.upper()}:inst_admit_1"}}\n', encoding="utf-8")

    # Pool C3: 2 raw payloads, but only 1 is eligible in bc_ledger
    c3_payload_file = pool_dir / "pool_c3.jsonl"
    c3_payload_file.write_text(
        '{"sample_id": "C3:eligible"}\n{"sample_id": "C3:unreviewed_ineligible"}\n',
        encoding="utf-8",
    )

    bc_ledger_path = pool_dir / "bc_ledger.jsonl"
    bc_ledger_rows = [
        {"sample_id": "B:inst_admit_1", "instance_id": "inst_admit_1", "pool": "B", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 100, "supervised_tokens": 50},
        {"sample_id": "C1:inst_admit_1", "instance_id": "inst_admit_1", "pool": "C1", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 120, "supervised_tokens": 60},
        {"sample_id": "C2:inst_admit_1", "instance_id": "inst_admit_1", "pool": "C2", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 150, "supervised_tokens": 70},
        {"sample_id": "C3:eligible", "instance_id": "inst_c3_1", "pool": "C3", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 200, "supervised_tokens": 80},
        {"sample_id": "C3:unreviewed_ineligible", "instance_id": "inst_c3_2", "pool": "C3", "decision": "REJECT", "training_eligible": False, "seq_tokens": 0, "supervised_tokens": 0},
    ]
    reviewed_export.write_jsonl(bc_ledger_path, bc_ledger_rows)

    # Seeds (untrainable)
    b_seeds_path = pool_dir / "b_teacher_seeds.jsonl"
    b_seeds_path.write_text('{"sample_id": "SEED:1"}\n', encoding="utf-8")

    # Database with existing tables
    db_path = formal_dir / "index.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE existing_table (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO existing_table VALUES (42)")
    conn.commit()
    conn.close()

    mock_renderer = MockRenderer()

    # 5. Execute export
    summary = reviewed_export.export(
        overlay_path=overlay_path,
        a_rule_path=a_rule_path,
        bc_rule_path=None,
        bc_ledger_path=bc_ledger_path,
        baseline_a_path=baseline_a_path,
        output_dir=pool_dir,
        db_path=db_path,
        renderer=mock_renderer,
        records=records,
    )

    # Verify export outputs
    assert summary["a_rule_candidates"] == 2
    assert summary["a_reviewed_admitted"] == 1  # inst_quarantined rejected

    # Check pool_a_reviewed.jsonl
    a_reviewed_payloads = reviewed_export.load_jsonl(pool_dir / "pool_a_reviewed.jsonl")
    assert len(a_reviewed_payloads) == 1
    assert a_reviewed_payloads[0]["sample_id"] == "A:inst_admit_1"
    assert a_reviewed_payloads[0]["keep_msgs"] == [1, 2]
    assert len(a_reviewed_payloads[0]["ids"]) == len(a_reviewed_payloads[0]["mask"])

    # Check a_reviewed_ledger.jsonl
    a_reviewed_ledger = reviewed_export.load_jsonl(pool_dir / "a_reviewed_ledger.jsonl")
    assert len(a_reviewed_ledger) == 2
    quarantined_ledger = next(r for r in a_reviewed_ledger if r["instance_id"] == "inst_quarantined")
    assert quarantined_ledger["decision"] == "REJECT"
    assert quarantined_ledger["training_eligible"] is False
    assert "MODEL_QUARANTINE" in quarantined_ledger["reasons"]

    # Check release_manifest.json
    manifest = json.loads((pool_dir / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit_baseline"]["trainable"] is False
    assert manifest["seeds_audit"]["trainable"] is False
    assert manifest["seeds_audit"]["eligible_count"] == 0

    c3_info = manifest["training_pools"]["C3"]
    assert c3_info["raw_payload_count"] == 2
    assert c3_info["eligible_count"] == 1  # unreviewed ineligible is not granted eligibility

    # Check view_groups
    view_groups = manifest["view_groups"]
    assert "inst_admit_1" in view_groups
    pools_for_inst = {v["pool"] for v in view_groups["inst_admit_1"]}
    assert {"A", "B", "C1", "C2"}.issubset(pools_for_inst)

    # Check SQLite preservation and registration
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Existing table preserved
    assert cur.execute("SELECT id FROM existing_table").fetchone()[0] == 42
    # Dedicated table pool_reviewed_a
    assert cur.execute("SELECT COUNT(*) FROM pool_reviewed_a WHERE training_eligible = 1").fetchone()[0] == 1
    assert cur.execute("SELECT COUNT(*) FROM pool_reviewed_a WHERE decision = 'REJECT'").fetchone()[0] == 1
    # Dedicated table pool_training_release
    release_rows = cur.execute("SELECT pool, trainable, eligible_count FROM pool_training_release").fetchall()
    release_map = {r[0]: (r[1], r[2]) for r in release_rows}
    assert release_map["A"] == (1, 1)
    assert release_map["C3"] == (1, 1)
    assert release_map["A_BASELINE_AUDIT"] == (0, 0)
    conn.close()

    # 6. Execute check verification
    check_code = reviewed_export.check(
        overlay_path=overlay_path,
        a_rule_path=a_rule_path,
        bc_rule_path=None,
        output_dir=pool_dir,
        db_path=db_path,
        renderer=mock_renderer,
        records=records,
    )
    assert check_code == 0

    # 7. Check fails when payload is tampered
    tampered_file = pool_dir / "pool_a_reviewed.jsonl"
    tampered_file.write_text('{"tampered": true}\n', encoding="utf-8")
    fail_code = reviewed_export.check(
        overlay_path=overlay_path,
        a_rule_path=a_rule_path,
        bc_rule_path=None,
        output_dir=pool_dir,
        db_path=db_path,
        renderer=mock_renderer,
        records=records,
    )
    assert fail_code == 1


class RealProtocolMockRenderer:
    """Mock simulating real to_sft_qwen.Renderer whose render() returns a str."""

    def render(self, messages: list[dict]) -> str:
        return "<chat_template_string_output>"


def test_real_protocol_renderer_mock_returns_str(monkeypatch):
    """Verify that renderer with .render() returning str is handled via Rendered(renderer, messages)."""
    monkeypatch.setattr("to_sft_qwen.Rendered", lambda r, msgs: MockRendered(msgs))

    renderer = RealProtocolMockRenderer()
    row = {
        "sample_id": "A:test_real_proto",
        "instance_id": "test_real_proto",
        "pool": "A",
        "decision": "ADMIT",
        "training_eligible": True,
        "keep_msgs": [1],
        "reasons": [],
    }
    record = {
        "instance_id": "test_real_proto",
        "messages": [
            {"role": "user", "content": "solve task"},
            {"role": "assistant", "content": "code solution"},
        ],
    }

    res = reviewed_export.render_a_row(row, record, renderer)
    assert res["payload"] is not None
    assert res["payload"]["sample_id"] == "A:test_real_proto"
    assert res["payload"]["keep_msgs"] == [1]
    assert len(res["payload"]["ids"]) == 20
    assert sum(res["payload"]["mask"]) == 10
    assert res["payload"]["spans"] == {"1": [10, 20]}


def test_validate_overlay_missing_hashes_rejected(tmp_path: Path):
    """Verify validate_overlay strictly rejects missing a/bc hashes."""
    a_rule_file = tmp_path / "a_rule.jsonl"
    a_rule_file.write_text('{"sample_id": "A:1"}\n', encoding="utf-8")
    a_sha = sha256(a_rule_file.read_bytes()).hexdigest()

    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text('{"sample_id": "1", "verdict": "KEEP"}\n', encoding="utf-8")
    labels_sha = sha256(labels_file.read_bytes()).hexdigest()

    overlay_path = tmp_path / "overlay.json"

    # Base valid overlay without a hash
    base_no_a_hash = {
        "complete_review_required": True,
        "labels": str(labels_file),
        "labels_sha256": labels_sha,
    }
    overlay_path.write_text(json.dumps(base_no_a_hash), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing required a_rule_ledger_sha256"):
        reviewed_export.validate_overlay(overlay_path, a_rule_file)

    # Overlay with a hash, now test explicit bc_rule_path
    base_with_a = dict(base_no_a_hash, a_rule_ledger_sha256=a_sha)
    overlay_path.write_text(json.dumps(base_with_a), encoding="utf-8")

    # Explicit bc_rule_path that does not exist
    missing_bc_file = tmp_path / "nonexistent_bc.jsonl"
    with pytest.raises(SystemExit, match="bc_rule_ledger not found"):
        reviewed_export.validate_overlay(overlay_path, a_rule_file, bc_rule_path=missing_bc_file)

    # Explicit bc_rule_path exists, but overlay lacks bc hash
    bc_rule_file = tmp_path / "bc_rule.jsonl"
    bc_rule_file.write_text('{"sample_id": "B:1"}\n', encoding="utf-8")
    bc_sha = sha256(bc_rule_file.read_bytes()).hexdigest()

    with pytest.raises(SystemExit, match="missing required bc_rule_ledger_sha256"):
        reviewed_export.validate_overlay(overlay_path, a_rule_file, bc_rule_path=bc_rule_file)

    # Explicit bc_rule_path with stale bc hash
    stale_bc_overlay = dict(base_with_a, bc_rule_ledger_sha256="0000000000000000000000000000000000000000000000000000000000000000")
    overlay_path.write_text(json.dumps(stale_bc_overlay), encoding="utf-8")
    with pytest.raises(SystemExit, match="bc rule ledger changed after model review"):
        reviewed_export.validate_overlay(overlay_path, a_rule_file, bc_rule_path=bc_rule_file)

    # Compatible with formal key bc_rule_ledger_sha256
    ok_formal = dict(base_with_a, bc_rule_ledger_sha256=bc_sha)
    overlay_path.write_text(json.dumps(ok_formal), encoding="utf-8")
    res_formal = reviewed_export.validate_overlay(overlay_path, a_rule_file, bc_rule_path=bc_rule_file)
    assert res_formal["sha256"] is not None

    # Compatible with legacy key rule_ledger_sha256
    ok_legacy = dict(base_with_a, rule_ledger_sha256=bc_sha)
    overlay_path.write_text(json.dumps(ok_legacy), encoding="utf-8")
    res_legacy = reviewed_export.validate_overlay(overlay_path, a_rule_file, bc_rule_path=bc_rule_file)
    assert res_legacy["sha256"] is not None


def _setup_export_workspace(tmp_path: Path):
    """Helper to set up a minimal export workspace returning paths, records, and renderer."""
    pool_dir = tmp_path / "pool_bc"
    formal_dir = tmp_path / "pool_a_formal"
    pool_dir.mkdir(parents=True)
    formal_dir.mkdir(parents=True)

    records = [
        {
            "instance_id": "inst_1",
            "messages": [
                {"role": "user", "content": "solve task"},
                {"role": "assistant", "content": "plan"},
                {"role": "assistant", "content": "code"},
            ],
        }
    ]

    a_rule_path = pool_dir / "a_rule_ledger.jsonl"
    a_rule_rows = [
        {
            "sample_id": "A:inst_1",
            "instance_id": "inst_1",
            "pool": "A",
            "decision": "ADMIT",
            "training_eligible": True,
            "keep_msgs": [1, 2],
            "reasons": [],
        }
    ]
    reviewed_export.write_jsonl(a_rule_path, a_rule_rows)
    a_rule_sha = sha256(a_rule_path.read_bytes()).hexdigest()

    labels_path = pool_dir / "labels.jsonl"
    labels_path.write_text('{"sample_id": "A:inst_1", "verdict": "KEEP"}\n', encoding="utf-8")
    labels_sha = sha256(labels_path.read_bytes()).hexdigest()

    overlay_path = pool_dir / "overlay.json"
    overlay_data = {
        "complete_review_required": True,
        "labels": str(labels_path),
        "labels_sha256": labels_sha,
        "a_rule_ledger_sha256": a_rule_sha,
        "excluded_by_model": [],
        "source_quarantine": [],
        "sample_masks": {},
        "source_masks": {},
    }
    reviewed_export.write_json(overlay_path, overlay_data)

    baseline_a_path = formal_dir / "sft_admitted.jsonl"
    baseline_a_path.write_text('{"sample_id": "A_BASE:1"}\n', encoding="utf-8")

    for p in ("b", "c1", "c2", "c3"):
        (pool_dir / f"pool_{p}.jsonl").write_text(f'{{"sample_id": "{p.upper()}:inst_1"}}\n', encoding="utf-8")

    bc_ledger_path = pool_dir / "bc_ledger.jsonl"
    bc_ledger_rows = [
        {"sample_id": "B:inst_1", "instance_id": "inst_1", "pool": "B", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 100, "supervised_tokens": 50},
        {"sample_id": "C1:inst_1", "instance_id": "inst_1", "pool": "C1", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 120, "supervised_tokens": 60},
        {"sample_id": "C2:inst_1", "instance_id": "inst_1", "pool": "C2", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 150, "supervised_tokens": 70},
        {"sample_id": "C3:inst_1", "instance_id": "inst_1", "pool": "C3", "decision": "ADMIT", "training_eligible": True, "seq_tokens": 200, "supervised_tokens": 80},
    ]
    reviewed_export.write_jsonl(bc_ledger_path, bc_ledger_rows)

    db_path = formal_dir / "index.sqlite3"
    renderer = MockRenderer()

    reviewed_export.export(
        overlay_path=overlay_path,
        a_rule_path=a_rule_path,
        bc_rule_path=None,
        bc_ledger_path=bc_ledger_path,
        baseline_a_path=baseline_a_path,
        output_dir=pool_dir,
        db_path=db_path,
        renderer=renderer,
        records=records,
    )

    return {
        "pool_dir": pool_dir,
        "formal_dir": formal_dir,
        "overlay_path": overlay_path,
        "a_rule_path": a_rule_path,
        "bc_ledger_path": bc_ledger_path,
        "baseline_a_path": baseline_a_path,
        "db_path": db_path,
        "renderer": renderer,
        "records": records,
    }


def test_check_rejects_db_field_tampered_same_count(tmp_path: Path):
    """Verify check rejects when SQLite row fields are modified even though row count is identical."""
    ws = _setup_export_workspace(tmp_path)
    db_path = ws["db_path"]

    # Verify pristine check passes
    assert reviewed_export.check(
        overlay_path=ws["overlay_path"],
        a_rule_path=ws["a_rule_path"],
        bc_rule_path=None,
        output_dir=ws["pool_dir"],
        db_path=db_path,
        renderer=ws["renderer"],
        records=ws["records"],
    ) == 0

    # 1. Tamper pool_reviewed_a: seq_tokens modified
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE pool_reviewed_a SET seq_tokens = 999999 WHERE sample_id = 'A:inst_1'")
    conn.commit()
    conn.close()

    assert reviewed_export.check(
        overlay_path=ws["overlay_path"],
        a_rule_path=ws["a_rule_path"],
        bc_rule_path=None,
        output_dir=ws["pool_dir"],
        db_path=db_path,
        renderer=ws["renderer"],
        records=ws["records"],
    ) == 1

    # Restore seq_tokens, then tamper training_eligible
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE pool_reviewed_a SET seq_tokens = 30 WHERE sample_id = 'A:inst_1'")
    conn.execute("UPDATE pool_reviewed_a SET training_eligible = 0 WHERE sample_id = 'A:inst_1'")
    conn.commit()
    conn.close()

    assert reviewed_export.check(
        overlay_path=ws["overlay_path"],
        a_rule_path=ws["a_rule_path"],
        bc_rule_path=None,
        output_dir=ws["pool_dir"],
        db_path=db_path,
        renderer=ws["renderer"],
        records=ws["records"],
    ) == 1

    # Restore training_eligible, then tamper pool_training_release file_sha256
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE pool_reviewed_a SET training_eligible = 1 WHERE sample_id = 'A:inst_1'")
    conn.execute("UPDATE pool_training_release SET file_sha256 = 'bad_hash' WHERE pool = 'A'")
    conn.commit()
    conn.close()

    assert reviewed_export.check(
        overlay_path=ws["overlay_path"],
        a_rule_path=ws["a_rule_path"],
        bc_rule_path=None,
        output_dir=ws["pool_dir"],
        db_path=db_path,
        renderer=ws["renderer"],
        records=ws["records"],
    ) == 1


def test_check_rejects_tampered_keep_msgs_bypass(tmp_path: Path):
    """Verify check recomputes rule+overlay and rejects if keep_msgs was tampered in payload and ledger."""
    ws = _setup_export_workspace(tmp_path)
    pool_dir = ws["pool_dir"]
    db_path = ws["db_path"]

    # Tamper payload and ledger keep_msgs to [1] instead of [1, 2]
    payload_file = pool_dir / "pool_a_reviewed.jsonl"
    payloads = reviewed_export.load_jsonl(payload_file)
    payloads[0]["keep_msgs"] = [1]
    # Re-save payload and update its sha256 everywhere so pure hash checks pass
    reviewed_export.write_jsonl(payload_file, payloads)
    new_payload_sha = sha256(payload_file.read_bytes()).hexdigest()

    ledger_file = pool_dir / "a_reviewed_ledger.jsonl"
    ledger = reviewed_export.load_jsonl(ledger_file)
    ledger[0]["keep_msgs"] = [1]
    ledger[0]["payload_sha256"] = new_payload_sha
    reviewed_export.write_jsonl(ledger_file, ledger)

    manifest_file = pool_dir / "release_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["training_pools"]["A"]["sha256"] = new_payload_sha
    reviewed_export.write_json(manifest_file, manifest)

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE pool_reviewed_a SET keep_msgs = '[1]', payload_sha256 = ? WHERE sample_id = 'A:inst_1'", (new_payload_sha,))
    conn.execute("UPDATE pool_training_release SET file_sha256 = ? WHERE pool = 'A'", (new_payload_sha,))
    conn.commit()
    conn.close()

    # check must fail because rule+overlay requires keep_msgs == [1, 2]
    code = reviewed_export.check(
        overlay_path=ws["overlay_path"],
        a_rule_path=ws["a_rule_path"],
        bc_rule_path=None,
        output_dir=pool_dir,
        db_path=db_path,
        renderer=ws["renderer"],
        records=ws["records"],
    )
    assert code == 1


def test_check_fails_when_record_missing(tmp_path: Path):
    """Verify check rejects when trajectory record is missing for payload verification."""
    ws = _setup_export_workspace(tmp_path)

    # Provide empty records
    code = reviewed_export.check(
        overlay_path=ws["overlay_path"],
        a_rule_path=ws["a_rule_path"],
        bc_rule_path=None,
        output_dir=ws["pool_dir"],
        db_path=ws["db_path"],
        renderer=ws["renderer"],
        records=[],
    )
    assert code == 1
