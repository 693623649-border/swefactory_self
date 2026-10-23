"""Tests for split isolation, benchmark family protection, and fail-closed gates.

Verifies:
  1. Case-insensitive protection across all 12 verified repositories.
  2. Flask and other benchmark families are denied Pool A admission (tested via pure gate and formal flow).
  3. Protected family failure samples are excluded from Pool C3 (training_eligible=False) and teacher seeds.
  4. Non-protected legitimate failure samples are not falsely rejected.
  5. Missing or invalid protection config fails closed (raises SplitIsolationConfigError).
  6. Index database check strictly rejects admitted benchmark families while allowing proper failures.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "results"
sys.path.insert(0, str(RESULTS))

import bc_export as bx  # noqa: E402
import build_index_db as bid  # noqa: E402
import segment_review as sr  # noqa: E402
import split_isolation as si  # noqa: E402


# --- 1. Case-insensitive protection ----------------------------------------

def test_case_insensitive_protection():
    cfg = si.load_protected_families()
    assert len(cfg.families) == 12
    assert cfg.dataset == "SWE-bench/SWE-bench_Verified"
    assert cfg.split == "test"
    assert cfg.source_count == 500
    assert cfg.source_agent_view_sha256 == "7fc3c8e1d9ae2e701e16b4d9439dbcbb0a541c2cdfc1ef2d5cb9a2f1a4e712fe"

    # All variations of pallets/flask must match
    assert si.is_benchmark_family("pallets/flask") is True
    assert si.is_benchmark_family("Pallets/Flask") is True
    assert si.is_benchmark_family("PALLETS/FLASK") is True
    assert si.is_benchmark_family("  pallets/flask  ") is True
    assert si.is_benchmark_family("pAlLeTs/FlAsK") is True

    # Other verified repos case-insensitivity
    for repo in ("django/django", "sympy/sympy", "pytest-dev/pytest", "scikit-learn/scikit-learn"):
        assert si.is_benchmark_family(repo.upper()) is True
        assert si.is_benchmark_family(repo.title()) is True

    # Non-protected repos must not match
    assert si.is_benchmark_family("unrelated/repo") is False
    assert si.is_benchmark_family("pallets-eco/flask-admin") is False
    assert si.is_benchmark_family("psf/black") is False
    assert si.is_benchmark_family("") is False
    assert si.is_benchmark_family(None) is False


# --- 2. Flask denied A admission via pure gate -----------------------------

def test_flask_denied_a_admission_pure_gate():
    # Pure gate function behavior
    ok, reason = si.evaluate_benchmark_gate("pallets/flask")
    assert ok is False
    assert reason == "benchmark_family"

    ok, reason = si.evaluate_benchmark_gate("PALLETS/FLASK")
    assert ok is False
    assert reason == "benchmark_family"

    ok, reason = si.evaluate_benchmark_gate("my-org/my-project")
    assert ok is True
    assert reason is None


def test_formal_admission_rejects_flask_in_simulated_pipeline():
    """Simulate formal admission gate loop with a perfectly passing flask trajectory."""
    protected_cfg = si.load_protected_families()
    protected_families = protected_cfg.families
    validation_families = {si.normalize_family("some-val/repo")}

    sample_task = {"user": "pallets", "repo": "flask", "instance_id": "pallets_flask_pr3574"}
    family = f"{sample_task['user']}/{sample_task['repo']}"

    reject = []
    # Test gate evaluation exactly as written in formal_export.py
    bench_ok, bench_reason = si.evaluate_benchmark_gate(family, protected_families)
    if not bench_ok and bench_reason:
        reject.append(bench_reason)
    if si.normalize_family(family) in validation_families:
        reject.append("validation_family")

    assert "benchmark_family" in reject
    admitted = not reject
    assert admitted is False


# --- 3. Protected family failures excluded from C3 and seeds --------------

def _make_mock_failure_diag():
    diag = MagicMock(spec=sr.Diagnosis)
    diag.final_msg = 5
    diag.messages = [{} for _ in range(6)]
    diag.final_text = "I could not resolve this issue. The tests failed."

    mock_test = MagicMock()
    mock_test.test.state = "fail"
    mock_test.test.test_ran = True
    mock_test.obs_msg = 4
    mock_test.command = "pytest tests"
    mock_test.test.as_dict.return_value = {"state": "fail", "evidence": "1 failed"}
    diag.test_events = [mock_test]

    mock_patch = MagicMock()
    mock_patch.content_lines = 10
    mock_patch.test_files = []
    mock_patch.source_files = ["app.py"]
    diag.patch = mock_patch
    return diag, mock_test


def test_protected_family_failure_denied_c3_and_seed():
    record = {
        "instance_id": "pallets_flask_pr3563",
        "_batch": "batch_001",
        "success": False,
        "finish_reason": "finish",
        "messages": [{} for _ in range(6)],
    }
    # Admission record marked not admitted, family = pallets/flask
    admission = {
        "instance_id": "pallets_flask_pr3563",
        "family": "pallets/flask",
        "admitted": False,
        "rejected": ["not_passed", "f2p_incomplete", "benchmark_family"],
    }
    diag, _ = _make_mock_failure_diag()

    # C3 evaluation must reject and never be training eligible
    c3_row = bx.evaluate_c3(record, admission, diag)
    assert c3_row is not None
    assert c3_row["decision"] == "REJECT"
    assert c3_row["training_eligible"] is False
    assert "SPLIT_ISOLATION_EXCLUDED" in c3_row["reasons"]

    # Teacher seed must return None
    seed = bx.evaluate_b_seeds(record, admission, diag)
    assert seed is None


def test_isolation_reasons_quarantine_c3_and_seed():
    """Any instance carrying validation_family, benchmark_family, or unknown_task is quarantined."""
    for reason in ("validation_family", "benchmark_family", "unknown_task"):
        admission = {
            "instance_id": f"inst_{reason}",
            "family": "unrelated/family",
            "admitted": False,
            "rejected": [reason, "not_passed"],
        }
        record = {
            "instance_id": f"inst_{reason}",
            "success": False,
            "finish_reason": "finish",
            "messages": [{} for _ in range(6)],
        }
        diag, _ = _make_mock_failure_diag()

        c3_row = bx.evaluate_c3(record, admission, diag)
        assert c3_row is not None
        assert c3_row["decision"] == "REJECT"
        assert c3_row["training_eligible"] is False
        assert "SPLIT_ISOLATION_EXCLUDED" in c3_row["reasons"]

        seed = bx.evaluate_b_seeds(record, admission, diag)
        assert seed is None


# --- 4. Unprotected legitimate failure not hurt ----------------------------

def test_unprotected_legitimate_failure_not_hurt():
    record = {
        "instance_id": "clean_repo_pr100",
        "_batch": "batch_002",
        "success": False,
        "finish_reason": "finish",
        "messages": [{} for _ in range(6)],
    }
    admission = {
        "instance_id": "clean_repo_pr100",
        "family": "unprotected/clean_repo",
        "admitted": False,
        "rejected": ["not_passed", "f2p_incomplete"],
    }
    diag, _ = _make_mock_failure_diag()

    # C3 evaluation should not be blocked by split isolation
    c3_row = bx.evaluate_c3(record, admission, diag)
    assert c3_row is not None
    assert "SPLIT_ISOLATION_EXCLUDED" not in c3_row["reasons"]
    # Depending on honesty screen, decision can be ADMIT or REVIEW, but not split-isolated
    assert c3_row["decision"] in ("ADMIT", "REVIEW")

    # Teacher seed evaluation should produce a valid seed
    seed = bx.evaluate_b_seeds(record, admission, diag)
    assert seed is not None
    assert seed["seed_id"] == "BSEED:clean_repo_pr100"
    assert seed["family"] == "unprotected/clean_repo"


# --- 5. Missing or invalid config fails closed -----------------------------

def test_missing_config_fails_closed(tmp_path):
    missing_path = tmp_path / "non_existent.json"
    with pytest.raises(si.SplitIsolationConfigError):
        si.load_protected_families(missing_path)

    with pytest.raises(si.SplitIsolationConfigError):
        si.is_benchmark_family("pallets/flask", config_path=missing_path)

    with pytest.raises(si.SplitIsolationConfigError):
        si.evaluate_benchmark_gate("pallets/flask", config_path=missing_path)


def test_invalid_config_fails_closed(tmp_path):
    # Empty JSON object
    invalid_path = tmp_path / "empty.json"
    invalid_path.write_text("{}", encoding="utf-8")
    with pytest.raises(si.SplitIsolationConfigError):
        si.load_protected_families(invalid_path)

    # Empty families list
    no_families = tmp_path / "no_families.json"
    no_families.write_text(json.dumps({
        "dataset": "test", "split": "test", "source_count": 500,
        "source_agent_view_sha256": "abc", "protected_families": []
    }), encoding="utf-8")
    with pytest.raises(si.SplitIsolationConfigError):
        si.load_protected_families(no_families)


# --- 6. Index DB check rejects admitted benchmark family but not failures --

def test_index_check_rejects_admitted_benchmark_family(tmp_path, monkeypatch):
    db_path = tmp_path / "test_index.sqlite3"
    rows = [{"instance_id": "pallets_flask_pr3574", "family": "pallets/flask", "admitted": True,
             "success": True, "rejected": []}]
    conn = sqlite3.connect(db_path)
    conn.executescript(bid.SCHEMA)
    conn.execute(
        "INSERT INTO samples (instance_id, family, admitted, success, rejected, length_gate) VALUES (?, ?, ?, ?, ?, ?)",
        ("pallets_flask_pr3574", "pallets/flask", 1, 1, "[]", 65536),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(bid, "DB", db_path)
    monkeypatch.setattr(bid, "load_jsonl", lambda path: rows if path.name == "admission_ledger.jsonl" else [])
    assert bid.check() == 1


def test_index_check_allows_benchmark_failures_and_normal_samples(tmp_path, monkeypatch):
    db_path = tmp_path / "test_index.sqlite3"
    rows = [
        {"instance_id": "pallets_flask_pr3563", "family": "pallets/flask", "admitted": False,
         "success": False, "rejected": ["benchmark_family", "not_passed"]},
        {"instance_id": "normal_repo_pr1", "family": "normal/repo", "admitted": True,
         "success": True, "rejected": []},
        {"instance_id": "normal_repo_pr2", "family": "normal/repo", "admitted": False,
         "success": False, "rejected": ["not_passed"]},
    ]
    traj_dir = tmp_path / "batch_test" / "deepseek"
    traj_dir.mkdir(parents=True)
    traj_path = traj_dir / "trajectories.jsonl"
    traj_path.write_text("".join(json.dumps({"instance_id": row["instance_id"], "messages": []}) + "\n" for row in rows), encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.executescript(bid.SCHEMA)
    for row in rows:
        conn.execute(
            "INSERT INTO samples (instance_id, family, admitted, success, rejected, length_gate, traj_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (row["instance_id"], row["family"], int(row["admitted"]), int(row["success"]), json.dumps(row["rejected"]), 65536, str(traj_path)),
        )
        for gate in bid.GATE_NAMES:
            conn.execute("INSERT INTO gates VALUES (?, ?, ?)", (row["instance_id"], gate, int(gate not in row["rejected"])))
    conn.commit()
    conn.close()
    monkeypatch.setattr(bid, "DB", db_path)
    monkeypatch.setattr(bid, "load_jsonl", lambda path: rows if path.name == "admission_ledger.jsonl" else [])
    assert bid.check() == 0


def test_unprovable_family_is_quarantined():
    record = {"instance_id": "unknown", "success": False, "finish_reason": "finish",
              "messages": [{} for _ in range(6)]}
    admission = {"instance_id": "unknown", "family": "", "admitted": False, "rejected": ["unknown_task"]}
    diag, _ = _make_mock_failure_diag()
    row = bx.evaluate_c3(record, admission, diag)
    assert row["decision"] == "REJECT" and row["training_eligible"] is False
    assert bx.evaluate_b_seeds(record, admission, diag) is None
