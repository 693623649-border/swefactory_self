"""Behavioral regressions for semantic source masks and review request receipts."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "results"))
import model_review as mr
import model_review_bulk as bulk


def test_review_holds_one_view_but_explicit_mask_propagates(tmp_path, monkeypatch):
    ledger = {f"{p}:x": {"sample_id": f"{p}:x", "instance_id": "x", "pool": p,
                           "decision": "ADMIT", "keep_msgs": [2, 4]}
              for p in ("A", "C1", "C2")}
    labels = [{"sample_id": f"{p}:x", "instance_id": "x", "pool": p,
               "verdict": verdict, "reviewer_model": "test", "notes": "",
               "label_json": json.dumps([{"msg": 2, "label": "KEEP"},
                                          {"msg": 4, "label": "MASK" if p == "C1" else "KEEP"}])}
              for p, verdict in (("A", "KEEP"), ("C1", "MASK"), ("C2", "REVIEW"))]
    label_path = tmp_path / "model_labels.jsonl"
    label_path.write_text("".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8")
    monkeypatch.setattr(mr, "POOL", tmp_path)
    monkeypatch.setattr(mr, "REVIEW", tmp_path)
    monkeypatch.setattr(mr, "load_ledger", lambda: ledger)
    result = mr.apply_review()
    assert result["source_masks"] == {"x": [4]}
    assert result["sample_masks"] == {"A:x": [4], "C1:x": [4], "C2:x": [2, 4]}
    assert result["pools"]["A"]["final_admitted"] == 1
    assert result["pools"]["C2"]["final_admitted"] == 0
    assert mr.apply_review()["sample_masks"] == result["sample_masks"]
    labels[2]["label_json"] = json.dumps([{"msg": 2, "label": "MASK"}, {"msg": 4, "label": "KEEP"}])
    label_path.write_text("".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8")
    assert mr.apply_review()["source_masks"] == {"x": [2, 4]}
    labels[1]["verdict"] = "QUARANTINE"
    label_path.write_text("".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8")
    quarantined = mr.apply_review()
    assert {r["sample_id"] for r in quarantined["excluded_by_model"]} == set(ledger)


def test_max_length_is_logged_and_does_not_repeat_same_budget(tmp_path, monkeypatch):
    calls = []
    def fake_post(payload, api_key, **kwargs):
        calls.append(payload)
        return {"model": "returned-model", "usage": {"prompt_tokens": 10,
                "completion_tokens": 32768, "completion_tokens_details": {"reasoning_tokens": 32768}},
                "choices": [{"finish_reason": "length", "message": {"content": "", "reasoning_content": "reason"}}]}
    monkeypatch.setattr(bulk, "_post", fake_post)
    usage = tmp_path / "usage.jsonl"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        bulk.review_one({"packet_id": "MR-A:x", "packet_sha256": "packet-hash"},
                        "test-only-key", max_tokens=32768, usage_path=usage, sleep_fn=lambda _: None)
    rows = [json.loads(line) for line in usage.open(encoding="utf-8")]
    assert len(calls) == len(rows) == 1
    assert calls[0]["reasoning_effort"] == "max"
    assert calls[0]["max_tokens"] == 32768
    assert rows[0]["finish_reason"] == "length"
    assert rows[0]["content_chars"] == 0
    assert "test-only-key" not in usage.read_text(encoding="utf-8")


def test_review_index_migrates_receipt_and_source_columns(tmp_path, monkeypatch):
    import sqlite3
    db_path = tmp_path / "index.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.executescript(mr.SCHEMA)
    monkeypatch.setattr(mr, "DB", db_path)
    row = {"packet_id": "MR-A:x", "sample_id": "A:x", "pool": "A", "instance_id": "x",
           "review_kind": "segment", "reviewer_model": "test", "reviewer_provider": "test",
           "reviewer_effort": "max", "verdict": "KEEP", "label_json": "[]", "notes": "",
           "submission_sha256": "submission", "packet_sha256": "packet", "created_at": "time",
           "review_config_sha256": "config", "trajectory_sha256": "trajectory"}
    mr.register([row])
    mr.register([row])
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*), review_config_sha256, trajectory_sha256 FROM pool_model_review").fetchone() == (1, "config", "trajectory")
