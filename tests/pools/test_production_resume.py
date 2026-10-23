"""Tests for production resume and orchestrator safeguards in results/run_formal_batches.py.

All tests use mocks or temporary files; no live network or Docker calls are made.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
sys.path.insert(0, str(RESULTS_DIR))

import run_formal_batches as rfb  # noqa: E402


# 1. Cross-process OS file lock
def test_process_lock_mutual_exclusion_and_release(tmp_path: Path):
    lock_file = tmp_path / ".orchestrator.lock"
    lock1 = rfb.ProcessLock(lock_file)
    lock2 = rfb.ProcessLock(lock_file)

    assert lock1.acquire() is True
    # Second instance must fail to acquire
    assert lock2.acquire() is False

    # Release lock1
    lock1.release()

    # Now lock2 should be able to acquire
    assert lock2.acquire() is True
    lock2.release()


# 2. Per-run UUID and frozen JSON receipt
def test_receipt_structure_and_sha(tmp_path: Path, monkeypatch):
    dummy_task_cfg = tmp_path / "task.yaml"
    dummy_task_cfg.write_text("task: test", encoding="utf-8")
    dummy_llm_cfg = tmp_path / "llm.yaml"
    dummy_llm_cfg.write_text("model: test-model", encoding="utf-8")

    monkeypatch.setattr(rfb, "ROOT", tmp_path)
    monkeypatch.setattr(rfb, "free_gb", lambda *_: 55.5)

    receipt_file = tmp_path / "run_receipt_test.json"
    args_dict = {"batch": 15, "target_total": 2000, "reserve_gb": 10.0}
    counts = {"total_task_ids": 100, "done_unique_trajectories": 10, "known_missing": 2}

    receipt_data = {
        "run_id": "run_test123",
        "args": args_dict,
        "configs_sha256": {
            "task_config": rfb.file_sha256(dummy_task_cfg),
            "llm_config": rfb.file_sha256(dummy_llm_cfg),
        },
        "initial_counts": counts,
    }
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")

    loaded = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "run_test123"
    assert loaded["args"]["target_total"] == 2000
    assert len(loaded["configs_sha256"]["task_config"]) == 64
    assert len(loaded["configs_sha256"]["llm_config"]) == 64


# 3. Trajectory reading: physical lines, U+2028 safety, first iid wins, results.jsonl excluded
def test_unique_trajectories_physical_lines_u2028(tmp_path: Path):
    batch_dir = tmp_path / "batch_001_1000" / "agent_sub"
    batch_dir.mkdir(parents=True)

    # Line 1 contains U+2028 (LINE SEPARATOR) which breaks str.splitlines()
    u2028_char = "\u2028"
    line1 = json.dumps({"instance_id": "inst_1", "patch": f"diff --git a/x b/x{u2028_char}+code\n"}, ensure_ascii=False)
    # Line 2 duplicate inst_1 (first wins)
    line2 = json.dumps({"instance_id": "inst_1", "patch": "diff2"})
    # Line 3 distinct inst_2
    line3 = json.dumps({"instance_id": "inst_2", "patch": "diff3"})

    traj_file = batch_dir / "trajectories.jsonl"
    with traj_file.open("w", encoding="utf-8") as f:
        f.write(f"{line1}\n{line2}\n{line3}\n")

    # Also place a results.jsonl with a fake instance that should NOT be counted as trajectory
    results_file = batch_dir / "results.jsonl"
    results_file.write_text(json.dumps({"instance_id": "fake_results_only"}) + "\n", encoding="utf-8")

    unique = rfb.load_unique_trajectories(tmp_path)
    assert "inst_1" in unique
    assert "inst_2" in unique
    assert "fake_results_only" not in unique
    assert len(unique) == 2


# 4. Docker pull: permanent missing vs transient retry (not marked missing)
def test_docker_pull_permanent_missing():
    err_manifest = "Error response from daemon: manifest unknown: manifest unknown"
    assert rfb.is_permanent_missing(err_manifest) is True

    err_repo = "Error response from daemon: repository does not exist or may require 'docker login'"
    assert rfb.is_permanent_missing(err_repo) is True

    err_not_found = "Error response from daemon: manifest for aweaiteam/scaleswe:tag1 not found: manifest unknown"
    assert rfb.is_permanent_missing(err_not_found) is True

    # Transient network / timeout errors must NOT be permanent
    assert rfb.is_permanent_missing("net/http: TLS handshake timeout") is False
    assert rfb.is_permanent_missing("toomanyrequests: You have reached your pull rate limit (429)") is False
    assert rfb.is_permanent_missing("i/o timeout") is False


def test_pull_single_image_transient_retry(monkeypatch):
    attempts = 0

    def mock_subprocess_run(cmd, **kwargs):
        nonlocal attempts
        attempts += 1
        # Fails with 429
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="429 Too Many Requests")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    monkeypatch.setattr(rfb.time, "sleep", lambda _: None)

    success, is_perm, err = rfb.pull_single_image("test_tag", timeout=10, max_retries=2)
    assert success is False
    assert is_perm is False  # NOT permanent missing!
    assert "429" in err
    assert attempts == 3  # initial + 2 retries


def test_pull_many_does_not_mark_transient_as_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rfb, "MISSING_FILE", tmp_path / "missing_images.json")
    failure_log = tmp_path / "pull_failures.jsonl"

    def mock_pull_single(tag, **_):
        if tag == "perm_tag":
            return False, True, "manifest unknown"
        elif tag == "transient_tag":
            return False, False, "429 rate limit"
        else:
            return True, False, ""

    monkeypatch.setattr(rfb, "pull_single_image", mock_pull_single)

    missing: set[str] = set()
    run_pulled: set[str] = set()
    pulled = rfb.pull_many(["good_tag", "perm_tag", "transient_tag"], parallel=2,
                          missing=missing, run_pulled_tags=run_pulled,
                          pull_failure_log=failure_log)

    assert pulled == ["good_tag"]
    assert "good_tag" in run_pulled
    assert "perm_tag" in missing
    # transient_tag must NOT be in missing
    assert "transient_tag" not in missing
    assert failure_log.exists()
    assert "transient_tag" in failure_log.read_text(encoding="utf-8")


# 5. Factory nonzero exit raises RuntimeError
def test_factory_nonzero_exit_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rfb, "FORMAL", tmp_path)
    monkeypatch.setattr(rfb, "ROOT", tmp_path)
    monkeypatch.setattr(rfb, "TASKS", tmp_path / "tasks.jsonl")

    def mock_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=2, stdout="", stderr="ImportError: missing dependency")

    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(RuntimeError, match="nonzero returncode 2"):
        rfb.run_batch(["id_1"], index=1, run_id="test_run")


# 6. Infra failure detection (0 steps/0 assistant) vs real test 0 success (steps > 0)
def test_check_batch_infra_failure_triggers(tmp_path: Path):
    out_dir = tmp_path / "batch_out"
    sub = out_dir / "sub"
    sub.mkdir(parents=True)

    # Empty trajectory or 0 steps
    traj = sub / "trajectories.jsonl"
    traj.write_text(json.dumps({"instance_id": "i1", "stats": {"steps": 0, "llm_calls": 0}, "trajectory": []}) + "\n",
                    encoding="utf-8")

    with pytest.raises(RuntimeError, match="INFRA FAILURE: batch finished with 0 total steps"):
        rfb.check_batch_infra_failure(out_dir, attempted_count=1)


def test_real_test_zero_success_not_infra_failure(tmp_path: Path):
    out_dir = tmp_path / "batch_out"
    sub = out_dir / "sub"
    sub.mkdir(parents=True)

    # 15 steps were executed, agent produced tool calls and patch, but test failed
    traj = sub / "trajectories.jsonl"
    traj.write_text(json.dumps({
        "instance_id": "i1",
        "success": False,
        "score": 0.0,
        "stats": {"steps": 15, "llm_calls": 15},
        "trajectory": [{"step": 1}] * 15,
    }) + "\n", encoding="utf-8")

    # Should NOT raise!
    rfb.check_batch_infra_failure(out_dir, attempted_count=1)


# 7. Prefetch worker joins cleanly and bubbles exceptions
def test_prefetch_worker_propagates_exception():
    def failing_target():
        raise ValueError("Prefetch network crash")

    worker = rfb.PrefetchWorker(target=failing_target)
    worker.start()

    with pytest.raises(ValueError, match="Prefetch network crash"):
        worker.join()


# 8. Container prune removed; only rmi non-baseline tags pulled in this run without -f
def test_prune_run_images_no_force_no_baseline(monkeypatch):
    rmi_calls = []

    def mock_sh(cmd, **kwargs):
        rmi_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rfb, "sh", mock_sh)

    baseline = {"base_tag1", "base_tag2"}
    pulled_in_run = {"base_tag1", "run_tag_old", "run_tag_prefetched"}
    keep = {"run_tag_prefetched"}

    rfb.prune_run_images(pulled_in_run, baseline=baseline, keep=keep)

    # Only run_tag_old should be removed
    assert len(rmi_calls) == 1
    call = rmi_calls[0]
    # Verify no "-f" flag in command
    assert "-f" not in call
    # Verify no container prune
    assert "container" not in call
    assert call[1] == "rmi"
    assert "run_tag_old" in call[2]
    # Removed from set
    assert "run_tag_old" not in pulled_in_run
    assert "run_tag_prefetched" in pulled_in_run
