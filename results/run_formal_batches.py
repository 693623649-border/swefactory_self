"""Rolling large-batch generation with prefetch, disk guard, lock protection and resume.

Production safeguards:
1. Cross-process OS file lock (msvcrt on Windows, fcntl on Unix), auto-released on exit;
2. Per-run UUID + frozen JSON receipt (args, config sha256, initial counts), independent logs;
3. --target-total (default 2000) using physical newline iteration on trajectories.jsonl
   (first iid wins; strictly no str.splitlines() to avoid U+2028 issues; results.jsonl excluded);
4. Docker pull timeout + retry (up to 2 times); only explicit manifest/tag unknown is permanent missing;
5. Factory nonzero exit or 0-step/0-assistant turn batch causes immediate nonzero orchestrator exit;
   real test failures (score=0, success=False) with steps > 0 are treated as normal failures;
6. Prefetch worker joins cleanly on exit and propagates exceptions to main thread;
7. No global container prune; only rmi non-baseline tags pulled in this run without -f;
8. Default disk reserve 10.0 GiB, batch=15, pull-parallel=15, max-concurrent=6.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "pool_a_formal"
TASKS = FORMAL / "tasks_formal.jsonl"
REPO = "aweaiteam/scaleswe"
MISSING_FILE = FORMAL / "missing_images.json"
RESERVE_GB = 10.0
EST_IMAGE_GB = 0.85
LOCK_FILE = FORMAL / ".orchestrator.lock"

PERMANENT_MISSING_KEYWORDS = (
    "manifest unknown",
    "repository does not exist",
    "tag unknown",
)


class ProcessLock:
    """Cross-process OS file lock using Windows msvcrt or Unix fcntl."""

    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.fp = None

    def acquire(self) -> bool:
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fp = open(self.lock_file, "a+b")
            self.fp.seek(0)  # every process must lock the same byte, not append EOF
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fp.seek(0)
            self.fp.truncate()
            self.fp.write(f"pid={os.getpid()}\n".encode("utf-8"))
            self.fp.flush()
            return True
        except (OSError, IOError):
            if self.fp:
                try:
                    self.fp.close()
                except Exception:
                    pass
                self.fp = None
            return False

    def release(self) -> None:
        if self.fp:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self.fp.seek(0)
                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self.fp.close()
            except Exception:
                pass
            self.fp = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Another orchestrator instance is already running (lock held at {self.lock_file})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class PrefetchWorker:
    """Background worker for prefetching images, joining and bubbling exceptions."""

    def __init__(self, target, args=()):
        self.target = target
        self.args = args
        self.exc: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=False)

    def _run(self) -> None:
        try:
            self.target(*self.args)
        except Exception as exc:
            self.exc = exc

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        if self.thread.is_alive():
            self.thread.join()
        if self.exc is not None:
            raise self.exc


def sh(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def free_gb(path: str = "C:\\") -> float:
    try:
        return shutil.disk_usage(path).free / 2**30
    except Exception:
        return 0.0


def local_ids() -> set[str]:
    out = sh(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    if out.returncode != 0:
        raise RuntimeError("docker images failed: " + out.stderr[:200])
    return {ln.strip().split(":", 1)[1] for ln in out.stdout.splitlines()
            if ln.startswith(REPO + ":")}


def load_missing() -> set[str]:
    if MISSING_FILE.exists():
        try:
            return set(json.loads(MISSING_FILE.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_missing(missing: set[str]) -> None:
    MISSING_FILE.write_text(json.dumps(sorted(missing), indent=1), encoding="utf-8")


def load_unique_trajectories(formal_dir: Path) -> set[str]:
    """Read instance_ids with completed trajectories from all batch_*/**/trajectories.jsonl.

    Uses physical newline iterator (for line in fp) to avoid Unicode line separators
    (U+2028 / U+2029) splitting bug caused by str.splitlines().
    First iid wins. results.jsonl is deliberately NOT used.
    """
    unique_ids: set[str] = set()
    for f in formal_dir.glob("batch_*/**/trajectories.jsonl"):
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        iid = record.get("instance_id")
                        if iid and iid not in unique_ids:
                            unique_ids.add(iid)
                    except Exception:
                        continue
        except Exception:
            continue
    return unique_ids


def is_permanent_missing(err_text: str) -> bool:
    """Only return True if manifest/tag is explicitly missing from repository."""
    low = err_text.lower()
    for kw in PERMANENT_MISSING_KEYWORDS:
        if kw in low:
            return True
    if "manifest for" in low and "not found" in low:
        return True
    if "not found" in low and any(term in low for term in ("manifest", "tag", "repository")):
        return True
    return False


def pull_single_image(tag: str, timeout: int = 900, max_retries: int = 2) -> tuple[bool, bool, str]:
    """Pull an image with retries on transient errors.

    Returns: (success: bool, is_permanent_missing: bool, error_msg: str)
    """
    cmd = ["docker", "pull", f"{REPO}:{tag}"]
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                return True, False, ""
            combined = (proc.stderr or "") + " " + (proc.stdout or "")
            last_err = combined.strip()
            if is_permanent_missing(combined):
                return False, True, last_err
        except subprocess.TimeoutExpired:
            last_err = f"docker pull timeout after {timeout}s"
        except Exception as exc:
            last_err = f"docker pull exception: {exc}"

        if attempt < max_retries:
            time.sleep(1)

    return False, False, last_err


def pull_many(ids: list[str], parallel: int, missing: set[str],
              run_pulled_tags: set[str] | None = None,
              pull_failure_log: Path | None = None) -> list[str]:
    """Pull images in parallel; record permanently-missing tags once.

    Transient errors are retried up to 2 times and NEVER marked as missing.
    """
    to_pull = [i for i in ids if i not in missing]
    pulled: list[str] = []
    failures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel)) as executor:
        future_to_tag = {executor.submit(pull_single_image, tag): tag for tag in to_pull}
        for future in concurrent.futures.as_completed(future_to_tag):
            tag = future_to_tag[future]
            try:
                success, perm_missing, err_msg = future.result()
            except Exception as exc:
                success, perm_missing, err_msg = False, False, str(exc)

            if success:
                pulled.append(tag)
                if run_pulled_tags is not None:
                    run_pulled_tags.add(tag)
            elif perm_missing:
                missing.add(tag)
                print(f"  PERMANENT MISSING {tag}: {err_msg[:120]}", flush=True)
            else:
                failures.append({"tag": tag, "error": err_msg[:200], "timestamp": time.time()})
                print(f"  TRANSIENT PULL FAILURE {tag} (not marked missing): {err_msg[:120]}", flush=True)

    save_missing(missing)
    if failures and pull_failure_log:
        try:
            with pull_failure_log.open("a", encoding="utf-8") as f:
                for fail in failures:
                    f.write(json.dumps(fail) + "\n")
        except Exception:
            pass
    return pulled


def prune_run_images(run_pulled_tags: set[str], baseline: set[str], keep: set[str]) -> None:
    """Only rmi images pulled in this run that are not in baseline and not in keep.

    No force flag (-f), never touch containers or volumes, retain baseline.
    """
    candidates = [tag for tag in list(run_pulled_tags) if tag not in baseline and tag not in keep]
    for tag in candidates:
        res = sh(["docker", "rmi", f"{REPO}:{tag}"], timeout=180)
        if res.returncode == 0:
            run_pulled_tags.discard(tag)
        else:
            print(f"  docker rmi {tag} skipped: {res.stderr[:80].strip()}", flush=True)


def check_batch_infra_failure(out_dir: Path, attempted_count: int) -> None:
    """Halt with nonzero exit if the batch executed 0 steps and 0 assistant calls.

    Real test failures (score=0, success=False) with steps > 0 are valid evaluations.
    """
    if attempted_count == 0:
        return
    total_steps = 0
    total_assistant_calls = 0
    trajectories_found = 0

    for f in out_dir.glob("*/trajectories.jsonl"):
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        trajectories_found += 1
                        stats = record.get("stats") or {}
                        total_steps += stats.get("steps", 0) or 0
                        total_assistant_calls += sum(m.get("role") == "assistant" for m in record.get("messages") or [])
                        traj = record.get("trajectory", [])
                        if not stats.get("steps") and traj:
                            total_steps += len(traj)
                    except Exception:
                        pass
        except Exception:
            pass

    if total_steps == 0 and total_assistant_calls == 0:
        raise RuntimeError(
            f"INFRA FAILURE: batch finished with 0 total steps and 0 assistant calls "
            f"across {attempted_count} attempted instances (trajectories: {trajectories_found})."
        )


def run_batch(ids: list[str], index: int, max_concurrent: int = 6, run_id: str = "") -> tuple[int, str]:
    """Execute a single batch of instances and record isolated logs."""
    suffix = f"_{run_id}" if run_id else ""
    out_dir = FORMAL / f"batch{suffix}_{index:03d}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.with_suffix(".ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")

    cmd = [sys.executable, "-X", "utf8", str(ROOT / "results" / "run_factory.py"),
           "--data-file", str(TASKS), "--config", "configs/tasks/dsh_scaleswe.yaml",
           "--llm-config", "configs/llm/local_deepthink.yaml", "--mode", "batch",
           "--instance-ids", *ids, "--max-steps", "60",
           "--max-concurrent", str(max_concurrent),
           "--output", str(out_dir)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")

    log_content = (proc.stdout or "") + (proc.stderr or "")
    (out_dir / f"batch_{index:03d}.log").write_text(log_content, encoding="utf-8")
    log_name = f"batch_{run_id}_{index:03d}.log" if run_id else f"batch_{index:03d}_{int(started)}.log"
    (out_dir.parent / log_name).write_text(log_content, encoding="utf-8")

    if proc.returncode != 0:
        err_tail = proc.stderr[-1000:] if proc.stderr else proc.stdout[-1000:]
        raise RuntimeError(f"run_factory failed with nonzero returncode {proc.returncode} in batch {index:03d}:\n{err_tail}")

    check_batch_infra_failure(out_dir, len(ids))

    accepted = 0
    for f in out_dir.glob("*/results.jsonl"):
        try:
            with f.open("r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    line = line.strip()
                    if line and json.loads(line).get("success"):
                        accepted += 1
        except Exception:
            pass

    print(f"batch {index:03d}: {accepted}/{len(ids)} accepted in "
          f"{time.time()-started:.0f}s -> {out_dir.name}", flush=True)
    return accepted, str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--pull-parallel", type=int, default=15)
    parser.add_argument("--max-concurrent", type=int, default=6,
                        help="instances the factory executes at once (peak RAM/CPU knob)")
    parser.add_argument("--target-total", type=int, default=2000,
                        help="stop once total unique trajectories in FORMAL reaches this number")
    parser.add_argument("--reserve-gb", type=float, default=RESERVE_GB,
                        help="disk reserve threshold in GiB")
    args = parser.parse_args()

    # 1) Acquire cross-process OS file lock
    lock = ProcessLock(LOCK_FILE)
    if not lock.acquire():
        sys.exit(f"ABORT: Another orchestrator instance is active (locked at {LOCK_FILE}).")

    try:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_log_file = FORMAL / f"orchestrator_{run_id}.log"
        receipt_file = FORMAL / f"run_receipt_{run_id}.json"
        pull_fail_log = FORMAL / f"pull_failures_{run_id}.jsonl"

        def log(msg: str) -> None:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            try:
                with run_log_file.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        all_ids: list[str] = []
        if TASKS.exists():
            with TASKS.open("r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        try:
                            all_ids.append(json.loads(line)["instance_id"])
                        except Exception:
                            pass

        missing = load_missing()
        done = load_unique_trajectories(FORMAL)

        receipt_data = {
            "run_id": run_id,
            "created_at": time.time(),
            "args": vars(args),
            "configs_sha256": {
                "task_config": file_sha256(ROOT / "configs" / "tasks" / "dsh_scaleswe.yaml"),
                "llm_config": file_sha256(ROOT / "configs" / "llm" / "local_deepthink.yaml"),
                "producer": file_sha256(Path(__file__)),
                "tasks": file_sha256(TASKS),
            },
            "initial_counts": {
                "total_task_ids": len(all_ids),
                "done_unique_trajectories": len(done),
                "known_missing": len(missing),
                "target_total": args.target_total,
                "free_gb": free_gb(),
            },
        }
        receipt_file.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

        log(f"Started run {run_id} | target_total={args.target_total} | existing_unique_trajectories={len(done)} | free={free_gb():.1f} GB")

        if len(done) >= args.target_total:
            log(f"Target total {args.target_total} already satisfied ({len(done)} >= {args.target_total}). Exiting.")
            return

        pending = [i for i in all_ids if i not in done and i not in missing]
        log(f"selection={len(all_ids)} done={len(done)} pending={len(pending)} known_missing={len(missing)}")

        batches = [pending[i:i + args.batch] for i in range(0, len(pending), args.batch)]
        if args.max_batches:
            batches = batches[: args.max_batches]

        baseline = local_ids()
        log(f"baseline images kept: {len(baseline)}")

        ledger = FORMAL / "batch_ledger.jsonl"
        run_pulled_tags: set[str] = set()
        prefetch_worker: PrefetchWorker | None = None
        prefetched_ids: set[str] = set()

        try:
            for index, ids in enumerate(batches, start=1):
                # Check if target already reached
                if len(done) >= args.target_total:
                    log(f"Target total {args.target_total} reached before batch {index:03d}. Stopping.")
                    break

                ids = [i for i in ids if i not in missing and i not in done]
                ids = ids[:max(0, args.target_total - len(done))]
                if not ids:
                    with ledger.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"batch": index, "run_id": run_id, "attempted": [],
                                            "skipped_all": True, "timestamp": time.time()}) + "\n")
                    continue

                if prefetch_worker is not None:
                    prefetch_worker.join()
                    prefetch_worker = None

                need = [i for i in ids if i not in local_ids()]
                required = args.reserve_gb + EST_IMAGE_GB * len(need)
                if free_gb() < required:
                    prune_run_images(run_pulled_tags, baseline, prefetched_ids)
                    if free_gb() < required:
                        log(f"STOP: disk guard tripped (free {free_gb():.1f} GB < required {required:.1f} GB)")
                        break

                if need:
                    log(f"batch {index:03d}: pulling {len(need)} images")
                    pull_many(need, args.pull_parallel, missing, run_pulled_tags, pull_fail_log)

                available = [i for i in ids if i in local_ids()]
                if not available:
                    continue

                if index < len(batches) and (len(done) + len(available)) < args.target_total:
                    nxt = [i for i in batches[index] if i not in missing and i not in done]
                    prefetched_ids = set(nxt)

                    def _prefetch_task(seq=nxt):
                        log(f"prefetch: {len(seq)} images for the next batch")
                        pull_many(seq, args.pull_parallel, missing, run_pulled_tags, pull_fail_log)

                    prefetch_worker = PrefetchWorker(target=_prefetch_task)
                    prefetch_worker.start()

                accepted, run_dir = run_batch(available, index, args.max_concurrent, run_id)

                # Incrementally track new completed trajectories from this batch
                for f in Path(run_dir).glob("*/trajectories.jsonl"):
                    try:
                        with f.open("r", encoding="utf-8", errors="replace") as fp:
                            for line in fp:
                                line = line.strip()
                                if line:
                                    try:
                                        rec = json.loads(line)
                                        iid = rec.get("instance_id")
                                        if iid:
                                            done.add(iid)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                with ledger.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "batch": index,
                        "run_id": run_id,
                        "attempted": available,
                        "skipped": [i for i in ids if i not in available],
                        "accepted": accepted,
                        "run_dir": run_dir,
                        "total_unique_trajectories": len(done),
                        "timestamp": time.time(),
                    }) + "\n")

                try:
                    prune_run_images(run_pulled_tags, baseline, prefetched_ids)
                except Exception as exc:
                    log(f"prune warning: {exc}")

                log(f"after batch {index:03d}: done_unique={len(done)}/{args.target_total} | free {free_gb():.1f} GB")

                if len(done) >= args.target_total:
                    log(f"Target total {args.target_total} achieved ({len(done)} unique trajectories). Exiting loop.")
                    break
        finally:
            if prefetch_worker is not None:
                log("Joining active prefetch worker before exit...")
                prefetch_worker.join()
                prefetch_worker = None

        if len(done) < args.target_total and not args.max_batches:
            raise RuntimeError(f"Production stopped before target: {len(done)}/{args.target_total}; inspect run log and pull receipts")
        log("orchestrator loop finished successfully")
    finally:
        lock.release()


if __name__ == "__main__":
    main()
