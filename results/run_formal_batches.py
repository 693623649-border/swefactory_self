"""Rolling large-batch generation with prefetch, disk guard and persistent skips.

Fixes over the previous orchestrator (all found from real run logs):
1. images whose registry tag is missing are remembered in missing_images.json and
   never re-pulled, so batches stop paying repeated timeout costs;
2. pruning is whitelist-based and wrapped so an exception cannot leak images into the
   small Docker volume;
3. a disk guard aborts cleanly when free space is below the reserve instead of
   filling the volume;
4. the next batch's images are pulled in a background thread while the current batch
   runs, overlapping the two dominant costs.

Usage: python results/run_formal_batches.py [--batch 15] [--max-batches N]
DEEPSEEK_API_KEY must come from the environment.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "pool_a_formal"
TASKS = FORMAL / "tasks_formal.jsonl"
REPO = "aweaiteam/scaleswe"
MISSING_FILE = FORMAL / "missing_images.json"
RESERVE_GB = 5.0  # lowered 8.0 -> 5.0 after pruning 29 GB of finished-task images (2026-10-23)
EST_IMAGE_GB = 0.85  # measured median 0.50 GB, mean 0.85 GB over 20 samples


def sh(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def free_gb(path: str = "C:\\") -> float:
    return shutil.disk_usage(path).free / 2**30


def local_ids() -> set[str]:
    out = sh(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    if out.returncode != 0:
        raise RuntimeError("docker images failed: " + out.stderr[:200])
    return {ln.strip().split(":", 1)[1] for ln in out.stdout.splitlines()
            if ln.startswith(REPO + ":")}


def load_missing() -> set[str]:
    if MISSING_FILE.exists():
        return set(json.loads(MISSING_FILE.read_text(encoding="utf-8")))
    return set()


def save_missing(missing: set[str]) -> None:
    MISSING_FILE.write_text(json.dumps(sorted(missing), indent=1), encoding="utf-8")


def done_ids() -> set[str]:
    done: set[str] = set()
    for f in FORMAL.glob("batch_*/**/results.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["instance_id"])
    return done


def pull_many(ids: list[str], parallel: int, missing: set[str]) -> list[str]:
    """Pull images in parallel; record permanently-missing tags once."""
    to_pull = [i for i in ids if i not in missing]
    pulled: list[str] = []
    for j in range(0, len(to_pull), parallel):
        group = to_pull[j:j + parallel]
        procs = [(i, subprocess.Popen(["docker", "pull", f"{REPO}:{i}"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                 for i in group]
        for i, proc in procs:
            if proc.wait() == 0:
                pulled.append(i)
            else:
                missing.add(i)
                print(f"  MISSING {i}", flush=True)
        save_missing(missing)
    return pulled


def prune(baseline: set[str], keep: set[str]) -> None:
    sh(["docker", "container", "prune", "-f"], timeout=180)
    for tag in local_ids():
        if tag not in baseline and tag not in keep:
            sh(["docker", "rmi", "-f", f"{REPO}:{tag}"], timeout=180)


def run_batch(ids: list[str], index: int, max_concurrent: int = 6) -> tuple[int, str]:
    out_dir = FORMAL / f"batch_{index:03d}_{int(time.time())}"
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
    (out_dir.parent / f"batch_{index:03d}.log").write_text(
        (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    accepted = 0
    for f in out_dir.glob("*/results.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("success"):
                accepted += 1
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
    args = parser.parse_args()

    all_ids = [json.loads(l)["instance_id"]
               for l in TASKS.read_text(encoding="utf-8").splitlines()]
    missing = load_missing()
    done = done_ids()
    # Permanently missing registry tags stay out of the queue: they used to sit at the
    # head of `pending` and turn a smoke batch into a no-op.
    pending = [i for i in all_ids if i not in done and i not in missing]
    print(f"selection={len(all_ids)} done={len(done)} pending={len(pending)} "
          f"known_missing={len(missing)} free={free_gb():.1f} GB", flush=True)
    batches = [pending[i:i + args.batch] for i in range(0, len(pending), args.batch)]
    if args.max_batches:
        batches = batches[: args.max_batches]

    baseline = local_ids()
    print(f"baseline images kept: {len(baseline)}", flush=True)
    ledger = FORMAL / "batch_ledger.jsonl"
    prefetch: threading.Thread | None = None
    prefetched_ids: set[str] = set()

    for index, ids in enumerate(batches, start=1):
        ids = [i for i in ids if i not in missing]
        if not ids:
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"batch": index, "attempted": [],
                                    "skipped_all": True}) + "\n")
            continue
        if prefetch is not None:
            prefetch.join()
            prefetch = None
        need = [i for i in ids if i not in local_ids()]
        required = RESERVE_GB + EST_IMAGE_GB * len(need)
        if free_gb() < required:
            prune(baseline, prefetched_ids)
            if free_gb() < required:
                print(f"STOP: disk guard tripped (free {free_gb():.1f} GB < "
                      f"required {required:.1f} GB)", flush=True)
                break
        if need:
            print(f"batch {index:03d}: pulling {len(need)} images", flush=True)
            pull_many(need, args.pull_parallel, missing)
        available = [i for i in ids if i in local_ids()]
        if not available:
            continue
        if index < len(batches):
            nxt = [i for i in batches[index] if i not in missing]
            prefetched_ids = set(nxt)

            def _prefetch(seq=nxt):
                print(f"prefetch: {len(seq)} images for the next batch", flush=True)
                pull_many(seq, args.pull_parallel, missing)

            prefetch = threading.Thread(target=_prefetch, daemon=True)
            prefetch.start()
        accepted, run_dir = run_batch(available, index, args.max_concurrent)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"batch": index, "attempted": available,
                                "skipped": [i for i in ids if i not in available],
                                "accepted": accepted, "run_dir": run_dir}) + "\n")
        try:
            prune(baseline, prefetched_ids)
        except Exception as exc:  # cleanup must never abort the loop
            print(f"prune warning: {exc}", flush=True)
        print(f"after batch {index:03d}: free {free_gb():.1f} GB", flush=True)
    print("orchestrator loop finished", flush=True)


if __name__ == "__main__":
    main()
