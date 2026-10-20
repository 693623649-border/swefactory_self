"""Rolling large-batch generation over the audited Scale-SWE selection.

Design constraints (see plan v2):
- Docker vhdx lives on C: with ~50 GB free -> working set of ~15 images
  (avg 2.4 GB each) pulled per batch, removed after the batch unless pre-existing.
- One fresh independent attempt per instance; resumable by scanning existing
  results.jsonl under results/pool_a_formal/.
- Teacher/config unchanged from the 50-task receipt: deepseek-flash low-effort
  thinking, 60 steps, concurrency 6, original contract (no public-contract-v1).

Usage: python results/run_formal_batches.py [--batch 15] [--max-batches N]
DEEPSEEK_API_KEY must come from the environment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "pool_a_formal"
TASKS = FORMAL / "tasks_formal.jsonl"
REPO = "aweaiteam/scaleswe"


def sh(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8")


def local_images() -> set[str]:
    out = sh(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"])
    if out.returncode != 0:
        raise RuntimeError("docker images failed: " + out.stderr[:200])
    return {ln.strip() for ln in out.stdout.splitlines() if ln.startswith(REPO + ":")}


def done_ids() -> set[str]:
    done: set[str] = set()
    for f in FORMAL.glob("*/results.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["instance_id"])
    return done


def run_batch(ids: list[str], index: int) -> tuple[int, int, str]:
    out_dir = FORMAL / f"batch_{index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / f"batch_{index:03d}.ids.json").write_text(json.dumps(ids, indent=2),
                                                                encoding="utf-8")
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "results" / "run_factory.py"),
           "--data-file", str(TASKS),
           "--config", "configs/tasks/dsh_scaleswe.yaml",
           "--llm-config", "configs/llm/local_deepthink.yaml",
           "--mode", "batch", "--instance-ids", *ids,
           "--max-steps", "60", "--max-concurrent", "6",
           "--output", str(out_dir)]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    (out_dir.parent / f"batch_{index:03d}.log").write_text(
        (proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    accepted = failed = 0
    run_dir = ""
    for f in out_dir.glob("*/results.jsonl"):
        run_dir = str(f.parent)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            accepted += bool(row.get("success"))
            failed += not row.get("success") and not row.get("error")
    print(f"batch {index:03d}: {accepted}/{len(ids)} accepted in {time.time()-started:.0f}s -> {run_dir}",
          flush=True)
    return accepted, failed, run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = until selection exhausted")
    parser.add_argument("--pull-parallel", type=int, default=4)
    args = parser.parse_args()

    all_ids = [json.loads(l)["instance_id"]
               for l in TASKS.read_text(encoding="utf-8").splitlines()]
    done = done_ids()
    pending = [i for i in all_ids if i not in done]
    print(f"selection={len(all_ids)} done={len(done)} pending={len(pending)}", flush=True)
    batches = [pending[i:i + args.batch] for i in range(0, len(pending), args.batch)]
    if args.max_batches:
        batches = batches[: args.max_batches]

    ledger = FORMAL / "batch_ledger.jsonl"
    for index, ids in enumerate(batches, start=1):
        preexisting = local_images()
        pulled: list[str] = []
        to_pull = [f"{REPO}:{i}" for i in ids if f"{REPO}:{i}" not in preexisting]
        for j in range(0, len(to_pull), args.pull_parallel):
            group = to_pull[j:j + args.pull_parallel]
            procs = [subprocess.Popen(["docker", "pull", img], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL) for img in group]
            for img, p in zip(group, procs):
                if p.wait() == 0:
                    pulled.append(img)
                else:
                    print(f"  PULL FAIL {img}", flush=True)
        missing = [i for i in ids if f"{REPO}:{i}" not in local_images()]
        runnable = [i for i in ids if i not in missing]
        if missing:
            print(f"batch {index}: {len(missing)} images unavailable, running {len(runnable)}",
                  flush=True)
        if not runnable:
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"batch": index, "ids": ids, "skipped_all": True}) + "\n")
            continue
        accepted, failed, run_dir = run_batch(runnable, index)
        for img in pulled:  # reclaim C: disk; keep pre-existing images untouched
            sh(["docker", "rmi", "-f", img], timeout=300)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"batch": index, "attempted": runnable, "skipped": missing,
                                "accepted": accepted, "run_dir": run_dir}) + "\n")
    print("orchestrator loop finished", flush=True)


if __name__ == "__main__":
    main()
