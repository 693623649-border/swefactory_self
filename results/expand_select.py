"""Round-2 instance selection: expand the formal pool toward 2,000 trajectories.

Round 1 (results/audit_select.py) took 903 instances, one per family, and the
factory has since attempted 878 of them. The target is now 2,000 *trajectories*
(including rejected ones), i.e. about 1,122 more unique attempts, so this round
reuses the exact same static qualification as round 1 and adds rows from
families that already ran plus the families round 1 never reached.

Rules preserved from round 1:
  * python only, problem_statement >= 200 chars, parseable FAIL_TO_PASS and
    PASS_TO_PASS, parent_commit and image_url present;
  * every instance_id already attempted or already queued is excluded;
  * the 198 validation families stay validation-only;
  * near-duplicate problem statements inside a repo family are dropped;
  * no family may exceed --family-cap rows in total.

New rows are appended to results/pool_a_formal/tasks_formal.jsonl so the
existing orchestrator and admission exporter pick them up unchanged; the
selection itself is recorded in results/pool_a_formal/expansion_summary.json.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "scaleswe_raw"
FORMAL = ROOT / "results" / "pool_a_formal"
TRAJECTORY_GLOBS = [
    "results/pilot/*/trajectories.jsonl",
    "results/dsh_pilot/*/trajectories.jsonl",
    "results/dsh_think/*/trajectories.jsonl",
    "results/pool_a/*/trajectories.jsonl",
    "results/pool_a50/*/trajectories.jsonl",
    "results/pool_a_formal/batch_*/**/trajectories.jsonl",
    "results/contract_retry_v1/*/*/trajectories.jsonl",
]


def parse_list(value) -> list | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def norm_key(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", text)[:800].encode()).hexdigest()[:16]


def family_of(row: dict) -> str:
    return f"{row.get('user', '')}/{row.get('repo', '')}"


def attempted_ids() -> set[str]:
    used: set[str] = set()
    for pattern in TRAJECTORY_GLOBS:
        for f in ROOT.glob(pattern):
            for line in f.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                try:
                    used.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return used


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for shard in sorted(RAW.glob("*.parquet")):
        table = pq.read_table(shard)
        for batch in table.to_batches(max_chunksize=512):
            rows.extend(batch.to_pylist())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1170)
    parser.add_argument("--family-cap", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qualification = json.loads((FORMAL / "qualification.json").read_text(encoding="utf-8"))
    validation = set(qualification["validation_families"])
    queued = [json.loads(l)["instance_id"]
              for l in (FORMAL / "tasks_formal.jsonl").read_text(encoding="utf-8").split("\n")
              if l.strip()]
    ledger_ids = {json.loads(l)["instance_id"]
                  for l in (FORMAL / "admission_ledger.jsonl").read_text(encoding="utf-8").split("\n")
                  if l.strip()}
    missing_images = set(json.loads((FORMAL / "missing_images.json").read_text(encoding="utf-8")))
    used = attempted_ids() | set(queued) | ledger_ids | missing_images
    print(f"queued={len(queued)} ledger={len(ledger_ids)} trajectories={len(attempted_ids())} "
          f"missing_images={len(missing_images)} used_total={len(used)}", flush=True)

    rows = load_rows()
    print(f"raw_rows={len(rows)}", flush=True)
    kept: list[dict] = []
    rejected = Counter()
    for r in rows:
        family = family_of(r)
        if not r.get("parent_commit") or not r.get("image_url"):
            rejected["missing_commit_or_image"] += 1
        elif (r.get("language") or "python") != "python":
            rejected["non_python"] += 1
        elif len(r.get("problem_statement") or "") < 200:
            rejected["short_problem"] += 1
        elif not parse_list(r.get("FAIL_TO_PASS")):
            rejected["f2p_missing"] += 1
        elif parse_list(r.get("PASS_TO_PASS")) is None:
            rejected["p2p_unparsable"] += 1
        elif family in validation:
            rejected["validation_family"] += 1
        else:
            kept.append(r)

    # Near-duplicate detection across everything (queue + candidates) so round-2
    # rows never repeat a problem already used in round 1.
    seen: dict[str, set[str]] = defaultdict(set)
    by_id = {r["instance_id"]: r for r in rows}
    for iid in queued:
        row = by_id.get(iid)
        if row is not None:
            seen[family_of(row)].add(norm_key(row.get("problem_statement") or ""))

    candidates: list[dict] = []
    for r in kept:
        iid = r["instance_id"]
        if iid in used:
            rejected["already_used"] += 1
            continue
        family = family_of(r)
        key = norm_key(r.get("problem_statement") or "")
        if key in seen[family]:
            rejected["near_dup_problem"] += 1
            continue
        seen[family].add(key)
        candidates.append(r)

    selected_per_family = Counter(family_of(by_id[i]) for i in queued if i in by_id)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        family = family_of(r)
        room = args.family_cap - selected_per_family[family]
        if room > len(by_family[family]):
            by_family[family].append(r)
    families = sorted(by_family, key=lambda f: (-len(by_family[f]), f))
    print(f"kept={len(kept)} candidates={len(candidates)} families_with_room={len(families)} "
          f"max_per_family={max((len(v) for v in by_family.values()), default=0)}", flush=True)

    selected: list[dict] = []
    index = 0
    while len(selected) < args.target:
        progressed = False
        for family in families:
            rows_f = by_family[family]
            if index < len(rows_f):
                selected.append(rows_f[index])
                progressed = True
                if len(selected) >= args.target:
                    break
        if not progressed:
            break
        index += 1

    fam_counts = Counter(family_of(r) for r in selected)
    summary = {
        "target": args.target,
        "selected": len(selected),
        "selected_families": len(fam_counts),
        "candidates": len(candidates),
        "used_excluded": len(used),
        "validation_families": len(validation),
        "family_cap": args.family_cap,
        "rejected": dict(rejected),
        "top_families": fam_counts.most_common(15),
        "already_queued": len(queued),
        "note": "appended to tasks_formal.jsonl; admission ledger and export pick them up unchanged",
    }
    print(json.dumps({k: summary[k] for k in ("target", "selected", "selected_families",
                                              "candidates", "rejected")}, ensure_ascii=False))
    if args.dry_run:
        return

    (FORMAL / "expansion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (FORMAL / "tasks_formal.jsonl").open("a", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (FORMAL / "expansion_ids.json").write_text(
        json.dumps([r["instance_id"] for r in selected], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"appended {len(selected)} rows -> tasks_formal.jsonl", flush=True)


if __name__ == "__main__":
    main()
