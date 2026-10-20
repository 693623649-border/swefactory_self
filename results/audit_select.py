"""Static qualification audit and stratified selection over Scale-SWE parquet.

Reads data/scaleswe_raw/*.parquet (20,181 rows), applies schema/family filters,
excludes the 77 instance_ids already consumed by earlier runs, drops near-duplicate
problem statements inside a repo family, then selects up to --target instances by
round-robin across families so every batch spans many repos.

Outputs:
  results/pool_a_formal/tasks_formal.jsonl   selected rows (factory-ready)
  results/pool_a_formal/qualification.json   stats, tags and family table

Static checks only; contract_ambiguous / leak_risk / infra_risk are runtime or
post-hoc tags and stay empty here on purpose.
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
OUT = ROOT / "results" / "pool_a_formal"
USED_GLOBS = [
    "results/pilot/*/trajectories.jsonl",
    "results/dsh_pilot/*/trajectories.jsonl",
    "results/dsh_think/*/trajectories.jsonl",
    "results/pool_a/*/trajectories.jsonl",
    "results/pool_a50/*/trajectories.jsonl",
    "results/contract_retry_v1/*/*/trajectories.jsonl",
]


def used_ids() -> set[str]:
    used: set[str] = set()
    for pattern in USED_GLOBS:
        for f in glob.glob(str(ROOT / pattern)):
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    used.add(json.loads(line)["instance_id"])
    return used


def parse_list(value) -> list | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            return None
    return None


def norm_key(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", "", text)[:800].encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=903)
    parser.add_argument("--family-cap", type=int, default=40)
    parser.add_argument("--val-family-ratio", type=int, default=6, help="every Nth family is validation-only")
    args = parser.parse_args()

    used = used_ids()
    rows = []
    cols = None
    for shard in sorted(RAW.glob("*.parquet")):
        table = pq.read_table(shard)
        cols = cols or table.column_names
        for batch in table.to_batches(max_chunksize=512):
            rows.extend(batch.to_pylist())
    print(f"rows={len(rows)} used_excluded={len(used)} columns={cols}")

    kept, rejected = [], Counter()
    seen_hash: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        iid = r.get("instance_id", "")
        f2p = parse_list(r.get("FAIL_TO_PASS"))
        p2p = parse_list(r.get("PASS_TO_PASS"))
        ps = r.get("problem_statement") or ""
        if not r.get("parent_commit") or not r.get("image_url"):
            rejected["missing_commit_or_image"] += 1
        elif (r.get("language") or "python") != "python":
            rejected["non_python"] += 1
        elif len(ps) < 200:
            rejected["short_problem"] += 1
        elif not f2p:
            rejected["f2p_missing"] += 1
        elif p2p is None:
            rejected["p2p_unparsable"] += 1
        elif iid in used:
            rejected["already_used"] += 1
        else:
            family = f"{r.get('user','')}/{r.get('repo','')}"
            key = norm_key(ps)
            if key in seen_hash[family]:
                rejected["near_dup_problem"] += 1
                continue
            seen_hash[family].add(key)
            kept.append(r)

    fam_count = Counter(f"{r.get('user','')}/{r.get('repo','')}" for r in kept)
    print(f"kept_after_filters={len(kept)} families={len(fam_count)}")
    print("top families:", fam_count.most_common(10))
    print("rejected:", dict(rejected))

    # family cap + round-robin interleave
    by_family = defaultdict(list)
    for r in kept:
        by_family[f"{r.get('user','')}/{r.get('repo','')}"].append(r)
    families = sorted(by_family, key=lambda f: -len(by_family[f]))
    val_families = set(families[:: args.val_family_ratio])
    train_families = [f for f in families if f not in val_families]
    for f in train_families:
        by_family[f] = by_family[f][: args.family_cap]
    selected: list = []
    idx = 0
    while len(selected) < args.target:
        progressed = False
        for f in train_families:
            if idx < len(by_family[f]):
                selected.append(by_family[f][idx])
                progressed = True
                if len(selected) >= args.target:
                    break
        if not progressed:
            break
        idx += 1

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tasks_formal.jsonl").open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sel_fam = Counter(f"{r.get('user','')}/{r.get('repo','')}" for r in selected)
    summary = {
        "total_rows": len(rows),
        "used_excluded": len(used),
        "kept_after_filters": len(kept),
        "rejected": dict(rejected),
        "families_total": len(fam_count),
        "family_cap": args.family_cap,
        "validation_families": sorted(val_families),
        "selected": len(selected),
        "selected_families": len(sel_fam),
        "top_selected_families": sel_fam.most_common(15),
        "static_tags_note": "contract_ambiguous/leak_risk/infra_risk are runtime or post-hoc",
    }
    (OUT / "qualification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("selected", "selected_families", "kept_after_filters")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
