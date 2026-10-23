"""Build the pool-A index database from the JSONL ledgers.

SQLite holds only index and audit rows; trajectories, ids+mask files and reports
stay on disk, referenced by path. Re-running is safe: every table is replaced from
the ledgers, so the database can never drift from the files it indexes.

Usage:
    python results/build_index_db.py            # build / refresh
    python results/build_index_db.py --report   # print the summary queries
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_isolation import (  # noqa: E402
    is_benchmark_family,
    load_protected_families,
)

FORMAL = Path(__file__).resolve().parents[1] / "results" / "pool_a_formal"
DB = FORMAL / "index.sqlite3"

SCHEMA = """
CREATE TABLE samples (
    instance_id     TEXT PRIMARY KEY,
    family          TEXT NOT NULL,
    batch           TEXT,
    admitted        INTEGER NOT NULL,
    rejected        TEXT,              -- JSON array of failed gate names
    success         INTEGER NOT NULL,  -- evaluator accepted
    f2p_passed      INTEGER,
    f2p_total       INTEGER,
    p2p_passed      INTEGER,
    p2p_total       INTEGER,
    node_evidence   TEXT,              -- junit_nodes | evaluator_acceptance
    patch_chars     INTEGER,
    files_changed   INTEGER,
    test_paths      TEXT,              -- JSON array
    test_runs       INTEGER,
    mutations       INTEGER,
    verified_after_last_edit INTEGER,
    steps           INTEGER,
    finish_reason   TEXT,
    seq_tokens      INTEGER,           -- rendered student-template length
    supervised_tokens INTEGER,         -- supervised assistant tokens
    length_gate     INTEGER,           -- 65536 at admission time
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    duration_s      REAL,
    tool_bash       INTEGER,
    tool_read       INTEGER,
    tool_edit       INTEGER,
    traj_path       TEXT,
    traj_sha256     TEXT
);

CREATE TABLE gates (
    instance_id TEXT NOT NULL REFERENCES samples(instance_id) ON DELETE CASCADE,
    gate        TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    PRIMARY KEY (instance_id, gate)
);

CREATE TABLE batches (
    batch        TEXT PRIMARY KEY,
    attempted    INTEGER,
    accepted     INTEGER,
    admitted     INTEGER,
    ran_after    TEXT
);

CREATE INDEX idx_samples_family   ON samples(family);
CREATE INDEX idx_samples_admitted ON samples(admitted);
CREATE INDEX idx_samples_batch    ON samples(batch);
CREATE INDEX idx_gates_gate       ON gates(gate, passed);

CREATE VIEW v_reject_reasons AS
SELECT value AS gate, COUNT(*) AS instances
FROM samples, json_each(samples.rejected)
WHERE samples.admitted = 0
GROUP BY value ORDER BY instances DESC;

CREATE VIEW v_needs_rerun AS
SELECT s.instance_id, s.family, s.batch, s.rejected
FROM samples s
WHERE s.admitted = 0
  AND s.rejected NOT LIKE '%infra_error%';

CREATE VIEW v_tool_usage AS
SELECT
    COUNT(*) AS total_samples,
    SUM(CASE WHEN admitted=1 THEN 1 ELSE 0 END) AS admitted_samples,
    SUM(tool_bash) AS total_bash,
    ROUND(AVG(tool_bash), 2) AS avg_bash_all,
    ROUND(AVG(CASE WHEN admitted=1 THEN tool_bash END), 2) AS avg_bash_adm,
    SUM(tool_read) AS total_read,
    ROUND(AVG(tool_read), 2) AS avg_read_all,
    ROUND(AVG(CASE WHEN admitted=1 THEN tool_read END), 2) AS avg_read_adm,
    SUM(tool_edit) AS total_edit,
    ROUND(AVG(tool_edit), 2) AS avg_edit_all,
    ROUND(AVG(CASE WHEN admitted=1 THEN tool_edit END), 2) AS avg_edit_adm
FROM samples;

CREATE VIEW v_step_brackets AS
SELECT
    CASE
        WHEN steps <= 10 THEN '00-10'
        WHEN steps <= 20 THEN '11-20'
        WHEN steps <= 30 THEN '21-30'
        WHEN steps <= 40 THEN '31-40'
        WHEN steps <= 50 THEN '41-50'
        ELSE '51-60'
    END AS bracket,
    COUNT(*) AS attempted,
    SUM(CASE WHEN admitted=1 THEN 1 ELSE 0 END) AS admitted,
    ROUND(100.0 * SUM(CASE WHEN admitted=1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS adm_pct,
    ROUND(AVG(prompt_tokens), 0) AS avg_prompt_tokens,
    ROUND(AVG(completion_tokens), 0) AS avg_completion_tokens,
    ROUND(AVG(CASE WHEN admitted=1 THEN seq_tokens END), 0) AS avg_adm_seq_tokens,
    ROUND(AVG(CASE WHEN admitted=1 THEN supervised_tokens END), 0) AS avg_adm_sup_tokens,
    SUM(CASE WHEN admitted=1 THEN tool_bash ELSE 0 END) AS adm_bash,
    SUM(CASE WHEN admitted=1 THEN tool_read ELSE 0 END) AS adm_read,
    SUM(CASE WHEN admitted=1 THEN tool_edit ELSE 0 END) AS adm_edit
FROM samples
GROUP BY bracket
ORDER BY bracket;
"""

GATE_NAMES = ("not_passed", "f2p_incomplete", "p2p_incomplete", "over_length",
              "test_paths_edited", "no_test_at_all", "empty_patch", "duplicate_patch",
              "infra_error", "validation_family", "benchmark_family", "family_duplicate", "unknown_task",
              "no_think", "empty_mask")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256(path.read_bytes()).hexdigest()


def trajectory_index() -> dict[str, Path]:
    """instance_id -> the trajectory file that contains it."""
    found: dict[str, Path] = {}
    for traj in sorted(FORMAL.glob("batch_*/**/trajectories.jsonl")):
        try:
            with traj.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        found.setdefault(json.loads(line)["instance_id"], traj)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            continue
    return found


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def build() -> dict:
    ledger = load_jsonl(FORMAL / "admission_ledger.jsonl")
    batch_rows = load_jsonl(FORMAL / "batch_ledger.jsonl")
    traj = trajectory_index()
    db = sqlite3.connect(DB)
    db.executescript("DROP VIEW IF EXISTS v_reject_reasons;"
                     "DROP VIEW IF EXISTS v_needs_rerun;"
                     "DROP VIEW IF EXISTS v_tool_usage;"
                     "DROP VIEW IF EXISTS v_step_brackets;"
                     "DROP TABLE IF EXISTS gates;"
                     "DROP TABLE IF EXISTS samples;"
                     "DROP TABLE IF EXISTS batches;")
    db.executescript(SCHEMA)

    for record in ledger:
        gates = record.get("rejected") or []
        admitted = bool(record.get("admitted"))
        passed = {name: int(name not in gates) for name in GATE_NAMES}
        if not record.get("success"):
            passed["not_passed"] = 0
        f2p = record.get("f2p") or {}
        p2p = record.get("p2p") or {}
        path = traj.get(record["instance_id"])
        db.execute(
            "INSERT OR REPLACE INTO samples VALUES ("
            "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record["instance_id"], record.get("family", ""), record.get("batch"),
             int(admitted), json.dumps(gates, ensure_ascii=False), int(bool(record.get("success"))),
             f2p.get("passed"), f2p.get("total"), p2p.get("passed"), p2p.get("total"),
             record.get("node_evidence"), record.get("patch_chars"), record.get("files"),
             json.dumps(record.get("test_paths") or []), record.get("test_runs"),
             record.get("mutations"), int(bool(record.get("verified_after_last_edit"))),
             record.get("steps"), record.get("finish_reason"), record.get("length"),
             record.get("supervised_tokens", 0), 65536,
             record.get("prompt_tokens"), record.get("completion_tokens"), record.get("duration"),
             record.get("tool_bash", 0), record.get("tool_read", 0), record.get("tool_edit", 0),
             str(path) if path else None,
             sha256_file(path) if path else None))
        for name, ok in passed.items():
            db.execute("INSERT OR REPLACE INTO gates VALUES (?,?,?)",
                       (record["instance_id"], name, ok))

    for row in batch_rows:
        ids = row.get("attempted") or []
        admitted_n = db.execute(
            "SELECT COUNT(*) FROM samples WHERE batch = ? AND admitted = 1",
            (f"batch_{row['batch']:03d}" if isinstance(row.get("batch"), int) else row.get("batch"),)
        ).fetchone()[0]
        name = Path(row["run_dir"]).name if row.get("run_dir") else (
            f"batch_{row['batch']:03d}" if isinstance(row.get("batch"), int) else str(row.get("batch")))
        admitted_n = db.execute("SELECT COUNT(*) FROM samples WHERE batch=? AND admitted=1", (name,)).fetchone()[0]
        db.execute("INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?)",
                   (name, len(ids), row.get("accepted"), admitted_n, row.get("run_dir")))
    db.commit()

    summary = {
        "db": str(DB),
        "samples": db.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
        "admitted": db.execute("SELECT COUNT(*) FROM samples WHERE admitted=1").fetchone()[0],
        "batches": db.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
        "trajectories_linked": db.execute(
            "SELECT COUNT(*) FROM samples WHERE traj_path IS NOT NULL").fetchone()[0],
        "traj_without_file": db.execute(
            "SELECT COUNT(*) FROM samples WHERE traj_path IS NOT NULL AND traj_sha256 IS NULL").fetchone()[0],
        "admitted_tokens": db.execute(
            "SELECT SUM(seq_tokens) FROM samples WHERE admitted=1").fetchone()[0],
    }
    db.close()
    return summary


def report() -> None:
    db = sqlite3.connect(DB)
    print("-- reject reasons --")
    for gate, n in db.execute("SELECT gate, instances FROM v_reject_reasons"):
        print(f"  {gate:20} {n}")
    print("-- admitted length bands --")
    for bound in (8192, 16384, 32768, 65536):
        n = db.execute("SELECT COUNT(*) FROM samples WHERE admitted=1 AND seq_tokens<=?",
                       (bound,)).fetchone()[0]
        print(f"  <= {bound:6}: {n}")
    print("-- tool usage summary (v_tool_usage) --")
    u = db.execute("SELECT total_bash, avg_bash_all, avg_bash_adm, "
                   "total_read, avg_read_all, avg_read_adm, "
                   "total_edit, avg_edit_all, avg_edit_adm FROM v_tool_usage").fetchone()
    if u:
        print(f"  bash: total={u[0]}, avg_all={u[1]}, avg_adm={u[2]}")
        print(f"  read: total={u[3]}, avg_all={u[4]}, avg_adm={u[5]}")
        print(f"  edit: total={u[6]}, avg_all={u[7]}, avg_adm={u[8]}")
    print("-- step brackets (v_step_brackets) --")
    print("  bracket | att | adm | adm% | avg_prompt | avg_comp | avg_adm_seq | avg_adm_sup | adm_bash | read | edit")
    for row in db.execute("SELECT bracket, attempted, admitted, adm_pct, avg_prompt_tokens, "
                          "avg_completion_tokens, avg_adm_seq_tokens, avg_adm_sup_tokens, "
                          "adm_bash, adm_read, adm_edit FROM v_step_brackets"):
        print(f"  {row[0]:7} | {row[1]:3} | {row[2]:3} | {row[3]:4.1f}% | {row[4]:10.0f} | {row[5]:8.0f} | {row[6] or 0:11.0f} | {row[7] or 0:11.0f} | {row[8]:8} | {row[9]:4} | {row[10]:4}")
    print("-- top families by attempts --")
    for family, n in db.execute(
            "SELECT family, COUNT(*) c FROM samples GROUP BY family ORDER BY c DESC LIMIT 5"):
        print(f"  {family:40} {n}")
    print("-- batches --")
    print("  rows:", db.execute("SELECT COUNT(*) FROM batches").fetchone()[0])
    print("-- needs rerun --")
    print("  ", db.execute("SELECT COUNT(*) FROM v_needs_rerun").fetchone()[0])
    db.close()


def check() -> int:
    """Fail when the index disagrees with the ledgers it claims to describe.

    This is the enforcement hook for the registration rule: a report whose numbers
    cannot be reproduced from the index is not allowed to ship.
    """
    ledger = load_jsonl(FORMAL / "admission_ledger.jsonl")
    db = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    problems: list[str] = []
    expected = {
        "samples": (db.execute("SELECT COUNT(*) FROM samples").fetchone()[0], len(ledger)),
        "admitted": (db.execute("SELECT COUNT(*) FROM samples WHERE admitted=1").fetchone()[0],
                     sum(1 for r in ledger if r.get("admitted"))),
        "gate_rows": (db.execute("SELECT COUNT(*) FROM gates").fetchone()[0], len(ledger) * len(GATE_NAMES)),
        "seq_tokens": (db.execute("SELECT COALESCE(SUM(seq_tokens),0) FROM samples WHERE admitted=1").fetchone()[0],
                       sum(r.get("length") or 0 for r in ledger if r.get("admitted"))),
        "supervised_tokens": (db.execute("SELECT COALESCE(SUM(supervised_tokens),0) FROM samples WHERE admitted=1").fetchone()[0],
                              sum(r.get("supervised_tokens") or 0 for r in ledger if r.get("admitted"))),
        "tool_bash": (db.execute("SELECT COALESCE(SUM(tool_bash),0) FROM samples").fetchone()[0],
                      sum(r.get("tool_bash") or 0 for r in ledger)),
        "tool_read": (db.execute("SELECT COALESCE(SUM(tool_read),0) FROM samples").fetchone()[0],
                      sum(r.get("tool_read") or 0 for r in ledger)),
        "tool_edit": (db.execute("SELECT COALESCE(SUM(tool_edit),0) FROM samples").fetchone()[0],
                      sum(r.get("tool_edit") or 0 for r in ledger)),
    }
    for name, (got, want) in expected.items():
        if got != want:
            problems.append(f"{name}: index={got} ledger={want}")
    stored = {g: n for g, n in db.execute("SELECT gate, instances FROM v_reject_reasons")}
    counted = Counter(x for r in ledger for x in (r.get("rejected") or []))
    for gate in set(stored) | set(counted):
        if stored.get(gate, 0) != counted.get(gate, 0):
            problems.append(f"reject[{gate}]: index={stored.get(gate, 0)} ledger={counted.get(gate, 0)}")
    orphans = [p for (p,) in db.execute("SELECT traj_path FROM samples") if not p or not Path(p).exists()]
    if orphans:
        problems.append(f"trajectories missing on disk: {len(orphans)}")
    dupes = db.execute("SELECT COUNT(*) FROM (SELECT instance_id FROM samples "
                       "GROUP BY instance_id HAVING COUNT(*) > 1)").fetchone()[0]
    if dupes:
        problems.append(f"duplicate instance ids: {dupes}")
    family_dupes = db.execute("SELECT lower(trim(family)), COUNT(*) FROM samples WHERE admitted=1 "
                              "GROUP BY lower(trim(family)) HAVING COUNT(*) > 1").fetchall()
    if family_dupes:
        problems.append(f"admitted families repeated: {family_dupes[:5]}")

    try:
        protected_cfg = load_protected_families()
        protected_families = protected_cfg.families
    except Exception as exc:
        problems.append(f"protected families config error: {exc}")
        protected_families = frozenset()

    benchmark_admitted = [
        (iid, fam)
        for (iid, fam) in db.execute("SELECT instance_id, family FROM samples WHERE admitted=1").fetchall()
        if is_benchmark_family(fam, protected_families)
    ]
    if benchmark_admitted:
        problems.append(f"admitted benchmark families detected in index: {benchmark_admitted[:5]}")

    db.close()
    if problems:
        print("INDEX OUT OF DATE — rebuild with: python results/build_index_db.py")
        for problem in problems:
            print("  -", problem)
        return 1
    print(f"OK: index matches ledger ({len(ledger)} samples, "
          f"{expected['admitted'][0]} admitted, {expected['seq_tokens'][0]} tokens)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the index disagrees with the ledgers")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check())
    if args.report:
        report()
        return
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    report()


if __name__ == "__main__":
    main()
