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
from collections import Counter
from hashlib import sha256
from pathlib import Path

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
    length_gate     INTEGER,           -- 65536 at admission time
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    duration_s      REAL,
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
"""

GATE_NAMES = ("not_passed", "f2p_incomplete", "p2p_incomplete", "over_length",
              "test_paths_edited", "no_test_at_all", "empty_patch", "duplicate_patch",
              "infra_error")


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
                        found[json.loads(line)["instance_id"]] = traj
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            continue
    return found


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build() -> dict:
    ledger = load_jsonl(FORMAL / "admission_ledger.jsonl")
    batch_rows = load_jsonl(FORMAL / "batch_ledger.jsonl")
    traj = trajectory_index()
    db = sqlite3.connect(DB)
    db.executescript("DROP VIEW IF EXISTS v_reject_reasons;"
                     "DROP VIEW IF EXISTS v_needs_rerun;"
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
            "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record["instance_id"], record.get("family", ""), record.get("batch"),
             int(admitted), json.dumps(gates, ensure_ascii=False), int(bool(record.get("success"))),
             f2p.get("passed"), f2p.get("total"), p2p.get("passed"), p2p.get("total"),
             record.get("node_evidence"), record.get("patch_chars"), record.get("files"),
             json.dumps(record.get("test_paths") or []), record.get("test_runs"),
             record.get("mutations"), int(bool(record.get("verified_after_last_edit"))),
             record.get("steps"), record.get("finish_reason"), record.get("length"), 65536,
             record.get("prompt_tokens"), record.get("completion_tokens"), record.get("duration"),
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
        name = f"batch_{row['batch']:03d}" if isinstance(row.get("batch"), int) else str(row.get("batch"))
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
    print("-- top families by attempts --")
    for family, n in db.execute(
            "SELECT family, COUNT(*) c FROM samples GROUP BY family ORDER BY c DESC LIMIT 5"):
        print(f"  {family:40} {n}")
    print("-- batches --")
    print("  rows:", db.execute("SELECT COUNT(*) FROM batches").fetchone()[0])
    print("-- needs rerun --")
    print("  ", db.execute("SELECT COUNT(*) FROM v_needs_rerun").fetchone()[0])
    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
        return
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    report()


if __name__ == "__main__":
    main()
