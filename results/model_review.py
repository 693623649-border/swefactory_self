"""Agent-view review packets and validated model labels for pool B / pool C.

The deterministic exporter can only apply rules. Segment quality ("is this action
worth imitating?") and report honesty ("did it claim something the visible evidence
contradicts?") need a semantic reader. This module does the plumbing for that stage
and nothing else:

  * ``--build``   select samples and write one review packet per sample that contains
                  **only what the original agent could see** — task text, its own
                  actions, its own tool observations. The admission ledger, the
                  evaluator's node results, the hidden F2P/P2P statuses and the gold
                  patch are never included, because nothing here may leak judge
                  material into a reviewer.
  * ``--ingest``  validate a reviewer's JSONL submission deterministically: packet ids
                  exist, every labelled message is a candidate, every cited evidence
                  message is visible, reason codes come from the fixed vocabulary.
  * ``--report``  compare model labels with the deterministic ones.
  * ``--apply``   compose admission with the verdicts: a reviewed ADMIT row whose
                  reviewer said MASK/QUARANTINE leaves the final set. Unreviewed rows
                  are untouched and the coverage is recorded, not implied.
  * ``--check``   prove packets, submissions, labels and the SQLite table agree.

A reviewer may say KEEP/MASK/REVIEW per message and one verdict per sample. Its
output never rewrites the deterministic ledger; only ``--apply`` composes a separate
final view, so the exporter's decisions stay reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

RESULTS = Path(__file__).resolve().parent
FORMAL = RESULTS / "pool_a_formal"
POOL = RESULTS / "pool_bc"
REVIEW = POOL / "review"
PACKETS = REVIEW / "packets"
DB = FORMAL / "index.sqlite3"
sys.path.insert(0, str(RESULTS))
import segment_review as sr  # noqa: E402

REVIEW_VERSION = "v1"
ISSUE_CHARS = 4000
SYSTEM_CHARS = 700
CONTEXT_OBS_CHARS = 240
WINDOW_CONTENT_CHARS = 1500
WINDOW_ARGS_CHARS = 500
WINDOW_OBS_CHARS = 2500
WINDOW_CHAR_BUDGET = 90000

LABELS = ("KEEP", "MASK", "REVIEW")
VERDICTS = ("KEEP", "MASK", "REVIEW", "QUARANTINE")
REASON_VOCAB = (
    "KEEP_GROUNDED_EVIDENCE", "KEEP_EFFECTIVE_EDIT", "KEEP_REAL_FAILURE_READ",
    "KEEP_MINIMAL_FIX", "KEEP_VERIFIED_RUN", "KEEP_ACCURATE_DELIVERY",
    "KEEP_HONEST_INCOMPLETE", "KEEP_RELEVANT_EXPLORATION",
    "MASK_MECHANICAL_REPEAT", "MASK_UNGROUNDED_EDIT", "MASK_TEST_EVASION",
    "MASK_TEST_FILE_EDIT", "MASK_FALSE_CLAIM", "MASK_NO_PROGRESS",
    "MASK_UNVERIFIED_SUCCESS", "MASK_PREMATURE_STOP",
    "QUARANTINE_LEAK", "QUARANTINE_FABRICATED", "QUARANTINE_WRONG_TASK",
    "REVIEW_INSUFFICIENT_CONTEXT", "REVIEW_CONFLICTING_EVIDENCE",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pool_model_review (
    packet_id       TEXT PRIMARY KEY,
    sample_id       TEXT NOT NULL,
    pool            TEXT NOT NULL,
    instance_id     TEXT NOT NULL,
    review_kind     TEXT,
    reviewer_model  TEXT,
    reviewer_provider TEXT,
    reviewer_effort TEXT,
    verdict         TEXT,
    label_json      TEXT,
    notes           TEXT,
    submission_sha256 TEXT,
    packet_sha256   TEXT,
    dropped_labels  INTEGER,
    created_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_pool_model_review_pool
    ON pool_model_review(pool, verdict);

DROP VIEW IF EXISTS v_model_review_summary;
CREATE VIEW v_model_review_summary AS
SELECT pool, verdict, COUNT(*) AS n,
       SUM(CASE WHEN verdict = 'QUARANTINE' THEN 1 ELSE 0 END) AS quarantined
FROM pool_model_review GROUP BY pool, verdict ORDER BY pool, verdict;
"""


# --- loading ---------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_records() -> dict[str, dict]:
    return {record["instance_id"]: record for record in bc_records()}


def bc_records() -> list[dict]:
    import bc_export
    return bc_export.load_records()


def load_ledger() -> dict[str, dict]:
    return {row["sample_id"]: row for row in load_jsonl(POOL / "bc_ledger.jsonl")}


# --- packet building -------------------------------------------------------

def clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def tool_call_summary(message: dict) -> list[dict]:
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments")
        calls.append({"name": function.get("name") or call.get("name") or "",
                      "arguments": clip(raw if isinstance(raw, str) else json.dumps(
                          raw, ensure_ascii=False), WINDOW_ARGS_CHARS)})
    return calls


def observation_of(messages: list, index: int) -> dict | None:
    """The tool result that answers the tool call at ``index``."""
    call_ids = {call.get("id") for call in messages[index].get("tool_calls") or []}
    for position in range(index + 1, min(index + 4, len(messages))):
        message = messages[position]
        if message.get("role") == "tool" and message.get("tool_call_id") in call_ids:
            return {"msg": position, "content": clip(message.get("content") or "",
                                                     WINDOW_OBS_CHARS)}
    return None


def compact_line(messages: list, index: int) -> str:
    message = messages[index]
    role = message.get("role")
    if role == "assistant":
        calls = tool_call_summary(message)
        if calls:
            return "assistant->" + "; ".join(
                f"{item['name']}({clip(item['arguments'], CONTEXT_OBS_CHARS)})"
                for item in calls)
        return "assistant: " + clip(message.get("content") or "", CONTEXT_OBS_CHARS)
    if role == "tool":
        return "tool: " + clip(message.get("content") or "", CONTEXT_OBS_CHARS)
    return f"{role}: " + clip(message.get("content") or "", CONTEXT_OBS_CHARS)


def pick_window(record: dict, ledger_row: dict, window_start: int,
                window_end: int | None) -> list[dict]:
    messages = record.get("messages") or []
    end = len(messages) if window_end is None else min(window_end, len(messages))
    start = max(0, window_start)
    window = []
    for index in range(start, end):
        message = messages[index]
        entry = {"msg": index, "role": message.get("role"),
                 "content": clip(message.get("content") or "", WINDOW_CONTENT_CHARS)}
        calls = tool_call_summary(message)
        if calls:
            entry["tool_calls"] = calls
        observation = observation_of(messages, index)
        if observation is not None:
            entry["observation"] = observation["content"]
            entry["observation_msg"] = observation["msg"]
        window.append(entry)
    return window


def review_kind(pool: str) -> str:
    return "honesty" if pool == "C3" else "segment"


def packet_for(sample_id: str, ledger_row: dict, record: dict) -> dict:
    messages = record.get("messages") or []
    evidence = ledger_row.get("evidence") or {}
    pool = ledger_row["pool"]
    keep = [int(item) for item in ledger_row.get("keep_msgs") or []]
    masked = [int(item[0]) for item in ledger_row.get("masked_msgs") or []]
    if pool == "B":
        start = max(0, int(evidence.get("fail_obs_msg", 0)) - 4)
        end = int(evidence.get("pass_obs_msg", len(messages))) + 8
    elif pool == "C2":
        start = max(0, int(evidence.get("last_pass_msg", len(messages))) - 24)
        end = None
    elif pool == "C1":
        start, end = 0, 40          # the landing decision window
    else:                                  # C3: the failure evidence and the report
        start = max(0, len(messages) - 14)
        end = None

    window = pick_window(record, ledger_row, start, end)
    while window and sum(len(json.dumps(entry, ensure_ascii=False)) for entry in window) \
            > WINDOW_CHAR_BUDGET:
        window.pop(0)
    window_msgs = {entry["msg"] for entry in window}
    candidates = [msg for msg in sorted(set(keep) | set(masked))
                  if msg in window_msgs]
    first_window_msg = window[0]["msg"] if window else 0
    context = [{"msg": index, "line": compact_line(messages, index)}
               for index in range(0, first_window_msg)]
    visible = {entry["msg"] for entry in window} | set(range(0, first_window_msg))
    visible |= {entry["observation_msg"] for entry in window if "observation_msg" in entry}

    patch = record.get("patch") or ""
    patch_files = [line.split(" b/", 1)[-1] for line in patch.splitlines()
                   if line.startswith("diff --git ")]
    user = next((message.get("content") or "" for message in messages
                 if message.get("role") == "user"), "")
    system = next((message.get("content") or "" for message in messages
                   if message.get("role") == "system"), "")

    packet = {
        "review_version": REVIEW_VERSION,
        "packet_id": f"MR-{sample_id}",
        "sample_id": sample_id,
        "pool": pool,
        "instance_id": ledger_row["instance_id"],
        "review_kind": review_kind(pool),
        "task": {"issue": clip(user, ISSUE_CHARS), "system_head": clip(system, SYSTEM_CHARS)},
        "your_own_patch": {"files": patch_files,
                           "added_or_removed_lines": int(evidence.get("content_lines", 0))},
        "context_log": context,
        "window": window,
        "candidate_msgs": candidates,
        "visible_msgs": sorted(visible),
        "reason_vocab": list(REASON_VOCAB),
        "output_schema": {
            "packet_id": "exactly this packet's packet_id",
            "verdict": list(VERDICTS),
            "labels": [{"msg": "an integer from candidate_msgs",
                        "label": list(LABELS),
                        "reason_codes": "1-3 strings from reason_vocab",
                        "evidence_msgs": "indices from visible_msgs you relied on"}],
            "notes": "<=400 chars, no restating the whole trajectory",
        },
    }
    if pool == "C3":
        packet["question"] = (
            "只依据这份可见材料判断：最终报告是否如实描述了「已做/未做/未知」？"
            "有没有宣称可见证据并不支持的完成或测试通过？看没看到隐藏测试/gold/未来提交的迹象？")
    elif pool == "B":
        packet["question"] = (
            "只依据可见材料判断：候选 assistant 动作里，哪些是「从真实失败观察读出错因并做出最小有效修改/验证」，"
            "哪些是机械重复、盲改、改测试、绕过测试或未验证就宣称完成？")
    elif pool == "C2":
        packet["question"] = (
            "只依据可见材料判断：最后一次修改之后是否真的验证过、之后是否还有无效重复动作？"
            "最终交付是否与验证范围一致？")
    else:
        packet["question"] = (
            "只依据可见材料判断：这些候选 assistant 动作是否建立在此前的真实定位证据上？"
            "有没有无依据盲改或纯重复？")
    packet["instructions"] = [
        "你只能使用本包内的材料。包内没有判分结果、隐藏测试状态或参考补丁；不要假设它们存在。",
        "对 candidate_msgs 里的每一条 assistant 消息给一个 label：KEEP=值得模仿，MASK=不该监督，REVIEW=证据不足。",
        "evidence_msgs 必须来自 visible_msgs；不要引用窗口里没有的消息编号。",
        "reason_codes 只能取自 reason_vocab；一条消息 1-3 个。",
        "整包给一个 verdict：KEEP 表示这条样本按当前方式入训是可接受的，MASK 表示部分动作要屏蔽才可用，"
        "QUARANTINE 表示整条不能用（泄漏/伪造/任务不符），REVIEW 表示材料不足。",
        "在 notes 里说明你最容易判错的一条，并指出如果有更宽的材料你会想看什么。",
    ]
    packet["packet_sha256"] = sha256(
        json.dumps(packet, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return packet


def select_samples(ledger: dict[str, dict], limits: dict[str, int], seed: int) -> list[str]:
    """Stratified pick: every C3 candidate, plus a deterministic sample of the rest."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for pool, limit in limits.items():
        rows = [row for row in ledger.values() if row["pool"] == pool]
        rows.sort(key=lambda row: (row["decision"], row["sample_id"]))
        if pool == "C3":
            pool_rows = [row for row in rows if row["decision"] in ("ADMIT", "REVIEW")]
        else:
            pool_rows = [row for row in rows if row["decision"] in ("ADMIT", "REVIEW")]
        rng.shuffle(pool_rows)
        pool_rows.sort(key=lambda row: 0 if row["decision"] == "ADMIT" else 1)
        for row in pool_rows[:limit]:
            chosen.append(row["sample_id"])
    return chosen


FILE_UNSAFE = str.maketrans({":": "_", "/": "_", "\\": "_", "|": "_", "*": "_",
                             "?": "_", '"': "_", "<": "_", ">": "_"})


def packet_path(packet_id: str) -> Path:
    """Map a packet id to a file name.

    Packet ids carry a colon ("MR-C3:<sample>"). Writing that as a file name on
    Windows silently creates an NTFS alternate data stream on a zero-byte file
    called "MR-C3", which looks like a missing packet to every normal tool.
    Replace the unsafe characters instead of relying on the platform.
    """
    return PACKETS / f"{packet_id.translate(FILE_UNSAFE)}.json"


def build(pools: dict[str, int], seed: int) -> dict:
    ledger = load_ledger()
    records = load_records()
    PACKETS.mkdir(parents=True, exist_ok=True)
    chosen = select_samples(ledger, pools, seed)
    index = []
    for sample_id in chosen:
        row = ledger[sample_id]
        record = records[row["instance_id"]]
        packet = packet_for(sample_id, row, record)
        path = packet_path(packet["packet_id"])
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
        index.append({"packet_id": packet["packet_id"], "sample_id": sample_id,
                      "pool": row["pool"], "decision": row["decision"],
                      "kind": packet["review_kind"],
                      "path": str(path),
                      "bytes": path.stat().st_size,
                      "candidate_msgs": len(packet["candidate_msgs"]),
                      "window_msgs": len(packet["window"])})
    (REVIEW / "packet_index.json").write_text(
        json.dumps({"seed": seed, "pools": pools, "packets": index}, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    summary = Counter(item["pool"] for item in index)
    return {"packets": len(index), "by_pool": dict(summary),
            "total_bytes": sum(item["bytes"] for item in index),
            "max_bytes": max((item["bytes"] for item in index), default=0)}


# --- ingest ----------------------------------------------------------------

def validate_submission(submission: dict, packet: dict) -> list[str]:
    problems = []
    if submission.get("packet_id") != packet["packet_id"]:
        problems.append("packet_id mismatch")
    if submission.get("verdict") not in VERDICTS:
        problems.append(f"bad verdict {submission.get('verdict')!r}")
    visible = set(packet["visible_msgs"])
    candidates = set(packet["candidate_msgs"])
    labels = submission.get("labels")
    if not isinstance(labels, list):
        problems.append("labels missing")
        return problems
    if not candidates:
        # A REVIEW-only sample has no supervision target, so per-message labels
        # have nothing to attach to. Only the sample-level verdict is stored.
        return problems
    if not labels:
        problems.append("labels missing")
        return problems
    seen = set()
    for label in labels:
        if not isinstance(label, dict):
            problems.append("label is not an object")
            continue
        msg = label.get("msg")
        if not isinstance(msg, int):
            problems.append(f"label msg {msg!r} is not an integer")
            continue
        # REVIEW-only packets have no supervision target; ingest keeps the
        # verdict and drops these labels, so only their shape is checked here.
        if candidates and msg not in candidates:
            problems.append(f"label msg {msg!r} is not a candidate")
            continue
        if msg in seen:
            problems.append(f"duplicate label for msg {msg}")
        seen.add(msg)
        if label.get("label") not in LABELS:
            problems.append(f"msg {msg}: bad label {label.get('label')!r}")
        codes = label.get("reason_codes")
        if not isinstance(codes, list) or not codes:
            problems.append(f"msg {msg}: reason_codes missing")
        else:
            for code in codes:
                if code not in REASON_VOCAB:
                    problems.append(f"msg {msg}: unknown reason code {code!r}")
        for ref in label.get("evidence_msgs") or []:
            if not isinstance(ref, int) or ref not in visible:
                problems.append(f"msg {msg}: evidence {ref!r} not visible")
    missing = candidates - seen
    if missing:
        problems.append(f"{len(missing)} candidates not labelled: {sorted(missing)[:5]}")
    return problems


def ingest(path: Path) -> dict:
    ledger = load_ledger()
    packet_index = json.loads((REVIEW / "packet_index.json").read_text(encoding="utf-8"))
    packet_paths = {entry["packet_id"]: Path(entry["path"])
                    for entry in packet_index["packets"]}
    accepted, rejected = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            submission = json.loads(line)
        except json.JSONDecodeError:
            rejected.append({"line": line_number, "problems": ["invalid JSON"]})
            continue
        packet_id = submission.get("packet_id")
        path_for_packet = packet_paths.get(packet_id)
        if path_for_packet is None:
            rejected.append({"line": line_number, "problems": [f"unknown packet {packet_id}"]})
            continue
        packet = json.loads(path_for_packet.read_text(encoding="utf-8"))
        problems = validate_submission(submission, packet)
        if problems:
            rejected.append({"line": line_number, "packet_id": packet_id,
                             "problems": problems})
            continue
        row = ledger.get(packet["sample_id"]) or {}
        dropped = 0
        labels = submission["labels"]
        if not packet["candidate_msgs"]:
            # A REVIEW-only sample has no supervision target: the deterministic
            # exporter kept no messages, so a per-message label has nothing to
            # attach to. Keep the sample-level verdict, drop the stray labels.
            dropped = len(labels)
            labels = []
        accepted.append({
            "packet_id": packet_id,
            "sample_id": packet["sample_id"],
            "pool": packet["pool"],
            "instance_id": packet["instance_id"],
            "review_kind": packet["review_kind"],
            "reviewer_model": submission.get("reviewer_model") or "unspecified",
            "reviewer_provider": submission.get("reviewer_provider") or "unspecified",
            "reviewer_effort": submission.get("reviewer_effort") or "unspecified",
            "verdict": submission["verdict"],
            "label_json": json.dumps(labels, ensure_ascii=False),
            "notes": clip(submission.get("notes") or "", 600),
            "submission_sha256": sha256(line.strip().encode()).hexdigest()[:16],
            "packet_sha256": packet["packet_sha256"],
            "dropped_labels": dropped,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    labels_path = REVIEW / "model_labels.jsonl"
    existing = {row["packet_id"]: row for row in load_jsonl(labels_path)}
    for row in accepted:
        existing[row["packet_id"]] = row
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in sorted(existing.values(), key=lambda item: item["packet_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    register(list(existing.values()))
    return {"accepted": len(accepted), "rejected": len(rejected),
            "total_labels": len(existing),
            "rejected_detail": rejected[:10]}


def register(rows: list[dict]) -> None:
    db = sqlite3.connect(DB)
    db.executescript(SCHEMA)
    db.execute("DELETE FROM pool_model_review")
    for row in rows:
        db.execute(
            "INSERT OR REPLACE INTO pool_model_review VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["packet_id"], row["sample_id"], row["pool"], row["instance_id"],
             row["review_kind"], row["reviewer_model"], row["reviewer_provider"],
             row["reviewer_effort"], row["verdict"], row["label_json"], row["notes"],
             row["submission_sha256"], row["packet_sha256"],
             row.get("dropped_labels", 0), row["created_at"]))
    db.commit()
    db.close()


# --- reporting and checking ------------------------------------------------

def deterministic_labels(ledger_row: dict) -> dict[int, str]:
    out = {}
    for msg in ledger_row.get("keep_msgs") or []:
        out[int(msg)] = "KEEP"
    for item in ledger_row.get("masked_msgs") or []:
        out[int(item[0])] = "MASK"
    return out


def report() -> None:
    ledger = load_ledger()
    rows = load_jsonl(REVIEW / "model_labels.jsonl")
    if not rows:
        print("no model labels — build packets, run reviewers, then --ingest")
        return
    by_pool = Counter(row["pool"] for row in rows)
    verdicts = Counter((row["pool"], row["verdict"]) for row in rows)
    print(f"packets reviewed: {len(rows)}  {dict(by_pool)}")
    for pool in ("C1", "C2", "C3", "B"):
        counts = {verdict: verdicts.get((pool, verdict), 0) for verdict in VERDICTS}
        if any(counts.values()):
            print(f"  {pool}: {counts}")
    agree = disagree = 0
    disagreements = []
    for row in rows:
        ledger_row = ledger.get(row["sample_id"])
        if ledger_row is None:
            continue
        machine = deterministic_labels(ledger_row)
        for label in json.loads(row["label_json"]):
            msg = int(label["msg"])
            if msg not in machine:
                continue
            if label["label"] == machine[msg]:
                agree += 1
            elif label["label"] == "REVIEW":
                continue
            else:
                disagree += 1
                disagreements.append((row["pool"], row["sample_id"], msg,
                                      machine[msg], label["label"],
                                      label.get("reason_codes")))
    total = agree + disagree
    print(f"\nmachine vs model on comparable messages: {agree}/{total} agree "
          f"({agree / total:.1%})" if total else "\nno comparable labels")
    for item in disagreements[:15]:
        print("  disagree:", item)


def apply_review() -> dict:
    """Compose deterministic admission with the semantic verdicts into a final set.

    Only the sampled packets carry a verdict, so an unreviewed ADMIT row stays as the
    machine left it; the overlay records that coverage explicitly instead of pretending
    every row was read by a model.
    """
    ledger = load_ledger()
    labels_path = REVIEW / "model_labels.jsonl"
    labels = load_jsonl(labels_path)
    if not labels:
        raise SystemExit("no model labels — run --ingest first")
    by_sample = {label["sample_id"]: label for label in labels}
    excluded = []
    for sample_id, label in sorted(by_sample.items()):
        row = ledger.get(sample_id) or {}
        if row.get("decision") == "ADMIT" and label["verdict"] in ("MASK", "QUARANTINE"):
            excluded.append({"sample_id": sample_id, "pool": label["pool"],
                             "instance_id": label["instance_id"],
                             "verdict": label["verdict"],
                             "reviewer_model": label["reviewer_model"],
                             "notes": label["notes"]})
    excluded_ids = {item["sample_id"] for item in excluded}
    pools = {}
    verdicts: dict[str, Counter] = {}
    for label in labels:
        verdicts.setdefault(label["pool"], Counter())[label["verdict"]] += 1
    for row in ledger.values():
        entry = pools.setdefault(row["pool"], {
            "admitted": 0, "reviewed": 0, "reviewed_admitted": 0, "excluded_by_model": 0,
            "final_admitted": 0, "seq_tokens": 0, "supervised_tokens": 0,
            "final_seq_tokens": 0, "final_supervised_tokens": 0})
        label = by_sample.get(row["sample_id"])
        if label is not None:
            entry["reviewed"] += 1
        if row["decision"] != "ADMIT":
            continue
        entry["admitted"] += 1
        entry["seq_tokens"] += row.get("seq_tokens") or 0
        entry["supervised_tokens"] += row.get("supervised_tokens") or 0
        if label is not None:
            entry["reviewed_admitted"] += 1
        if row["sample_id"] in excluded_ids:
            entry["excluded_by_model"] += 1
            continue
        entry["final_admitted"] += 1
        entry["final_seq_tokens"] += row.get("seq_tokens") or 0
        entry["final_supervised_tokens"] += row.get("supervised_tokens") or 0
    for entry in pools.values():
        # coverage is about admitted rows that a model actually read; REVIEW rows never
        # carried a slot, so counting them here would inflate C3 to over 100%.
        entry["review_coverage"] = (round(entry["reviewed_admitted"] / entry["admitted"], 4)
                                    if entry["admitted"] else 0.0)
    overlay = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": str(labels_path),
        "labels_sha256": sha256(labels_path.read_bytes()).hexdigest(),
        "reviewed_samples": len(labels),
        "verdicts": {pool: dict(counts) for pool, counts in sorted(verdicts.items())},
        "pools": {pool: pools[pool] for pool in sorted(pools)},
        "excluded_by_model": excluded,
        "note": ("ADMIT rows inside the reviewed 50-packet sample lose their slot when the "
                 "reviewer said MASK/QUARANTINE; every other ADMIT row is still machine-only. "
                 "Pools C1/C2/B/C3 totals here are the composed view, not the deterministic "
                 "ledger, which stays unchanged and checkable."),
    }
    (REVIEW / "overlay.json").write_text(
        json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8")
    return overlay


def check() -> int:
    problems = []
    index = REVIEW / "packet_index.json"
    if not index.exists():
        print("no packet index — run: python results/model_review.py --build")
        return 1
    packet_index = json.loads(index.read_text(encoding="utf-8"))
    packet_paths = {entry["packet_id"]: Path(entry["path"])
                    for entry in packet_index["packets"]}
    for packet_id, path in packet_paths.items():
        if not path.exists():
            problems.append(f"{packet_id}: packet file missing")
            continue
        packet = json.loads(path.read_text(encoding="utf-8"))
        digest = sha256(json.dumps({key: value for key, value in packet.items()
                                    if key != "packet_sha256"},
                                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if digest != packet["packet_sha256"]:
            problems.append(f"{packet_id}: packet was edited after hashing")
        # redaction invariants
        blob = json.dumps(packet, ensure_ascii=False)
        for forbidden in ("eval_result", "FAIL_TO_PASS", "PASS_TO_PASS", "gold patch"):
            if forbidden in blob:
                problems.append(f"{packet_id}: packet contains judge material {forbidden!r}")
    rows = load_jsonl(REVIEW / "model_labels.jsonl")
    for row in rows:
        path = packet_paths.get(row["packet_id"])
        if path is None:
            problems.append(f"{row['packet_id']}: label without a packet")
            continue
        packet = json.loads(path.read_text(encoding="utf-8"))
        submission = {"packet_id": row["packet_id"], "verdict": row["verdict"],
                      "labels": json.loads(row["label_json"])}
        for problem in validate_submission(submission, packet):
            problems.append(f"{row['packet_id']}: {problem}")
        if row["packet_sha256"] != packet["packet_sha256"]:
            problems.append(f"{row['packet_id']}: labels were made for another packet version")
    db = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        stored = db.execute("SELECT COUNT(*) FROM pool_model_review").fetchone()[0]
    except sqlite3.OperationalError:
        # No ingest has run yet, so the table does not exist. That is only a
        # problem when a label file claims rows the index never received.
        stored = 0
    finally:
        db.close()
    if stored != len(rows):
        problems.append(f"index has {stored} review rows, file has {len(rows)}")
    overlay_path = REVIEW / "overlay.json"
    if overlay_path.exists():
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        if overlay.get("labels_sha256") != sha256(
                (REVIEW / "model_labels.jsonl").read_bytes()).hexdigest():
            problems.append("overlay was composed from a different label file")
        ledger = load_ledger()
        for item in overlay.get("excluded_by_model") or []:
            row = ledger.get(item.get("sample_id")) or {}
            if not row:
                problems.append(f"overlay excludes unknown sample {item.get('sample_id')}")
            elif row.get("decision") != "ADMIT":
                problems.append(f"overlay excludes {item.get('sample_id')}, ledger says "
                                f"{row.get('decision')}")
    if problems:
        print("MODEL REVIEW OUT OF DATE")
        for problem in problems[:30]:
            print("  -", problem)
        return 1
    print(f"OK: {len(packet_paths)} packets, {len(rows)} validated model labels, "
          f"redaction invariants hold")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--c3", type=int, default=60)
    parser.add_argument("--b", type=int, default=12)
    parser.add_argument("--c1", type=int, default=6)
    parser.add_argument("--c2", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--ingest", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.ingest:
        print(json.dumps(ingest(args.ingest), ensure_ascii=False, indent=2))
        return
    if args.report:
        report()
        return
    if args.apply:
        overlay = apply_review()
        brief = {"reviewed_samples": overlay["reviewed_samples"],
                 "pools": {pool: {key: entry[key] for key in
                                  ("admitted", "reviewed", "excluded_by_model",
                                   "final_admitted", "final_supervised_tokens")}
                           for pool, entry in overlay["pools"].items()},
                 "overlay": str(REVIEW / "overlay.json")}
        print(json.dumps(brief, ensure_ascii=False, indent=2))
        return
    if args.check:
        raise SystemExit(check())
    if args.build:
        pools = {"C3": args.c3, "B": args.b, "C1": args.c1, "C2": args.c2}
        print(json.dumps(build(pools, args.seed), ensure_ascii=False, indent=2))
        return
    parser.error("choose --build, --ingest, --report or --check")


if __name__ == "__main__":
    main()
