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

REVIEW_VERSION = "v2-full-assistant"
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def load_records() -> dict[str, dict]:
    return {record["instance_id"]: record for record in bc_records()}


def bc_records() -> list[dict]:
    import bc_export
    return bc_export.load_records()


def load_ledger() -> dict[str, dict]:
    path = POOL / "bc_rule_ledger.jsonl"
    if not path.exists():
        path = POOL / "bc_ledger.jsonl"
    rows = load_jsonl(path) + load_jsonl(POOL / "a_rule_ledger.jsonl")
    return {row["sample_id"]: row for row in rows}


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
    # Every supervised assistant target remains visible in full, including reasoning
    # and native tool arguments. Tool observations are bounded and explicitly marked.
    window = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system":
            continue
        entry = {"msg": index, "role": role}
        content = message.get("content") or ""
        entry["content"] = content if role in ("assistant", "user") else clip(content, WINDOW_OBS_CHARS)
        if role == "assistant":
            for key in ("reasoning_raw", "reasoning_content", "reasoning_text", "tool_calls"):
                if message.get(key):
                    entry[key] = message[key]
        elif role == "tool":
            entry["content_chars"] = len(content)
            entry["content_sha256"] = sha256(content.encode()).hexdigest()
            entry["content_truncated"] = len(content) > WINDOW_OBS_CHARS
            if entry["content_truncated"]:
                half = WINDOW_OBS_CHARS // 2
                entry["content"] = content[:half] + "\n[tool observation middle omitted]\n" + content[-half:]
            entry["tool_call_id"] = message.get("tool_call_id")
            entry["name"] = message.get("name")
        window.append(entry)
    visible = {item["msg"] for item in window}
    candidates = sorted(set(keep) | set(masked))
    if any(m not in visible or messages[m].get("role") != "assistant" for m in candidates):
        raise ValueError(f"{sample_id}: supervision target is not a visible assistant")
    context = []
    trajectory_hash = sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    patch = record.get("patch") or ""
    patch_files = [line.split(" b/", 1)[-1] for line in patch.splitlines()
                   if line.startswith("diff --git ")]
    user = next((message.get("content") or "" for message in messages
                 if message.get("role") == "user"), "")
    system = next((message.get("content") or "" for message in messages
                   if message.get("role") == "system"), "")

    packet = {
        "review_version": REVIEW_VERSION,
        "trajectory_sha256": trajectory_hash,
        "supervised_msgs": keep,
        "coverage": "all supervised assistant messages; tool observations clipped to 2500 characters",
        "packet_id": f"MR-{sample_id}",
        "sample_id": sample_id,
        "pool": pool,
        "instance_id": ledger_row["instance_id"],
        "review_kind": review_kind(pool),
        "task": {"issue": user, "system_head": clip(system, SYSTEM_CHARS)},
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
            "哪些是机械重复、盲改、改测试、绕过测试或未验证就宣称完成？"
            "必须从命令和观察确认原先失败的测试在源码修改后被重新执行并通过；"
            "测试数量不减不能证明覆盖同一失败。影子测试副本、跳过失败节点、只更换运行目录/导入方式"
            "不能证明源码纠错；若无法确认原失败被修复并复验，整包 verdict 必须 REVIEW 或 QUARANTINE。")
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


def packet_index_paths() -> dict[str, Path]:
    """Every packet a label may reference: the sampled set plus the bulk full-pool set.

    The two indexes are written by different commands (``--build`` here,
    ``model_review_bulk.py --build`` for the whole pool); ingest and check must accept
    labels against either one, otherwise every bulk label looks orphaned.
    """
    paths: dict[str, Path] = {}
    for index_path in (REVIEW / "packet_index.json", REVIEW / "bulk" / "bulk_index.json"):
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index.get("packets") or []:
            paths[entry["packet_id"]] = Path(entry["path"])
    return paths


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
    packet_paths = packet_index_paths()
    accepted, rejected = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
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
        if submission.get("packet_sha256") != packet["packet_sha256"]:
            problems.append("submission packet_sha256 missing or stale")
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
            "trajectory_sha256": packet.get("trajectory_sha256"),
            "review_config_sha256": submission.get("review_config_sha256"),
            "review_config": submission.get("review_config"),
            "dropped_labels": dropped,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    labels_path = REVIEW / "model_labels.jsonl"
    # Archive the previous label snapshot; only labels for current packet bytes stay active.
    old_rows = load_jsonl(labels_path)
    if labels_path.exists():
        archive = REVIEW / "history"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / (sha256(labels_path.read_bytes()).hexdigest() + ".jsonl")).write_bytes(labels_path.read_bytes())
    existing = {}
    for row in old_rows:
        if ledger.get(row["sample_id"], {}).get("decision") != "ADMIT":
            continue
        current = packet_paths.get(row["packet_id"])
        if current and current.exists():
            packet_now = json.loads(current.read_text(encoding="utf-8"))
            if row.get("packet_sha256") == packet_now["packet_sha256"]:
                existing[row["packet_id"]] = row
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
    for name in ("review_config_sha256", "trajectory_sha256"):
        if name not in {column[1] for column in db.execute("PRAGMA table_info(pool_model_review)")}:
            db.execute(f"ALTER TABLE pool_model_review ADD COLUMN {name} TEXT")
    db.execute("DELETE FROM pool_model_review")
    for row in rows:
        db.execute(
            "INSERT OR REPLACE INTO pool_model_review (packet_id,sample_id,pool,instance_id,review_kind,reviewer_model,reviewer_provider,reviewer_effort,verdict,label_json,notes,submission_sha256,packet_sha256,dropped_labels,created_at,review_config_sha256,trajectory_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["packet_id"], row["sample_id"], row["pool"], row["instance_id"],
             row["review_kind"], row["reviewer_model"], row["reviewer_provider"],
             row["reviewer_effort"], row["verdict"], row["label_json"], row["notes"],
             row["submission_sha256"], row["packet_sha256"],
             row.get("dropped_labels", 0), row["created_at"], row.get("review_config_sha256"), row.get("trajectory_sha256")))
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
    for pool in ("A", "C1", "C2", "C3", "B"):
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


import bc_export


def _rule_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return bc_export.rule_digest(rows)


def apply_review(require_complete: bool = False) -> dict:
    """Compose deterministic admission with the semantic verdicts into a final view.

    Two different things hide behind one ``MASK``:

      * a *sample* verdict QUARANTINE means the trajectory itself is poisoned (judge
        leakage, fabricated evidence, wrong task). Narrowing supervision cannot repair
        that, so the whole sample leaves the set;
      * a *message* label MASK/REVIEW is the normal case — this action should not be
        imitated. Dropping the whole sample for one repeated command threw away the
        good edits around it, so the mask is composed at message level: those messages
        leave ``keep_msgs`` and everything else still trains.

    Unreviewed rows keep exactly what the machine decided; coverage is recorded, never
    implied. Token-level numbers come from re-exporting the pools with this overlay
    (``bc_export.py --mask-overlay``), because only the renderer knows token counts.
    """
    ledger = load_ledger()
    labels_path = REVIEW / "model_labels.jsonl"
    labels = load_jsonl(labels_path)
    if not labels:
        raise SystemExit("no model labels — run --ingest first")
    if require_complete:
        packet_paths = packet_index_paths()
        records = load_records()
        active = []
        problems = []
        by_id = {label["sample_id"]: label for label in labels}
        for sample, row in ledger.items():
            if row["decision"] != "ADMIT":
                continue
            label = by_id.get(sample)
            packet_file = packet_paths.get(f"MR-{sample}")
            if label is None or packet_file is None:
                problems.append(f"{sample}: missing model review")
                continue
            packet = json.loads(packet_file.read_text(encoding="utf-8"))
            current = packet_for(sample, row, records[row["instance_id"]])
            if (current["packet_sha256"] != packet["packet_sha256"]
                    or label.get("packet_sha256") != packet["packet_sha256"]
                    or not label.get("review_config_sha256")):
                problems.append(f"{sample}: stale packet or review configuration")
                continue
            submission = {"packet_id": label["packet_id"], "verdict": label["verdict"],
                          "labels": json.loads(label["label_json"])}
            issues = validate_submission(submission, packet)
            if issues:
                problems.append(f"{sample}: {issues}")
            else:
                from model_review_bulk import make_review_config, compute_review_config_sha256
                expected_config = make_review_config()
                if (label.get("review_config") != expected_config or
                        label.get("review_config_sha256") != compute_review_config_sha256(expected_config)):
                    problems.append(f"{sample}: reviewer request does not match frozen max configuration")
                else:
                    active.append(label)
        if problems:
            raise SystemExit("incomplete model review: " + "; ".join(problems[:10]))
        labels = active
    by_sample = {label["sample_id"]: label for label in labels}
    # Negative supervision is a property of the source action, shared by all views.
    source_masks = defaultdict(set)
    sample_holds = defaultdict(set)
    source_quarantine = {}
    for label in labels:
        sample = label["sample_id"]
        row = ledger.get(sample) or {}
        if row.get("decision") != "ADMIT":
            continue
        iid = label["instance_id"]
        if label["verdict"] == "QUARANTINE":
            source_quarantine[iid] = label
            continue
        if label["verdict"] == "REVIEW":
            sample_holds[sample].update(row.get("keep_msgs") or [])
        source_masks[iid].update(int(item["msg"]) for item in json.loads(label["label_json"])
                                 if item["label"] == "MASK")
        sample_holds[sample].update(int(item["msg"]) for item in json.loads(label["label_json"])
                                    if item["label"] == "REVIEW")
    excluded, sample_masks = [], {}
    for sample_id, row in sorted(ledger.items()):
        if row.get("decision") != "ADMIT":
            continue
        iid = row.get("instance_id") or sample_id.split(":", 1)[-1]
        if iid in source_quarantine:
            label = source_quarantine[iid]
            excluded.append({"sample_id": sample_id, "pool": row["pool"],
                             "instance_id": iid, "verdict": "QUARANTINE",
                             "reviewer_model": label["reviewer_model"], "notes": label["notes"]})
        else:
            dropped = sorted(set(row.get("keep_msgs") or []) & (source_masks.get(iid, set()) | sample_holds.get(sample_id, set())))
            if dropped:
                sample_masks[sample_id] = dropped
    excluded_ids = {item["sample_id"] for item in excluded}
    pools: dict[str, dict] = {}
    verdicts: dict[str, Counter] = {}
    for label in labels:
        verdicts.setdefault(label["pool"], Counter())[label["verdict"]] += 1
    for row in ledger.values():
        entry = pools.setdefault(row["pool"], {
            "admitted": 0, "reviewed": 0, "reviewed_admitted": 0, "quarantined": 0,
            "keep_messages": 0, "model_masked_messages": 0, "final_admitted": 0})
        label = by_sample.get(row["sample_id"])
        if label is not None:
            entry["reviewed"] += 1
        if row["decision"] != "ADMIT":
            continue
        entry["admitted"] += 1
        entry["keep_messages"] += len(row.get("keep_msgs") or [])
        if label is not None:
            entry["reviewed_admitted"] += 1
        if row["sample_id"] in excluded_ids:
            entry["quarantined"] += 1
            continue
        if set(row.get("keep_msgs") or []) - set(sample_masks.get(row["sample_id"], [])):
            entry["final_admitted"] += 1
        entry["model_masked_messages"] += len(sample_masks.get(row["sample_id"], []))
    for entry in pools.values():
        # coverage is about admitted rows a model actually read; REVIEW rows never
        # carried a slot, so counting them here would inflate C3 to over 100%.
        entry["review_coverage"] = (round(entry["reviewed_admitted"] / entry["admitted"], 4)
                                    if entry["admitted"] else 0.0)
        entry["masked_fraction"] = (round(entry["model_masked_messages"] /
                                         entry["keep_messages"], 4)
                                    if entry["keep_messages"] else 0.0)
    overlay = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": str(labels_path),
        "labels_sha256": sha256(labels_path.read_bytes()).hexdigest(),
        "reviewed_samples": len(labels),
        "verdicts": {pool: dict(counts) for pool, counts in sorted(verdicts.items())},
        "pools": {pool: pools[pool] for pool in sorted(pools)},
        "excluded_by_model": excluded,
        "sample_masks": dict(sorted(sample_masks.items())),
        "complete_review_required": require_complete,
        "reviewed_sample_ids": sorted(by_sample),
        "source_masks": {iid: sorted(msgs) for iid, msgs in sorted(source_masks.items())},
        "source_quarantine": sorted(source_quarantine),
        "bc_rule_ledger_sha256": _rule_digest(POOL / "bc_rule_ledger.jsonl"),
        "a_rule_ledger_sha256": _rule_digest(POOL / "a_rule_ledger.jsonl"),
        "note": ("QUARANTINE removes the sample; per-message MASK/REVIEW removes only "
                 "those messages from keep_msgs. bc_export.py --mask-overlay applies "
                 "this file to the payloads and recomputes supervised tokens; the "
                 "deterministic ledger stays unchanged and checkable."),
    }
    (REVIEW / "overlay.json").write_text(
        json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8")
    return overlay


def check() -> int:
    problems = []
    packet_paths = packet_index_paths()
    if not packet_paths:
        print("no packet index — run: python results/model_review.py --build")
        return 1
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
        # Check copied judge fields, not words in agent-visible evidence (a model
        # mentioning a gold search is evidence to quarantine, not proof of copied metadata).
        def judge_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"eval_result", "FAIL_TO_PASS", "PASS_TO_PASS", "gold_patch"}:
                        yield key
                    yield from judge_keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from judge_keys(child)
        for forbidden in judge_keys(packet):
            problems.append(f"{packet_id}: packet contains copied judge field {forbidden!r}")
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
            elif row.get("decision") == "ADMIT":
                continue
            elif row.get("decision") == "REJECT" and "MODEL_QUARANTINE" in (
                    row.get("reasons") or []):
                continue   # bc_export already composed the overlay into the ledger
            else:
                problems.append(f"overlay excludes {item.get('sample_id')}, ledger says "
                                f"{row.get('decision')}")
        for sample, messages in (overlay.get("sample_masks") or {}).items():
            row = ledger.get(sample) or {}
            if not row:
                problems.append(f"overlay masks unknown sample {sample}")
                continue
            known = {int(m) for m in row.get("keep_msgs") or []}
            known |= {int(msg) for msg, reason in row.get("masked_msgs") or []
                      if reason == "MODEL_REVIEW"}
            outside = [msg for msg in messages if msg not in known]
            if outside:
                problems.append(f"overlay masks {outside} that {sample} never supervised")
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
        overlay = apply_review(require_complete=True)
        brief = {"reviewed_samples": overlay["reviewed_samples"],
                 "pools": {pool: {key: entry[key] for key in
                                  ("admitted", "reviewed", "reviewed_admitted",
                                   "quarantined", "masked_fraction", "final_admitted")}
                           for pool, entry in overlay["pools"].items()},
                 "masked_samples": len(overlay["sample_masks"]),
                 "masked_messages": sum(len(msgs) for msgs in overlay["sample_masks"].values()),
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
