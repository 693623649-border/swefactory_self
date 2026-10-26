"""Full-pool model review: put every ADMIT row of every pool through a reviewer model.

``model_review.py`` samples 50 packets so a human-sized spot check can be compared with
the deterministic labels.  That sample cannot stand in for the pool: an 8.7% coverage
says nothing about the other 91%, and the whole point of the model stage is to catch the
semantic failures the rules cannot express (a verified-but-hollow fix, a report that
claims more than the visible transcript shows, a pass obtained on a shadow copy of the
tests).

This module does the same thing at pool scale and nothing else:

  * ``--build``  make one agent-view packet per ADMIT row (same ``packet_for`` the
                 sampled stage uses, same redaction) into ``review/bulk/packets`` and
                 write ``review/bulk/bulk_index.json`` with a snapshot sha. Fails
                 non-zero if any target is missing records.
  * ``--run``    send each packet to the reviewer API, validate the answer with the
                 same ``validate_submission`` the sampled stage uses, retry with the
                 concrete problem list when an answer is malformed (never overwriting
                 a mismatched packet_id), log per-call API usage immediately, and
                 append every accepted answer with request receipts to
                 ``review/bulk/submissions.jsonl``. Skips only when packet_sha and
                 frozen review_config_sha both match and schema is valid.
  * ``--ingest`` fold only the latest valid fresh submission for current targets into
                 ``model_labels.jsonl`` + SQLite through ``model_review.ingest``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

RESULTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RESULTS))
import model_review as mr  # noqa: E402

BULK = mr.REVIEW / "bulk"
PACKETS = BULK / "packets"
SUBMISSIONS = BULK / "submissions.jsonl"
USAGE = BULK / "usage.jsonl"

API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_EFFORT = "max"
DEFAULT_MAX_TOKENS = 65536
DEFAULT_CONCURRENCY = 6
EFFORT_CHOICES = ("low", "medium", "high", "max")

SYSTEM_PROMPT = """You are a strict dataset-quality reviewer for code-agent trajectories.

You receive ONE review packet as JSON. It contains only material the original agent could
see: the task statement, its own tool calls and tool observations, a trimmed context log,
and the message indices that are candidates for supervision. It never contains the
judge's tests, the gold patch or evaluation results — do not speculate about them.

Follow packet.instructions and answer packet.question. Return exactly one JSON object:

{
  "packet_id": "<the packet's packet_id, verbatim>",
  "verdict": "KEEP" | "MASK" | "REVIEW" | "QUARANTINE",
  "labels": [{"msg": <int in candidate_msgs>, "label": "KEEP"|"MASK"|"REVIEW",
              "reason_codes": ["<1-3 codes from reason_vocab>"],
              "evidence_msgs": [<ints from visible_msgs>]}],
  "notes": "<=400 chars about the single most error-prone part of this packet"
}

Rules:
- Label EVERY index in candidate_msgs exactly once; the grader rejects partial answers.
- "KEEP" = this action is worth imitating; "MASK" = it should not teach the model
  (mechanical repetition, ungrounded edit, test evasion, false claim, no progress).
  "REVIEW" = genuinely need a human. Use it sparingly.
- Sample verdict: KEEP = the whole trajectory is honest and worth learning from;
  MASK = at least one action must be excluded; QUARANTINE = judge/gold leakage,
  fabricated evidence or the wrong task; REVIEW = not decidable from this packet.
- Trying to get the fix from outside the task is not exploration: searching unreachable
  git objects or other refs for future commits (git fsck --unreachable/--lost-found,
  git log --all for newer commits), or reading/downloading an installed or newer release
  of the same project. Label it MASK with MASK_FUTURE_COMMIT_PROBE when nothing usable came
  back; use QUARANTINE_LEAK once reference or future-fix code became visible and a later
  action used it. Reading the repo's existing history or third-party dependencies is fine.
- If any label carries a QUARANTINE_* reason code, the sample verdict MUST be QUARANTINE;
  masking single messages cannot remove a leak, and the grader rejects anything else.
- reason_codes must come from reason_vocab. evidence_msgs must be visible_msgs.
- Judge only what the packet shows. Saying less is better than inventing evidence.
- Output the JSON object only, with no prose, no code fences.
"""

_USAGE_LOCK = threading.Lock()
_SUBMISSIONS_LOCK = threading.Lock()


# --- config & freshness ----------------------------------------------------

def make_review_config(
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = API_URL,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict:
    """Canonical frozen review configuration dict."""
    return {
        "system_prompt": system_prompt,
        "base_url": base_url,
        "model": model,
        "effort": effort,
        "max_tokens": max_tokens,
    }


def compute_review_config_sha256(config: dict) -> str:
    """Deterministic hash of frozen review configuration."""
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True)
    return sha256(canonical.encode("utf-8")).hexdigest()


def is_submission_fresh(submission: dict, packet: dict, config_sha: str) -> bool:
    """Check if submission is valid, schema-compliant, and fresh.

    Requires identical packet_sha256 and review_config_sha256. Old sample reviews
    or historical submissions without sha hashes never qualify.
    """
    if not isinstance(submission, dict):
        return False
    if submission.get("packet_id") != packet.get("packet_id"):
        return False
    packet_sha = submission.get("packet_sha256")
    if not packet_sha or packet_sha != packet.get("packet_sha256"):
        return False
    sub_config_sha = submission.get("review_config_sha256")
    if not sub_config_sha or sub_config_sha != config_sha:
        return False
    problems = mr.validate_submission(submission, packet)
    return not bool(problems)


# --- usage logging ---------------------------------------------------------

def record_usage(
    packet_id: str,
    model: str,
    raw_usage: dict | None,
    usage_path: Path | None = None,
) -> dict:
    """Record one usage line immediately per API call. Never includes API keys."""
    target_path = usage_path if usage_path is not None else USAGE
    got = raw_usage or {}
    prompt_tokens = got.get("prompt_tokens") or 0
    completion_tokens = got.get("completion_tokens") or 0
    total_tokens = got.get("total_tokens") or (prompt_tokens + completion_tokens)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packet_id": packet_id,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    for k, v in got.items():
        if k not in entry and "key" not in k.lower() and "secret" not in k.lower():
            entry[k] = v
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _USAGE_LOCK:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    return entry


# --- selection / packet building -------------------------------------------

def targets() -> list[dict]:
    """Every ADMIT row, newest ledger state, in a deterministic order."""
    ledger = mr.load_ledger()
    rows = [row for row in ledger.values() if row.get("decision") == "ADMIT"]
    return sorted(rows, key=lambda row: (row["pool"], row["sample_id"]))


def build(bulk_dir: Path | None = None, packets_dir: Path | None = None) -> dict:
    target_bulk = bulk_dir if bulk_dir is not None else BULK
    target_packets = packets_dir if packets_dir is not None else PACKETS
    records = mr.load_records()
    target_packets.mkdir(parents=True, exist_ok=True)
    entries, skipped = [], []
    t_rows = targets()
    for row in t_rows:
        sample_id = row["sample_id"]
        record = records.get(row["instance_id"])
        if record is None:
            skipped.append(sample_id)
            continue
        packet = mr.packet_for(sample_id, row, record)
        path = target_packets / mr.packet_path(packet["packet_id"]).name
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
        entries.append({
            "packet_id": packet["packet_id"],
            "path": str(path),
            "sample_id": sample_id,
            "pool": row["pool"],
            "packet_sha256": packet["packet_sha256"],
        })
    if skipped:
        raise SystemExit(
            f"bulk build failed: {len(skipped)} targets missing records: {skipped[:10]}"
        )
    snapshot = sha256(
        "".join(sorted(f"{e['sample_id']}:{e['packet_sha256']}" for e in entries)).encode("utf-8")
    ).hexdigest()[:16]
    index = {
        "review_version": mr.REVIEW_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": snapshot,
        "target_count": len(t_rows),
        "packets": entries,
    }
    index_path = target_bulk / "bulk_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    sizes = [os.path.getsize(entry["path"]) for entry in entries]
    return {
        "packets": len(entries),
        "snapshot": snapshot,
        "total_bytes": sum(sizes),
        "max_bytes": max(sizes) if sizes else 0,
        "index": str(index_path),
    }


# --- reviewer call ----------------------------------------------------------

def _post(
    payload: dict,
    api_key: str,
    base_url: str = API_URL,
    timeout: float = 900.0,
) -> dict:
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_answer(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if len(text.split("```")) > 1 else text
        text = text.removeprefix("json").strip()
    return json.loads(text)


def review_one(
    packet: dict,
    api_key: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = API_URL,
    usage_path: Path | None = None,
    sleep_fn: callable = time.sleep,
) -> dict:
    """Review one packet; retry with validator's problem list on bad output.

    Never overwrites answer.packet_id. Logs per-call usage immediately.
    """
    config = make_review_config(
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        base_url=base_url,
        system_prompt=SYSTEM_PROMPT,
    )
    config_sha = compute_review_config_sha256(config)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]
    problems: list[str] = []
    for attempt in range(1, 4):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled"},
            "reasoning_effort": effort,
            "response_format": {"type": "json_object"},
        }
        # If API rejects reasoning_effort or payload, it raises directly without downgrading.
        raw = _post(payload, api_key, base_url=base_url)
        # Record usage immediately per call, even if answer parsing or validation fails later
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        call_receipt = {**(raw.get("usage") or {}), "finish_reason": choice.get("finish_reason"),
                        "response_model": raw.get("model"), "requested_effort": effort,
                        "requested_max_tokens": max_tokens, "packet_sha256": packet["packet_sha256"],
                        "review_config_sha256": config_sha,
                        "content_chars": len(message.get("content") or ""),
                        "reasoning_chars": len(message.get("reasoning_content") or "")}
        record_usage(packet["packet_id"], model, call_receipt, usage_path=usage_path)
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"{packet['packet_id']}: reviewer output budget exhausted ({max_tokens}); no schema-only retry")
        content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
        try:
            answer = _parse_answer(content or "")
        except json.JSONDecodeError as error:
            problems = [f"answer was not JSON: {error}"]
        else:
            # Do NOT overwrite answer['packet_id']! Validate and let mismatch trigger retry
            problems = mr.validate_submission(answer, packet)
            if not problems:
                answer["packet_sha256"] = packet["packet_sha256"]
                answer["review_config_sha256"] = config_sha
                answer["reviewer_model"] = model
                answer["reviewer_provider"] = "deepseek"
                answer["reviewer_effort"] = effort
                answer["provider"] = "deepseek"
                answer["effort"] = effort  # requested effort value, no false claims
                answer["review_config"] = config  # request configuration receipt
                answer.setdefault("labels", [])
                return answer
        messages = messages + [
            {"role": "assistant", "content": content or "(empty)"},
            {"role": "user", "content": "That answer was rejected: "
             + "; ".join(problems[:6]) + ". Return the full corrected JSON object only."},
        ]
        sleep_fn(2 * attempt)
    raise RuntimeError(f"{packet['packet_id']}: " + "; ".join(problems[:6]))


def run(
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = API_URL,
    bulk_dir: Path | None = None,
    submissions_path: Path | None = None,
    usage_path: Path | None = None,
) -> dict:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")

    target_bulk = bulk_dir if bulk_dir is not None else BULK
    target_submissions = submissions_path if submissions_path is not None else SUBMISSIONS
    target_usage = usage_path if usage_path is not None else USAGE

    index_path = target_bulk / "bulk_index.json"
    if not index_path.exists():
        raise SystemExit(f"bulk index missing: {index_path} — run --build first")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    config = make_review_config(
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        base_url=base_url,
        system_prompt=SYSTEM_PROMPT,
    )
    config_sha = compute_review_config_sha256(config)

    # Collect existing bulk submissions; old sample labels or missing sha cannot substitute
    existing_submissions: dict[str, list[dict]] = defaultdict(list)
    if target_submissions.exists():
        for line in target_submissions.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                sub = json.loads(line)
                if isinstance(sub, dict) and "packet_id" in sub:
                    existing_submissions[sub["packet_id"]].append(sub)
            except json.JSONDecodeError:
                continue

    done_packets = set()
    for entry in index["packets"]:
        packet_id = entry["packet_id"]
        packet_file = Path(entry["path"])
        if not packet_file.exists():
            continue
        try:
            packet = json.loads(packet_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        subs = existing_submissions.get(packet_id, [])
        for sub in reversed(subs):
            if is_submission_fresh(sub, packet, config_sha):
                done_packets.add(packet_id)
                break

    pending = [entry for entry in index["packets"] if entry["packet_id"] not in done_packets]
    if limit:
        pending = pending[:limit]
    print(f"bulk packets={len(index['packets'])} done={len(done_packets)} "
          f"pending={len(pending)} concurrency={concurrency} effort={effort} "
          f"config_sha={config_sha[:8]}", flush=True)

    target_bulk.mkdir(parents=True, exist_ok=True)
    accepted = failed = 0
    started = time.time()

    def work(entry: dict) -> tuple[dict, dict]:
        packet = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        return entry, review_one(
            packet,
            api_key,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            base_url=base_url,
            usage_path=target_usage,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(work, entry): entry for entry in pending}
        for position, future in enumerate(as_completed(futures), 1):
            entry = futures[future]
            try:
                _, answer = future.result()
            except Exception as error:  # noqa: BLE001 - one bad packet must not stop the run
                failed += 1
                print(f"[{position}/{len(pending)}] FAIL {entry['packet_id']}: "
                      f"{error}", flush=True)
                continue
            with _SUBMISSIONS_LOCK:
                with target_submissions.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(answer, ensure_ascii=False) + "\n")
                    out.flush()
            accepted += 1
            if accepted % 10 == 0 or position == len(pending):
                rate = (time.time() - started) / max(accepted, 1)
                print(f"[{position}/{len(pending)}] accepted={accepted} failed={failed} "
                      f"{rate:.1f}s/packet eta={(len(pending) - position) * rate / 60:.0f}min",
                      flush=True)

    return {
        "accepted": accepted,
        "failed": failed,
        "pending_total": len(pending),
        "submissions": str(target_submissions),
    }


def ingest(
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: str = API_URL,
    submissions_path: Path | None = None,
    index_path: Path | None = None,
) -> dict:
    """Ingest only the latest valid fresh submission for current targets."""
    target_index = index_path if index_path is not None else BULK / "bulk_index.json"
    target_submissions = submissions_path if submissions_path is not None else SUBMISSIONS
    if not target_index.exists():
        raise SystemExit(f"bulk index missing: {target_index} — run --build first")

    index = json.loads(target_index.read_text(encoding="utf-8"))
    config = make_review_config(
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        base_url=base_url,
        system_prompt=SYSTEM_PROMPT,
    )
    expected_config_sha = compute_review_config_sha256(config)

    packets_by_id: dict[str, dict] = {}
    for entry in index.get("packets") or []:
        p_path = Path(entry["path"])
        if p_path.exists():
            packets_by_id[entry["packet_id"]] = json.loads(p_path.read_text(encoding="utf-8"))

    if not target_submissions.exists():
        raise SystemExit(f"no submissions at {target_submissions} — run --run first")

    # Ingest ONLY current valid fresh selected final submission. Old snapshots preserved.
    fresh_last: dict[str, dict] = {}
    for line in target_submissions.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            sub = json.loads(line)
        except json.JSONDecodeError:
            continue
        packet_id = sub.get("packet_id")
        if packet_id not in packets_by_id:
            continue
        packet = packets_by_id[packet_id]
        if is_submission_fresh(sub, packet, expected_config_sha):
            fresh_last[packet_id] = sub

    if not fresh_last:
        return {"accepted": 0, "rejected": 0, "total_labels": 0, "fresh_selected": 0}

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for sub in fresh_last.values():
            tmp.write(json.dumps(sub, ensure_ascii=False) + "\n")

    try:
        res = mr.ingest(tmp_path)
        res["fresh_selected"] = len(fresh_last)
        return res
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=list(EFFORT_CHOICES), default=DEFAULT_EFFORT,
                        help="reasoning effort to request (low/medium/high/max; default: max)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--base-url", default=API_URL)
    parser.add_argument("--ingest", action="store_true")
    args = parser.parse_args()

    if args.build:
        print(json.dumps(build(), ensure_ascii=False, indent=2))
    elif args.run:
        result = run(limit=args.limit, concurrency=args.concurrency, model=args.model,
                     effort=args.effort, max_tokens=args.max_tokens, base_url=args.base_url)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["failed"]:
            raise SystemExit(1)
    elif args.ingest:
        print(json.dumps(
            ingest(
                model=args.model,
                effort=args.effort,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
            ),
            ensure_ascii=False,
            indent=2,
        ))
    else:
        parser.error("choose --build, --run or --ingest")


if __name__ == "__main__":
    main()
