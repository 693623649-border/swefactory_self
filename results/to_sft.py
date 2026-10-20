"""Convert factory trajectories into SFT samples with an assistant-only mask.

The factory already emits the scaffold's native format (tool calls live inside the
assistant text), so nothing is rewritten here: assistant turns are simply the
supervised spans and everything else is context.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

QWEN_TOKENIZER = r"E:\Deeplearning\local_train\model\Qwen3.5-4B\tokenizer.json"


def load_tokenizer():
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(QWEN_TOKENIZER), "qwen3.5-4b"
    except Exception:
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="factory results dir (contains trajectories.jsonl)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-success", action="store_true", default=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in open(os.path.join(args.run, "trajectories.jsonl"), encoding="utf-8")]
    tokenizer, tokenizer_name = load_tokenizer()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    totals = {"messages": 0, "assistant_messages": 0, "chars": 0, "supervised_chars": 0,
              "tokens": 0, "supervised_tokens": 0}
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            if args.require_success and not record.get("success"):
                dropped += 1
                continue
            messages = [{"role": m["role"], "content": m.get("content") or ""} for m in record["messages"]]
            labels = [1 if m["role"] == "assistant" else 0 for m in messages]
            if not any(labels):
                dropped += 1
                continue
            chars = sum(len(m["content"]) for m in messages)
            sup_chars = sum(len(m["content"]) for m, l in zip(messages, labels) if l)
            tokens = sup_tokens = 0
            if tokenizer is not None:
                tokens = sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) for m in messages)
                sup_tokens = sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids)
                                 for m, l in zip(messages, labels) if l)
            sample = {"instance_id": record["instance_id"], "messages": messages, "labels": labels,
                      "score": record.get("score"), "steps": (record.get("stats") or {}).get("steps"),
                      "patch_chars": len(record.get("patch") or ""),
                      "stats": {"messages": len(messages), "assistant_messages": sum(labels),
                                "chars": chars, "supervised_chars": sup_chars,
                                "tokens": tokens, "supervised_tokens": sup_tokens}}
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1
            for key in totals:
                totals[key] += sample["stats"][key]
    summary = {"run": args.run, "tokenizer": tokenizer_name, "kept": kept, "dropped": dropped,
               "totals": totals,
               "supervised_fraction": round(totals["supervised_tokens"] / totals["tokens"], 4) if totals["tokens"] else None,
               "per_sample": {k: (round(v / kept, 1) if kept else None) for k, v in totals.items()}}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
