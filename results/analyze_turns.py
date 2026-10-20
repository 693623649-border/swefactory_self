"""Token composition of multi-turn factory trajectories.

Separates what the student must learn (assistant turns, including tool-call
arguments) from what is context only (tool observations), and shows how the
context grows turn by turn.
"""
import json
import glob
import os
import sys

BASE = r"E:\Deeplearning\local_train\swefactory_self\results"
QWEN = r"E:\Deeplearning\local_train\model\Qwen3.5-4B\tokenizer.json"


def main(run_glob: str) -> None:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(QWEN)

    def count(text: str) -> int:
        return len(tok.encode(text or "", add_special_tokens=False).ids)

    runs = sorted([p for p in glob.glob(os.path.join(BASE, run_glob, "*")) if os.path.isdir(p)],
                  key=os.path.getmtime)
    run = runs[-1]
    records = [json.loads(line) for line in
               open(os.path.join(run, "trajectories.jsonl"), encoding="utf-8")]
    print("run:", os.path.basename(run), "| records:", len(records))
    totals = {"assistant": 0, "tool": 0, "system": 0, "user": 0,
              "tool_args": 0, "tool_output": 0, "turns": 0}
    for record in records:
        per_role = {}
        turns = growth = 0
        context = 0
        series = []
        for message in record["messages"]:
            role = message["role"]
            text = message.get("content") or ""
            if role == "assistant" and message.get("tool_calls"):
                text += json.dumps(message["tool_calls"], ensure_ascii=False)
                totals["tool_args"] += count(json.dumps(message["tool_calls"], ensure_ascii=False))
            if role == "tool":
                totals["tool_output"] += count(text)
            per_role[role] = per_role.get(role, 0) + count(text)
            context += count(text)
            if role == "assistant":
                turns += 1
                series.append(context)
        for key in ("assistant", "tool", "system", "user"):
            totals[key] += per_role.get(key, 0)
        totals["turns"] += turns
        print(f'{record["instance_id"][:38]:38} ok={int(record["success"])} turns={turns:3} '
              f'asst={per_role.get("assistant",0):6} tool={per_role.get("tool",0):6} '
              f'ctx_end={series[-1] if series else 0:6} growth={series[-1]-series[0] if len(series)>1 else 0:6}')
    total = sum(totals[k] for k in ("assistant", "tool", "system", "user")) or 1
    print("\ntotals:", json.dumps(totals, ensure_ascii=False))
    print(f"share: assistant={totals['assistant']/total:.3f} tool={totals['tool']/total:.3f} "
          f"system={totals['system']/total:.3f} user={totals['user']/total:.3f}")
    print(f"tool observations are {totals['tool_output']/(total or 1):.3f} of all tokens; "
          f"untrained context = {(totals['tool']+totals['system']+totals['user'])/total:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dsh_pilot")
