"""Render factory trajectories through the student's Qwen template with a mask.

Native tool calling means the assistant turn carries ``tool_calls`` instead of
text, so a content-length estimate is meaningless: the supervised span is the
rendered assistant block, tool calls included. Boundaries come from the template
itself and are attributed by character overlap, because consecutive tool messages
merge into one block and BPE merges can cross boundaries.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

MODEL = Path(r"E:\Deeplearning\local_train\model\Qwen3.5-4B")


class Renderer:
    def __init__(self) -> None:
        import datetime
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
        env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False,
                                            keep_trailing_newline=True)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        env.globals["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
        env.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
        self.template = env.from_string((MODEL / "chat_template.jinja").read_text(encoding="utf-8"))
        config = json.loads((MODEL / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.bos = config.get("bos_token") or ""
        self.eos = config.get("eos_token") or ""

    def normalize(self, messages):
        out = []
        for message in messages:
            copy = dict(message)
            calls = copy.get("tool_calls")
            if calls:
                rebuilt = []
                for call in calls:
                    function = dict(call.get("function") or {})
                    args = function.get("arguments")
                    if isinstance(args, str):
                        function["arguments"] = json.loads(args)
                    rebuilt.append({**call, "function": function})
                copy["tool_calls"] = rebuilt
            out.append(copy)
        return out

    def render(self, messages) -> str:
        return self.template.render(messages=self.normalize(messages), tools=None,
                                    add_generation_prompt=False,
                                    bos_token=self.bos, eos_token=self.eos)

    def prefix_text(self, messages, before_last_query: bool) -> str:
        stub = [{"role": "user", "content": ""}]
        if before_last_query or not any(m.get("role") == "user" for m in messages):
            full = self.render(messages + stub)
            tail = self.render(stub)
            if not tail or not full.endswith(tail):
                raise ValueError("empty-user stub is not a stable suffix")
            return full[: -len(tail)]
        return self.render(messages)


def blocks(messages):
    out = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool" and out and out[-1]["role"] == "tool":
            out[-1]["indices"].append(index)
        else:
            out.append({"role": role, "indices": [index]})
    return out


def build(renderer: Renderer, messages):
    grouped = blocks(messages)
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    starts, previous = [], ""
    for block in grouped:
        end = block["indices"][-1] + 1
        text = renderer.prefix_text(messages[:end], before_last_query=end - 1 <= last_user)
        if not text.startswith(previous):
            raise ValueError(f"not prefix-stable at block ending {end}")
        starts.append(len(previous))
        previous = text
    full = previous
    bounds = starts + [len(full)]
    ids, offsets = renderer.tokenizer.encode(full, add_special_tokens=False).ids, None
    encoding = renderer.tokenizer.encode(full, add_special_tokens=False)
    ids, offsets = encoding.ids, encoding.offsets
    mask, supervision = [], [1 if b["role"] == "assistant" else 0 for b in grouped]
    for start, end in offsets:
        owner, best = 0, -1
        for position in range(len(grouped)):
            overlap = min(end, bounds[position + 1]) - max(start, bounds[position])
            if overlap > best:
                owner, best = position, overlap
        mask.append(supervision[owner])
    return ids, mask, len(grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    renderer = Renderer()
    records = [json.loads(line) for line in
               open(os.path.join(args.run, "trajectories.jsonl"), encoding="utf-8")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    totals = {"records": 0, "dropped": 0, "tokens": 0, "supervised": 0, "tool_blocks": 0}
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            if not record.get("success"):
                totals["dropped"] += 1
                continue
            ids, mask, count = build(renderer, record["messages"])
            handle.write(json.dumps({"instance_id": record["instance_id"], "ids": ids,
                                     "mask": mask}, ensure_ascii=False) + "\n")
            totals["records"] += 1
            totals["tokens"] += len(ids)
            totals["supervised"] += sum(mask)
            totals["tool_blocks"] += count
    summary = {**totals,
               "supervised_fraction": round(totals["supervised"] / totals["tokens"], 4) if totals["tokens"] else None,
               "tokens_per_sample": round(totals["tokens"] / totals["records"], 1) if totals["records"] else None,
               "supervised_per_sample": round(totals["supervised"] / totals["records"], 1) if totals["records"] else None}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
