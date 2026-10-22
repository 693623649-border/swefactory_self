"""Render factory trajectories through the student's Qwen template with a mask.

Native tool calling means the assistant turn carries ``tool_calls`` instead of
text, so a content-length estimate is meaningless: the supervised span is the
rendered assistant block, tool calls included. Boundaries come from the template
itself and are attributed by character overlap, because consecutive tool messages
merge into one block and BPE merges can cross boundaries.
"""
from __future__ import annotations
import argparse
import bisect
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
            # The factory stores model reasoning as `reasoning_raw`; the student
            # template only renders `reasoning_content`. Dropping this mapping
            # silently trains a non-thinking model.
            reasoning = copy.pop("reasoning_raw", None) or copy.get("reasoning_content")
            if reasoning:
                copy["reasoning_content"] = reasoning
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


class Rendered:
    """One rendering of a trajectory, reusable for several supervision plans.

    Pool B/C export renders the same instance for more than one pool, so the
    template walk and tokenisation happen once per instance and only the mask is
    recomputed per plan. ``owners[token]`` is the message block a token belongs to.
    """

    def __init__(self, renderer: Renderer, messages) -> None:
        self.messages = messages
        self.grouped = blocks(messages)
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"),
                        default=-1)
        starts, previous = [], ""
        for block in self.grouped:
            end = block["indices"][-1] + 1
            text = renderer.prefix_text(messages[:end], before_last_query=end - 1 <= last_user)
            if not text.startswith(previous):
                raise ValueError(f"not prefix-stable at block ending {end}")
            starts.append(len(previous))
            previous = text
        self.text = previous
        self.bounds = starts + [len(previous)]
        encoding = renderer.tokenizer.encode(self.text, add_special_tokens=False)
        self.ids = encoding.ids
        self.owners: list[int] = []
        bounds = self.bounds
        count = len(self.grouped)
        for start, end in encoding.offsets:
            # block intervals are contiguous and sorted, so only blocks intersecting
            # [start, end) can win; scanning them is exact and avoids the O(n) walk
            first = bisect.bisect_right(bounds, start) - 1
            last = bisect.bisect_right(bounds, end) - 1
            first = min(max(first, 0), count - 1)
            last = min(max(last, first), count - 1)
            owner, best = first, -1
            for position in range(first, last + 1):
                overlap = min(end, bounds[position + 1]) - max(start, bounds[position])
                if overlap > best:
                    owner, best = position, overlap
            self.owners.append(owner)

    def mask(self, keep=None) -> list[int]:
        keep_set = set(keep) if keep is not None else None
        supervision = []
        for block in self.grouped:
            allowed = block["role"] == "assistant"
            if allowed and keep_set is not None:
                allowed = len(block["indices"]) == 1 and block["indices"][0] in keep_set
            supervision.append(1 if allowed else 0)
        return [supervision[owner] for owner in self.owners]

    def spans(self, mask=None) -> dict:
        """``{message_index: (first_token, last_token)}`` for supervised blocks."""
        values = self.mask() if mask is None else mask
        spans: dict[int, tuple[int, int]] = {}
        for token, (owner, value) in enumerate(zip(self.owners, values)):
            if not value:
                continue
            message_index = self.grouped[owner]["indices"][0]
            first, last = spans.get(message_index, (token, token))
            spans[message_index] = (min(first, token), max(last, token))
        return spans


def build(renderer: Renderer, messages, keep=None, with_spans: bool = False):
    """Render a trajectory and return ``(ids, mask, blocks)``.

    ``keep`` optionally restricts supervision to the given message indices; only
    assistant blocks can ever be supervised, so any other index is ignored.
    ``with_spans`` additionally returns the per-message token spans, which lets a
    pool exporter prove the mask covers exactly the intended messages.
    """
    rendered = Rendered(renderer, messages)
    mask = rendered.mask(keep)
    if with_spans:
        return rendered.ids, mask, len(rendered.grouped), rendered.spans(mask)
    return rendered.ids, mask, len(rendered.grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=81920,
                        help="drop samples longer than this (80*1024 window)")
    args = parser.parse_args()
    renderer = Renderer()
    records = [json.loads(line) for line in
               open(os.path.join(args.run, "trajectories.jsonl"), encoding="utf-8")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    totals = {"records": 0, "dropped": 0, "over_length": 0, "no_think": 0,
              "tokens": 0, "supervised": 0, "tool_blocks": 0}
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            if not record.get("success"):
                totals["dropped"] += 1
                continue
            if not any((m.get("reasoning_raw") or "").strip() for m in record["messages"]):
                totals["no_think"] += 1     # thinking was requested; empty means it was lost
                continue
            ids, mask, count = build(renderer, record["messages"])
            if len(ids) > args.max_tokens:
                totals["over_length"] += 1
                continue
            rendered = renderer.render(record["messages"])
            # Require at least one *non-empty* think block; an empty one on some
            # turn is fine (the template emits it whenever reasoning is absent).
            empty = rendered.count("<think>\n\n</think>")
            non_empty = rendered.count("<think>") - empty
            if non_empty <= 0:
                totals["no_think"] += 1
                continue
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
