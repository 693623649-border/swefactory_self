"""Compare factory tool schemas with an actual eval request receipt.

No teacher or container is started. A static list of parameter names cannot
establish parity. The optional Qwen marker check is a serialization smoke only.
"""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aweagent.scaffold.dsh_compat import DSHCompatAgent  # noqa: E402

MODEL = Path(r"E:\Deeplearning\local_train\model\Qwen3.5-4B")

def check_schema(eval_tools: list[dict]) -> dict:
    """Compare the full model-visible schemas from a captured eval request."""
    if not eval_tools:
        raise ValueError("eval request contains no tool schemas")
    factory_tools = [tool.schema for tool in DSHCompatAgent("/workspace/demo").get_tools()]
    factory = {tool["function"]["name"]: tool for tool in factory_tools}
    actual = {tool["function"]["name"]: tool for tool in eval_tools}
    differences = []
    if list(factory) != list(actual):
        differences.append("tool_order_or_names")
    for name in sorted(set(factory) | set(actual)):
        if factory.get(name) != actual.get(name):
            differences.append(name)
    def digest(value):
        return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"schema_equal": not differences, "differences": differences,
            "factory_schema_sha256": digest(factory_tools),
            "eval_schema_sha256": digest(eval_tools),
            "scope": "Tool schemas only; does not prove tool execution, prompts, or token parity"}


def check_round_trip() -> dict:
    from jinja2 import Environment
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    template = Environment(trim_blocks=False, lstrip_blocks=False,
                           keep_trailing_newline=True).from_string(
        (MODEL / "chat_template.jinja").read_text(encoding="utf-8"))
    env_globals = {"raise_exception": lambda m: (_ for _ in ()).throw(ValueError(m)),
                   "tojson": lambda v, **k: json.dumps(v, ensure_ascii=False),
                   "strftime_now": lambda f: "2026-10-20"}
    template.globals.update(env_globals)

    messages = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "bash", "arguments": {"command": "pytest -q tests/x.py", "description": "run target test"}}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "1 passed"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {
                "name": "edit", "arguments": {"file_path": "/workspace/demo/a.py",
                                              "old_string": "a = 1", "new_string": "a = 2"}}},
        ]},
    ]
    rendered = template.render(messages=messages, tools=None, add_generation_prompt=False,
                               bos_token="", eos_token="<|im_end|>")
    assert "<tool_call>" in rendered and "<function=bash>" in rendered, "Qwen tool_call block missing"
    assert "DSML" not in rendered, "factory DSML markers leaked into the Qwen rendering"
    ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    for needle in ("<function=bash>", "<parameter=command>", "<function=edit>", "<parameter=file_path>"):
        assert needle in decoded, f"lost after tokenize round-trip: {needle}"
    return {"rendered_chars": len(rendered), "tokens": len(ids),
            "tool_blocks": rendered.count("<tool_call>"),
            "dsml_markers": rendered.count("DSML")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-request", required=True, type=Path,
                        help="Raw chat request, or requests.json from the DSH contract smoke")
    parser.add_argument("--round-trip", action="store_true",
                        help="Also run the local Qwen serialization smoke (not token parity)")
    args = parser.parse_args()
    raw = args.eval_request.read_bytes()
    request = json.loads(raw)
    if isinstance(request, list):
        request = request[0]
    report = check_schema(request.get("tools") or request.get("toolSchemas") or [])
    report["eval_request_sha256"] = sha256(raw).hexdigest()
    if args.round_trip:
        report["serialization_smoke"] = check_round_trip()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["schema_equal"] else 1)


if __name__ == "__main__":
    main()
