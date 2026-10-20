"""Prove the DSH scaffold matches the evaluation tool surface. No teacher, no container.

Two checks:
1. schema parity - names, parameters, required lists and defaults against the DSH
   sources (packages/fs/tool-fs/src/{read,edit}.ts and packages/shell/tool-bash/src/index.ts);
2. format round-trip - a native tool_calls message rendered through the student's
   Qwen template must parse back to the same names and arguments, with no DSML left.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aweagent.scaffold.dsh_compat import DSHCompatAgent  # noqa: E402

MODEL = Path(r"E:\Deeplearning\local_train\model\Qwen3.5-4B")

EXPECTED = {
    "bash": {"required": ["command", "description"],
             "properties": ["command", "description"]},
    "read": {"required": ["file_path"], "properties": ["file_path", "offset", "limit"]},
    "edit": {"required": ["file_path", "old_string", "new_string"],
             "properties": ["file_path", "old_string", "new_string", "replace_all"]},
}


def check_schema() -> dict:
    agent = DSHCompatAgent("/workspace/demo")
    tools = agent.get_tools()
    names = [tool.name for tool in tools]
    assert names == ["bash", "read", "edit"], f"tool set must be bash/read/edit, got {names}"
    report = {}
    for tool in tools:
        schema = tool.schema["function"]
        params = schema["parameters"]
        expected = EXPECTED[tool.name]
        assert sorted(params["properties"]) == sorted(expected["properties"]), (tool.name, params["properties"])
        assert sorted(params.get("required", [])) == sorted(expected["required"]), (tool.name, params.get("required"))
        report[tool.name] = {"properties": sorted(params["properties"]),
                             "required": params.get("required", []),
                             "description_len": len(schema["description"])}
    # Defaults that the DSH side fixes
    read = next(t for t in tools if t.name == "read")
    assert "400" in read.parameters["properties"]["limit"]["description"]
    assert "2000" in read.parameters["properties"]["limit"]["description"]
    assert "false" in next(t for t in tools if t.name == "edit").parameters["properties"]["replace_all"]["description"]
    # Forbidden actions must not exist
    assert all(tool.name not in ("finish", "think", "str_replace_editor", "execute_bash") for tool in tools)
    return report


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
    print(json.dumps({"schema": check_schema(), "round_trip": check_round_trip()},
                     ensure_ascii=False, indent=2))
    print("PASS: tool surface matches DSH and survives the student template round-trip")


if __name__ == "__main__":
    main()
