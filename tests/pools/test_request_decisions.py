import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results"))
sys.path.insert(0, str(ROOT.parent / "training" / "swe_world"))
from to_sft_qwen import Renderer, build, build_decision
from tokenized_sft import validate_row

AVAILABLE = (
    Path("E:/Deeplearning/local_train/model/Qwen3.5-4B/tokenizer.json").exists()
    and importlib.util.find_spec("tokenizers") and importlib.util.find_spec("jinja2")
)


@unittest.skipUnless(AVAILABLE, "local Qwen tokenizer assets required")
class RequestDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer()

    def view(self):
        return {"messages": [
            {"role": "system", "content": "Repair this workspace."},
            {"role": "user", "content": "Fix the task."},
        ], "tools": [{"type": "function", "function": {
            "name": "read", "description": "读取 <file>", "parameters": {"type": "object"}}}]}

    def test_failed_call_is_zero_loss_but_context_is_retained(self):
        view = self.view()
        assistant = {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call1", "type": "function",
            "function": {"name": "read", "arguments": {"file_path": "missing.py"}}}]}
        sample = build_decision(self.renderer, view, assistant, [
            {"role": "tool", "tool_call_id": "call1", "content": "Error: file not found"}],
            sample_id="case:1", response_status="completed")
        self.assertEqual(sum(sample["mask"]), 0)
        self.assertEqual(sample["masked_response_reason"], "TOOL_CONTRACT_ERROR")
        self.assertGreater(sample["response_tokens"], 0)

    def test_serving_prefix_is_immutable_at_bpe_generation_boundary(self):
        view = self.view()
        assistant = {"role": "assistant", "content": "The fix is ready."}
        sample = build_decision(self.renderer, view, assistant, sample_id="case:1", response_status="completed")
        prefix = self.renderer.render_request(view["messages"], view["tools"])
        expected = self.renderer.tokenizer.encode(prefix, add_special_tokens=False).ids
        self.assertEqual(sample["ids"][:sample["prompt_tokens"]], expected)
        self.assertFalse(any(sample["mask"][:sample["prompt_tokens"]]))
        self.assertTrue(all(sample["mask"][sample["prompt_tokens"]:]))
        self.assertFalse(sample["training_eligible"])
        self.assertEqual(sample["keep_msgs"], [2])
        self.assertEqual(sample["spans"], {"2": [sample["prompt_tokens"], len(sample["ids"]) - 1]})
        with self.assertRaisesRegex(ValueError, "training_eligible=false"):
            validate_row(sample, 65536)
        self.assertEqual(validate_row({**sample, "training_eligible": True}, 65536),
                         (len(sample["ids"]), sample["response_tokens"]))
        self.assertIn("读取 <file>", prefix)

    def test_control_user_changes_history_view_without_losing_controls(self):
        view = self.view()
        view["messages"] += [
            {"role": "assistant", "content": "continue", "reasoning_raw": "HISTORY_PRIVATE_MARKER"},
            {"role": "user", "content": "APPLY_CHANGE_NOW"},
        ]
        rendered = self.renderer.render_request(view["messages"], view["tools"])
        self.assertIn("APPLY_CHANGE_NOW", rendered)
        self.assertNotIn("HISTORY_PRIVATE_MARKER", rendered)

    def test_legacy_build_and_incomplete_or_overlong_decisions(self):
        messages = [{"role": "user", "content": "Fix it"},
                    {"role": "assistant", "content": "Done", "reasoning_raw": "I checked."}]
        ids, mask, _, spans = build(self.renderer, messages, with_spans=True)
        self.assertEqual(len(ids), len(mask))
        self.assertGreater(sum(mask), 0)
        self.assertIn(1, spans)
        for status, limit, expected in [("incomplete", 65536, "INCOMPLETE_RESPONSE"),
                                        ("completed", 1, "OVER_LENGTH")]:
            sample = build_decision(self.renderer, self.view(), messages[-1], sample_id="case:2",
                                    response_status=status, max_tokens=limit)
            self.assertEqual(sample["masked_response_reason"], expected)
            self.assertFalse(any(sample["mask"]))
            self.assertGreater(sample["prompt_tokens"], 1)
            self.assertEqual(sample["keep_msgs"], [])
            self.assertEqual(sample["spans"], {})

    def test_missing_receipt_and_summary_are_not_positive(self):
        call = {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function",
                "function": {"name": "read", "arguments": {"file_path": "x"}}}]}
        for assistant, reason in [(call, "MISSING_TOOL_RESULT"),
                                  ({"role": "assistant", "content": "Done", "reasoning_summary": "summary"},
                                   "SUMMARY_IS_NOT_RAW_REASONING"),
                                  ({"role": "assistant", "content": "Done", "reasoning_opaque": True},
                                   "OPAQUE_REASONING_UNAVAILABLE")]:
            sample = build_decision(self.renderer, self.view(), assistant, sample_id="case:3",
                                    response_status="completed")
            self.assertEqual(sample["masked_response_reason"], reason)
            self.assertFalse(any(sample["mask"]))
        success = build_decision(self.renderer, self.view(), call,
                                 [{"role": "tool", "tool_call_id": "x", "content": "file contents"}],
                                 sample_id="case:4", response_status="completed")
        self.assertGreater(sum(success["mask"]), 0)
        invalid = {**call, "tool_calls": [{"id": "x", "type": "function",
                   "function": {"name": "unknown", "arguments": {}}}]}
        sample = build_decision(self.renderer, self.view(), invalid,
                                [{"role": "tool", "tool_call_id": "x", "content": "anything"}],
                                sample_id="case:5", response_status="completed")
        self.assertEqual(sample["masked_response_reason"], "INVALID_TOOL_CALL")
        self.assertFalse(any(sample["mask"]))


if __name__ == "__main__":
    unittest.main()
