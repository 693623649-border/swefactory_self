"""Real file I/O with synthetic completed results; no runtime or model is opened."""
from __future__ import annotations

import asyncio
import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from aweagent.core.agent.loop import AgentResult
from aweagent.core.agent.trajectory import Action, Trajectory
from aweagent.core.llm.config import LLMConfig
from aweagent.core.llm.types import Message, ToolCall
from aweagent.core.runtime.config import RuntimeConfig
from aweagent.core.task.runner import TaskRunner, _append_jsonl, _build_trajectory_record
from aweagent.core.task.types import EvalResult, Instance, TaskResult


class RunnerArchiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "中文路径✨"
        self.root.mkdir()
        self.instance = Instance(id="中文✨", dataset_id="公开数据", metadata={"task_type": "合成"})
        self.task = Mock()
        self.task.default_evaluator.return_value = None
        self.task.get_instances.return_value = [self.instance]
        self.runner = TaskRunner(
            self.task, Mock(), LLMConfig(model="synthetic/归档"), RuntimeConfig(),
            max_retries=3, output_path=self.root,
            config_snapshot={"说明": "中文✨", "api_key": "synthetic-secret"},
        )
        trajectory = Trajectory()
        trajectory.add_step(1, Action(type="tool_call", content="编辑✨",
                                     tool_calls=[{"id": "调用", "function": {"name": "read"}}],
                                     reasoning_text="推理✨"), ["观察✨"])
        self.native = AgentResult(
            trajectory=trajectory, patch="中文补丁✨", finish_reason="completed",
            messages=[Message(role="assistant", content="完成✨", reasoning_raw={"text": "推理✨"},
                              tool_calls=[ToolCall(id="调用", name="read", arguments='{"path":"中文"}')])],
            metadata={"stats": {"tokens": 7}, "final_answer": "答案✨",
                      "answer_provenance": {"source": "合成"}},
        )
        self.result = TaskResult(self.instance.id, self.native, EvalResult(accepted=True, score=1,
                                 details={"说明": "合成✨"}), metadata={"duration": 0.5})
        self.runner._run_instance = AsyncMock(return_value=self.result)

    async def test_utf8_roundtrip_preserves_complete_records_and_config(self):
        expected = _build_trajectory_record(self.result)
        real_open = builtins.open

        def legacy_default(*args, **kwargs):
            kwargs.setdefault("encoding", "gbk")
            return real_open(*args, **kwargs)

        with patch("aweagent.core.task.runner.open", side_effect=legacy_default):
            results = await self.runner.run_all()
        self.assertIs(results[0], self.result)
        self.assertTrue(results[0].success)
        directory = self.runner.run_dir
        data = (directory / "trajectories.jsonl").read_bytes()
        self.assertIn("✨".encode("utf-8"), data)
        self.assertEqual(len(data.splitlines()), 1)
        self.assertEqual(json.loads(data), expected)
        self.assertEqual(set(expected), {"instance_id", "success", "score", "finish_reason", "error",
                         "duration", "patch", "stats", "trajectory", "messages", "eval_result",
                         "final_answer", "answer_provenance"})
        summary = json.loads((directory / "results.jsonl").read_text(encoding="utf-8"))
        self.assertTrue(summary["success"])
        self.assertEqual(summary["archive"]["status"], "saved")
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["config"], {"说明": "中文✨"})
        self.runner._run_instance.assert_awaited_once()

    async def test_archive_io_failures_keep_native_result_and_never_retry_solve(self):
        native_receipt = self.root / "native-prediction.json"
        native_receipt.write_bytes(b'{"synthetic_native_receipt": true}\n')
        original = native_receipt.read_bytes()
        for kind in ("trajectory", "summary"):
            with self.subTest(kind=kind):
                folder = self.root / kind
                folder.mkdir()
                summary, trajectory = folder / "results.jsonl", folder / "trajectories.jsonl"
                (trajectory if kind == "trajectory" else summary).mkdir()
                self.result.error = "existing solve detail" if kind == "summary" else None
                self.runner._run_instance.reset_mock()
                with self.assertLogs("aweagent.core.task.runner", level="ERROR"):
                    result = await self.runner._run_instance_with_retry(
                        self.instance, summary, asyncio.Lock(), trajectory)
                self.assertIs(result, self.result)
                self.assertIs(result.agent_result, self.native)
                self.assertTrue(result.eval_result.accepted)
                self.assertEqual(result.agent_result.patch, "中文补丁✨")
                self.assertFalse(result.success)
                self.assertIn(f"[archive:{kind}]", result.error)
                self.assertEqual(result.metadata["archive"]["status"], "failed")
                self.assertIn(kind, result.metadata["archive"]["errors"])
                self.runner._run_instance.assert_awaited_once()
                self.assertEqual(native_receipt.read_bytes(), original)
                if kind == "trajectory":
                    saved = json.loads(summary.read_text(encoding="utf-8"))
                    self.assertFalse(saved["success"])
                    self.assertTrue(saved["archive"]["eval_accepted"])
                else:
                    self.assertIn("existing solve detail", result.error)

    async def test_completed_solve_retry_stops_before_archiving(self):
        self.runner._run_instance.side_effect = [RuntimeError("synthetic solve failure"), self.result]
        with patch("aweagent.core.task.runner.asyncio.sleep", new_callable=AsyncMock):
            result = await self.runner._run_instance_with_retry(
                self.instance, self.root / "results.jsonl", asyncio.Lock(), self.root / "trajectories.jsonl")
        self.assertTrue(result.success)
        self.assertEqual(self.runner._run_instance.await_count, 2)
        self.assertEqual(len((self.root / "results.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    async def test_exhausted_solve_is_archived_once_as_failure(self):
        self.runner._run_instance.side_effect = RuntimeError("synthetic solve failure")
        with patch("aweagent.core.task.runner.asyncio.sleep", new_callable=AsyncMock):
            result = await self.runner._run_instance_with_retry(
                self.instance, self.root / "results.jsonl", asyncio.Lock(), self.root / "trajectories.jsonl")
        self.assertFalse(result.success)
        self.assertEqual(self.runner._run_instance.await_count, 3)
        self.assertIsNone(result.agent_result)
        self.assertFalse((self.root / "trajectories.jsonl").exists())
        self.assertFalse(json.loads((self.root / "results.jsonl").read_text(encoding="utf-8"))["success"])

    async def test_run_directory_collision_does_not_mix_or_reexecute(self):
        old = self.root / "existing"
        old.mkdir()
        sentinel = old / "trajectories.jsonl"
        sentinel.write_bytes(b'{"historical":true}\n')
        with patch("aweagent.core.task.runner._build_run_dir", return_value=old):
            with self.assertRaises(FileExistsError):
                await self.runner.run_all()
        self.assertEqual(sentinel.read_bytes(), b'{"historical":true}\n')
        self.runner._run_instance.assert_not_awaited()

    async def test_config_archive_failure_stops_before_any_solve(self):
        with patch("aweagent.core.task.runner._save_run_config", side_effect=OSError("synthetic disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                await self.runner.run_all()
        self.runner._run_instance.assert_not_awaited()

    def test_partial_append_is_rolled_back_before_next_jsonl_record(self):
        target = self.root / "完整✨.jsonl"
        _append_jsonl(target, {"id": "旧行✨"})
        before = target.read_bytes()
        real_open = builtins.open

        class PartialWrite:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def write(self, line):
                self.stream.write(line[:8])
                self.stream.flush()
                raise OSError("synthetic partial write")

        with patch("aweagent.core.task.runner.open", side_effect=lambda *a, **kw: PartialWrite(real_open(*a, **kw))):
            with self.assertRaisesRegex(OSError, "partial write"):
                _append_jsonl(target, {"id": "损坏行✨"})
        self.assertEqual(target.read_bytes(), before)
        _append_jsonl(target, {"id": "新行✨"})
        self.assertEqual([json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()],
                         [{"id": "旧行✨"}, {"id": "新行✨"}])


if __name__ == "__main__":
    unittest.main()
