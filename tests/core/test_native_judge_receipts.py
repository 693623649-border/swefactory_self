import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


path = Path(__file__).resolve().parents[2] / "tools/judge_dsh_smoke.py"
spec = importlib.util.spec_from_file_location("native_judge_receipts", path)
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


class JudgeReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_before_requires_declared_commit_and_unchanged_tree(self):
        instance = SimpleNamespace(base_commit="base", workdir="/testbed")
        for before, after, expected in [
            ("base\ntree\n", "snapshot\ntree\n", True),
            ("wrong\ntree\n", "snapshot\ntree\n", False),
            ("base\ntree\n", "snapshot\nchanged-tree\n", False),
        ]:
            with self.subTest(before=before, after=after):
                evaluator = judge.JudgeEvaluator()
                session = SimpleNamespace(execute=AsyncMock(side_effect=[
                    SimpleNamespace(success=True, stdout=before),
                    SimpleNamespace(success=True, stdout=after),
                ]))
                with patch.object(judge.ScaleSWEEvaluator, "pre_patch_setup", new=AsyncMock()):
                    if expected:
                        await evaluator.pre_patch_setup(instance, session)
                    else:
                        with self.assertRaises(RuntimeError):
                            await evaluator.pre_patch_setup(instance, session)
                self.assertEqual(evaluator.evidence["fresh_before"], expected)

    async def test_independent_container_is_observed_not_assumed(self):
        for identity, network, mounts, image, expected in [
            ("judge", "none", [], "image", True),
            ("solver", "none", [], "image", False),
            ("judge", "bridge", [], "image", False),
            ("judge", "none", [{"Type": "bind", "Destination": "/testbed"}], "image", False),
            ("judge", "none", [], "different-image", False),
        ]:
            with self.subTest(identity=identity, network=network, mounts=mounts, image=image):
                runtime = judge.JudgeRuntime(SimpleNamespace(), {"container_id": "solver", "image_id": "image"})
                session = SimpleNamespace(_container=SimpleNamespace(id=identity, attrs={
                    "Image": image, "HostConfig": {"NetworkMode": network}, "Mounts": mounts,
                }), close=AsyncMock())
                with patch.object(judge.DockerRuntime, "create_session", new=AsyncMock(return_value=session)):
                    if expected:
                        self.assertIs(await runtime.create_session("image"), session)
                    else:
                        with self.assertRaises(RuntimeError):
                            await runtime.create_session("image")
                        session.close.assert_awaited_once()
                self.assertEqual(runtime.evidence["independent"], expected)

    def test_cleanup_probe_failure_does_not_claim_container_absent(self):
        runtime = judge.JudgeRuntime(SimpleNamespace(), {})
        runtime.evidence["container_id"] = "judge"
        for code, error, expected in [(1, "Error: No such container: judge", True),
                                      (1, "Cannot connect to Docker", False), (0, "", False)]:
            with patch.object(judge.subprocess, "run", return_value=SimpleNamespace(
                    returncode=code, stdout="", stderr=error)):
                self.assertEqual(runtime.cleanup_verified(), expected)


if __name__ == "__main__":
    unittest.main()
