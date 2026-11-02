import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from aweagent.core.eval import workspace_hardening
from aweagent.core.runtime.types import ExecutionResult
from aweagent.core.task.types import EvalResult


path = Path(__file__).resolve().parents[2] / "tools/judge_dsh_smoke.py"
spec = importlib.util.spec_from_file_location("native_judge_receipts", path)
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


class FakeSession:
    """Run Git in an isolated repo and constrain the copy audit to its sandbox."""

    def __init__(self, root, sandbox):
        self.root = root
        self.sandbox = sandbox
        self.bad_audit = False
        if sys.platform == "win32":
            self.bash = str(Path(shutil.which("git")).resolve().parent.parent / "bin/bash.exe")
        else:
            self.bash = shutil.which("bash")

    async def execute(self, command, cwd=None, timeout=None):
        if command.endswith(" --audit-copies"):
            if self.bad_audit:
                return ExecutionResult(stdout="PRIVATE_SENTINEL", stderr="", exit_code=1)
            actual_audit = workspace_hardening.audit_source_copies
            def confined_audit(root, modules):
                return actual_audit(root, modules, scan_root=self.sandbox, import_roots=[])
            stdout = io.StringIO()
            argv = ["hardener", *shlex.split(command)[3:]]
            with patch.object(sys, "argv", argv), \
                 patch.object(workspace_hardening, "audit_source_copies", side_effect=confined_audit), \
                 redirect_stdout(stdout):
                code = workspace_hardening.main()
            return ExecutionResult(stdout=stdout.getvalue(), stderr="", exit_code=code)
        if command.endswith(" --sanitize-git"):
            # Windows cannot pass the embedded helper as one large shell argument.
            args = [sys.executable, workspace_hardening.__file__, "--root", str(self.root), "--sanitize-git"]
        else:
            args = [self.bash, "-c", command]
        result = subprocess.run(args, cwd=self.root, capture_output=True, text=True, timeout=timeout)
        return ExecutionResult(stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode)


class JudgeReceiptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="judge-receipt-test-")
        self.addCleanup(self.temp.cleanup)
        self.sandbox = Path(self.temp.name)
        self.root = self.sandbox / "workspace"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Synthetic Test")
        self.git("config", "user.email", "synthetic@example.invalid")
        package = self.root / "samplepkg"
        package.mkdir()
        (package / "__init__.py").write_text("value = 1\n", encoding="utf-8")
        (self.root / ".gitignore").write_text("ignored-build/\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "synthetic base")
        self.base = self.git("rev-parse", "HEAD")
        self.instance = SimpleNamespace(id="fixture", base_commit=self.base, workdir=str(self.root), setup_commands=[])
        self.session = FakeSession(self.root, self.sandbox)
        self.evaluator = judge.JudgeEvaluator()

    def git(self, *args):
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    async def test_clean_base_is_verified_with_a_source_copy_audit(self):
        await self.evaluator.pre_patch_setup(self.instance, self.session)
        evidence = self.evaluator.evidence
        self.assertTrue(evidence["fresh_before"])
        self.assertEqual(evidence["diagnostics"]["stage"], "complete")
        self.assertTrue(evidence["diagnostics"]["base_matched"])
        self.assertTrue(evidence["diagnostics"]["source_copies"]["passed"])
        self.assertTrue(evidence["diagnostics"]["tree_matches_base"])

    async def test_new_commit_with_same_tree_is_not_mistaken_for_a_dirty_tree(self):
        self.instance.setup_commands = ["git commit --allow-empty -qm synthetic-snapshot"]
        await self.evaluator.pre_patch_setup(self.instance, self.session)
        evidence = self.evaluator.evidence
        self.assertNotEqual(evidence["prepared_commit"], self.base)
        self.assertTrue(evidence["diagnostics"]["base_ancestor"])
        self.assertTrue(evidence["diagnostics"]["tree_matches_base"])
        self.assertTrue(evidence["fresh_before"])

    async def test_packaging_metadata_is_reported_without_claiming_it_was_source(self):
        self.instance.setup_commands = [
            "mkdir -p samplepkg.egg-info && printf metadata > samplepkg.egg-info/PKG-INFO"
        ]
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertFalse(self.evaluator.evidence["fresh_before"])
        self.assertEqual(diagnostics["stage"], "prepared_tree")
        self.assertFalse(diagnostics["tree_matches_base"])
        self.assertEqual(diagnostics["committed_changes"]["added"], 1)
        self.assertEqual(diagnostics["prepared_status"]["tracked"], 0)
        self.assertTrue(diagnostics["source_copies"]["passed"])

    async def test_initial_dirty_source_cannot_be_reset_by_setup(self):
        source = self.root / "samplepkg/__init__.py"
        source.write_text("modified = True\n", encoding="utf-8")
        self.instance.setup_commands = ["git checkout -- samplepkg/__init__.py"]
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertEqual(diagnostics["stage"], "initial_status")
        self.assertEqual(diagnostics["initial_status"]["tracked"], 1)
        self.assertEqual(source.read_text(encoding="utf-8"), "modified = True\n")

    async def test_wrong_base_is_rejected_before_setup(self):
        self.instance.base_commit = "f" * 40
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertEqual(diagnostics["stage"], "initial_identity")
        self.assertFalse(diagnostics["base_matched"])

    async def test_untracked_and_ignored_source_are_rejected(self):
        for folder, key in (("untracked", "untracked"), ("ignored-build", "ignored")):
            with self.subTest(folder=folder):
                copy = self.root / folder / "copied.py"
                copy.parent.mkdir(exist_ok=True)
                copy.write_text("value = 2\n", encoding="utf-8")
                evaluator = judge.JudgeEvaluator()
                with self.assertRaises(RuntimeError):
                    await evaluator.pre_patch_setup(self.instance, self.session)
                self.assertEqual(evaluator.evidence["diagnostics"]["initial_status"][key], 1)
                copy.unlink()
                copy.parent.rmdir()

    async def test_tracked_source_change_from_setup_is_rejected(self):
        self.instance.setup_commands = ["printf 'modified = True\\n' > samplepkg/__init__.py"]
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        self.assertEqual(self.evaluator.evidence["diagnostics"]["committed_changes"]["modified"], 1)

    async def test_new_source_from_setup_is_rejected(self):
        self.instance.setup_commands = ["cp samplepkg/__init__.py source_copy.py"]
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertEqual(diagnostics["stage"], "prepared_tree")
        self.assertEqual(diagnostics["committed_changes"]["added"], 1)

    async def test_ignored_source_from_setup_is_rejected(self):
        self.instance.setup_commands = ["mkdir -p ignored-build && cp samplepkg/__init__.py ignored-build/copy.py"]
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertEqual(diagnostics["stage"], "prepared_tree")
        self.assertEqual(diagnostics["prepared_status"]["ignored"], 1)

    async def test_ignored_source_in_submodule_is_rejected(self):
        origin = self.sandbox / "child-origin"
        origin.mkdir()
        def child_git(*args):
            return subprocess.run(["git", *args], cwd=origin, capture_output=True, text=True, check=True)
        child_git("init", "-q")
        child_git("config", "user.name", "Synthetic Test")
        child_git("config", "user.email", "synthetic@example.invalid")
        (origin / ".gitignore").write_text("ignored-build/\n", encoding="utf-8")
        (origin / "module.py").write_text("value = 1\n", encoding="utf-8")
        child_git("add", ".")
        child_git("commit", "-qm", "synthetic child")
        self.git("-c", "protocol.file.allow=always", "submodule", "add", "-q", str(origin), "child")
        self.git("commit", "-qam", "add synthetic child")
        self.instance.base_commit = self.git("rev-parse", "HEAD")
        copied = self.root / "child/ignored-build/copy.py"
        copied.parent.mkdir()
        copied.write_text("value = 2\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertEqual(diagnostics["stage"], "initial_status")
        self.assertEqual(diagnostics["initial_status"]["ignored"], 1)

    async def test_external_source_copy_is_rejected_without_exposing_paths(self):
        copy = self.sandbox / "site-packages/samplepkg"
        copy.mkdir(parents=True)
        (copy / "__init__.py").write_text("value = 2\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertEqual(diagnostics["stage"], "source_copy_audit")
        self.assertFalse(diagnostics["source_copies"]["passed"])
        self.assertEqual(diagnostics["source_copies"]["external"], 1)
        self.assertNotIn(str(copy), json.dumps(diagnostics))

    async def test_broken_audit_fails_closed_without_exposing_output(self):
        self.session.bad_audit = True
        with self.assertRaises(RuntimeError):
            await self.evaluator.pre_patch_setup(self.instance, self.session)
        diagnostics = self.evaluator.evidence["diagnostics"]
        self.assertEqual(diagnostics["stage"], "source_copy_audit")
        self.assertFalse(self.evaluator.evidence["fresh_before"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(diagnostics))

    async def test_reused_evaluator_does_not_keep_success_after_early_failure(self):
        await self.evaluator.pre_patch_setup(self.instance, self.session)
        self.assertTrue(self.evaluator.evidence["fresh_before"])
        self.instance.image = "synthetic-image"
        def fail_session(image):
            raise RuntimeError("synthetic failure")
        runtime = SimpleNamespace(session=fail_session)
        result = await self.evaluator.evaluate(self.instance, "", runtime)
        self.assertFalse(result.accepted)
        self.assertFalse(self.evaluator.evidence["fresh_before"])
        self.assertEqual(self.evaluator.evidence["diagnostics"]["stage"], "not_started")

    async def test_safe_summary_excludes_private_result_details(self):
        source = self.sandbox / "synthetic-tasks.jsonl"
        source.write_text(json.dumps({"instance_id": "fixture", "parent_commit": self.base}) + "\n", encoding="utf-8")
        prediction = self.sandbox / "prediction.json"
        prediction.write_text(json.dumps({"instance_id": "fixture", "model_patch": ""}), encoding="utf-8")
        output = self.sandbox / "judge.json"
        evaluator = SimpleNamespace(
            evidence={"fresh_before": False, "diagnostics": {"stage": "setup"}},
            evaluate=AsyncMock(return_value=EvalResult(accepted=False, details={"error": "PRIVATE_SENTINEL"})))
        runtime = SimpleNamespace(evidence={"independent": True}, cleanup_verified=lambda: True)
        argv = ["judge", "--source-tasks", str(source), "--instance-id", "fixture",
                "--prediction", str(prediction), "--output", str(output)]
        stdout = io.StringIO()
        with patch.object(judge, "JudgeEvaluator", return_value=evaluator), \
             patch.object(judge, "JudgeRuntime", return_value=runtime), \
             patch.object(sys, "argv", argv), redirect_stdout(stdout):
            await judge.main()
        self.assertEqual(json.loads(stdout.getvalue())["diagnostics"], {"stage": "setup"})
        self.assertIn("PRIVATE_SENTINEL", output.read_text(encoding="utf-8"))
        self.assertNotIn("PRIVATE_SENTINEL", stdout.getvalue())

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
