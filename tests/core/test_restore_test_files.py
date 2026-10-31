"""restore_test_files against a real Git repository (synthetic fixtures only)."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aweagent.core.eval.base import PatchTestEvaluator
from aweagent.core.eval.utils import is_test_path, restore_test_files
from aweagent.core.runtime.types import ExecutionResult
from aweagent.core.task.types import EvalResult, Instance

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or shutil.which("git") is None, reason="needs bash and git")

BASE = {
    "tests/test_sample.py": "def test_ok():\n    assert True\n",
    "tests/cases.json": '{"accept": ["01:23:45:67:89:ab"]}\n',
    "tests/data/带 空格.json": '{"unicode": true}\n',
    "tests/test_removed.py": "def test_removed():\n    pass\n",
    "pkg/mod_test.py": "def test_mod():\n    pass\n",
    "pkg/core.py": "VALUE = 1\n",
    "setup.cfg": "[metadata]\nname = demo\n",
}


class LocalSession:
    """Run commands with the same bash semantics the container session uses."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.commands: list[str] = []

    async def execute(self, command, cwd=None, timeout=None, env=None):
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            return ExecutionResult(stderr="injected failure", exit_code=1)
        proc = subprocess.run([BASH, "-c", command], cwd=cwd, capture_output=True,
                              env={**os.environ, **(env or {})})
        return ExecutionResult(stdout=proc.stdout.decode("utf-8", "replace"),
                               stderr=proc.stderr.decode("utf-8", "replace"), exit_code=proc.returncode)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout.decode()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "fixture")
    (root / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    for name, text in BASE.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode())
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def tamper(root: Path) -> None:
    (root / "tests/test_sample.py").write_text("def test_ok():\n    assert 1\n", encoding="utf-8")
    (root / "tests/cases.json").write_text('{"accept": ["0123456789abcd"]}\n', encoding="utf-8")
    (root / "tests/data/带 空格.json").write_text('{"unicode": false}\n', encoding="utf-8")
    (root / "tests/test_removed.py").unlink()
    (root / "pkg/mod_test.py").write_text("", encoding="utf-8")
    (root / "tests/test_added.py").write_text("def test_added():\n    pass\n", encoding="utf-8")
    (root / "conftest.py").write_text("import pytest\n", encoding="utf-8")
    (root / "tests/新 夹具.json").write_text("{}\n", encoding="utf-8")
    (root / "tests/cache.pyc").write_bytes(b"ignored")
    (root / "pkg/core.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "pkg/new_module.py").write_text("NEW = True\n", encoding="utf-8")
    (root / "setup.cfg").write_text("[metadata]\nname = demo\nversion = 2\n", encoding="utf-8")


def test_is_test_path_covers_fixtures_and_named_tests():
    assert is_test_path("tests/cases.json")
    assert is_test_path("src/pkg/Tests/data/x.bin")
    assert is_test_path("pkg/test_x.py") and is_test_path("pkg/x_test.py") and is_test_path("conftest.py")
    assert not is_test_path("tests")  # a file named like a directory is not inside one
    assert not is_test_path("pkg/testing.py") and not is_test_path("pkg/contest.py")
    assert not is_test_path("setup.cfg")


def test_restores_fixtures_deleted_tests_and_keeps_source_patch(repo: Path):
    tamper(repo)
    receipt = asyncio.run(restore_test_files(LocalSession(), str(repo)))

    for name, text in BASE.items():
        if name.startswith("tests/") or name == "pkg/mod_test.py":
            assert (repo / name).read_bytes() == text.encode(), name
    # Added test-surface files are removed; ignored files are not the patch's.
    assert not (repo / "tests/test_added.py").exists()
    assert not (repo / "conftest.py").exists()
    assert not (repo / "tests/新 夹具.json").exists()
    assert (repo / "tests/cache.pyc").exists()
    # Source, new modules and build configuration keep the patch's changes.
    assert (repo / "pkg/core.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (repo / "pkg/new_module.py").exists()
    assert "version = 2" in (repo / "setup.cfg").read_text(encoding="utf-8")
    assert receipt["restored_paths"] == 5
    assert sorted(receipt["removed_added_paths"]) == ["conftest.py", "tests/test_added.py", "tests/新 夹具.json"]
    assert git(repo, "status", "--porcelain", "--", "tests", "pkg/mod_test.py") == ""


def test_missing_conventional_directories_are_normal(repo: Path):
    # Base has tests/ only; test/ Test/ Tests/ are absent, which must not fail.
    receipt = asyncio.run(restore_test_files(LocalSession(), str(repo)))
    assert receipt == {"restored_paths": 5, "removed_added_paths": []}


def test_real_checkout_failure_raises(repo: Path):
    tamper(repo)
    with pytest.raises(RuntimeError, match="checkout HEAD test paths"):
        asyncio.run(restore_test_files(LocalSession(fail_on="git checkout HEAD --"), str(repo)))


def test_unverified_restore_raises(repo: Path):
    tamper(repo)
    with pytest.raises(RuntimeError, match="verify restored"):
        asyncio.run(restore_test_files(LocalSession(fail_on="git diff --quiet HEAD"), str(repo)))


class _Runtime:
    def __init__(self, session):
        self._session = session

    def session(self, image):
        session = self._session

        class _Ctx:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


class _PassingEvaluator(PatchTestEvaluator):
    async def run_tests(self, instance, session):
        return EvalResult(accepted=True, score=1.0)


def test_evaluator_rejects_when_restore_fails(repo: Path):
    tamper(repo)
    instance = Instance(id="fixture", dataset_id="synthetic", workdir=str(repo))
    result = asyncio.run(_PassingEvaluator().evaluate(instance, "", _Runtime(LocalSession(fail_on="git checkout HEAD --"))))
    assert result.accepted is False and "test restore failed" in result.details["error"]
    ok = asyncio.run(_PassingEvaluator().evaluate(instance, "", _Runtime(LocalSession())))
    assert ok.accepted is True and ok.details["test_restore"]["restored_paths"] == 5


def test_git_apply_ignored_hook_and_index_additions_are_removed(repo):
    (repo/'.gitignore').write_text('*.pyc\ntests/conftest.py\ntests/base.bin\n', encoding='utf-8')
    git(repo, 'add', '.gitignore')
    git(repo, 'commit', '-qm', 'ignore hooks')
    (repo/'tests/base.bin').write_bytes(b'trusted image asset')
    (repo/'tests/cache.pyc').write_bytes(b'trusted cache')
    patch = ('diff --git a/tests/conftest.py b/tests/conftest.py\nnew file mode 100644\n'
             '--- /dev/null\n+++ b/tests/conftest.py\n@@ -0,0 +1 @@\n+raise RuntimeError("injected")\n')
    patch_path = repo/'input.diff'
    patch_path.write_text(patch, encoding='utf-8')
    git(repo, 'apply', str(patch_path))
    (repo/'tests/staged.txt').write_text('injected', encoding='utf-8')
    git(repo, 'add', 'tests/staged.txt')
    (repo/'pkg/core.py').write_text('VALUE = 2\n', encoding='utf-8')
    receipt = asyncio.run(restore_test_files(LocalSession(), str(repo), patch))
    assert receipt['removed_added_paths'] == ['tests/conftest.py', 'tests/staged.txt']
    assert not (repo/'tests/conftest.py').exists()
    assert not (repo/'tests/staged.txt').exists()
    assert (repo/'tests/base.bin').read_bytes() == b'trusted image asset'
    assert (repo/'tests/cache.pyc').read_bytes() == b'trusted cache'
    assert (repo/'pkg/core.py').read_text(encoding="utf-8") == 'VALUE = 2\n'
    assert not git(repo, 'diff', '--cached', '--name-only')
