"""Real Git object tests: future answers cannot survive in nested repositories."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tarfile
import zipfile
import subprocess

import pytest

from aweagent.core.eval.workspace_hardening import (
    audit_source_copies, git, inspect_git, project_modules, repositories, sanitize_git, text,
)
from aweagent.core.eval.setup import PreAgentSetup
from aweagent.core.runtime.types import ExecutionResult


def repository(path):
    path.mkdir()
    git(path, 'init', '-q')
    git(path, 'config', 'user.name', 'Fixture')
    git(path, 'config', 'user.email', 'fixture@example.invalid')
    git(path, 'config', 'commit.gpgsign', 'false')
    (path / 'code.py').write_text('before\n')
    git(path, 'add', '.')
    git(path, 'commit', '-qm', 'before')
    return text(git(path, 'rev-parse', 'HEAD'))


def add_future(path):
    before = text(git(path, 'rev-parse', 'HEAD'))
    git(path, 'tag', 'historical')
    (path / 'code.py').write_text('future\n')
    git(path, 'add', '.')
    git(path, 'commit', '-qm', 'future')
    future = text(git(path, 'rev-parse', 'HEAD'))
    git(path, 'tag', '-a', 'future-answer', '-m', 'future')
    git(path, 'checkout', '-q', '--detach', before)
    return future


def test_nested_future_tags_refs_objects_are_removed_without_changing_checkout(tmp_path):
    source, root = tmp_path / 'source', tmp_path / 'task'
    repository(source)
    repository(root)
    git(root, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(source), 'nested')
    git(root, 'commit', '-qm', 'submodule')
    nested = root / 'nested'
    git(nested, 'config', 'user.name', 'Fixture')
    git(nested, 'config', 'user.email', 'fixture@example.invalid')
    git(nested, 'config', 'commit.gpgsign', 'false')
    nested_future = add_future(nested)
    root_future = add_future(root)
    before = {r: text(git(r, 'rev-parse', 'HEAD')) for r in repositories(root)}
    receipt = sanitize_git(root)
    assert len(receipt) == 2
    for repo, future in ((root, root_future), (nested, nested_future)):
        assert text(git(repo, 'rev-parse', 'HEAD')) == before[repo]
        assert text(git(repo, 'tag', '--list')) == 'historical'
        assert git(repo, 'cat-file', '-e', future, check=False).returncode != 0
        assert not inspect_git(repo)['unreachable_objects']
        assert (repo / 'code.py').read_text() == 'before\n'


def test_submodule_revision_drift_is_not_silently_blessed(tmp_path):
    source, root = tmp_path / 'source', tmp_path / 'task'
    repository(source); repository(root)
    git(root, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(source), 'nested')
    git(root, 'commit', '-qm', 'submodule')
    git(root / 'nested', 'checkout', '--orphan', 'wrong')
    git(root / 'nested', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'wrong')
    with pytest.raises(RuntimeError, match='pinned gitlink'):
        sanitize_git(root)


def test_late_annotated_tag_on_old_commit_is_removed(tmp_path):
    root = tmp_path / 'task'
    repository(root)
    subprocess.run(['git', '-C', str(root), 'tag', '-a', 'later-release', '-m', 'later'],
                   env={**os.environ, 'GIT_COMMITTER_DATE': '2099-01-01T00:00:00Z'}, check=True)
    assert 'refs/tags/later-release' in inspect_git(root)['forbidden_refs']
    sanitize_git(root)
    assert text(git(root, 'tag', '--list')) == ''


def test_graft_hidden_future_commit_cannot_pass_cleanup(tmp_path):
    root = tmp_path / 'task'
    repository(root)
    future = add_future(root)
    graft = root / '.git' / 'info' / 'grafts'
    graft.parent.mkdir(parents=True, exist_ok=True)
    graft.write_text('{} {}\n'.format(text(git(root, 'rev-parse', 'HEAD')), future))
    with pytest.raises(RuntimeError, match='grafts'):
        sanitize_git(root)
    assert git(root, 'cat-file', '-e', future, check=False).returncode == 0


def test_index_only_future_object_cannot_pass_cleanup(tmp_path):
    root = tmp_path / 'task'
    repository(root)
    blob = subprocess.run(
        ['git', '-C', str(root), 'hash-object', '-w', '--stdin'],
        input=b'future answer\n', capture_output=True, check=True,
    ).stdout.decode().strip()
    git(root, 'update-index', '--add', '--cacheinfo', '100644,{},answer.py'.format(blob))
    with pytest.raises(RuntimeError, match='index differs from HEAD'):
        sanitize_git(root)
    assert git(root, 'cat-file', '-e', blob, check=False).returncode == 0


def test_staged_future_submodule_gitlink_cannot_pass_cleanup(tmp_path):
    source, root = tmp_path / 'source', tmp_path / 'task'
    repository(source)
    future = add_future(source)
    repository(root)
    git(root, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(source), 'nested')
    git(root, 'commit', '-qm', 'submodule')
    nested = root / 'nested'
    git(nested, 'checkout', '-q', future)
    git(root, 'update-index', '--cacheinfo', '160000,{},nested'.format(future))
    with pytest.raises(RuntimeError, match='index differs from HEAD'):
        sanitize_git(root)
    assert git(nested, 'cat-file', '-e', future, check=False).returncode == 0


def test_external_package_and_wheel_are_found_but_workspace_is_preserved(tmp_path):
    root = tmp_path / 'task'; root.mkdir()
    package = root / 'samplepkg'; package.mkdir(); (package / '__init__.py').write_text('')
    outside = tmp_path / 'site-packages' / 'samplepkg'; outside.mkdir(parents=True)
    (outside / '__init__.py').write_text('other version')
    wheel = tmp_path / 'cached.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('samplepkg/__init__.py', 'cached version')
    assert project_modules(root) == ['samplepkg']
    result = audit_source_copies(root, ['samplepkg'], scan_root=tmp_path)
    assert set(result['external_copies']) == {str(outside), str(wheel)}
    assert not result['unreadable_paths']


def test_bytecode_and_tar_copies_are_found_without_extracting(tmp_path):
    root = tmp_path / 'task'; root.mkdir()
    (root / 'samplepkg').mkdir(); (root / 'samplepkg' / '__init__.py').write_text('')
    legacy = tmp_path / 'legacy' / 'samplepkg.pyc'; legacy.parent.mkdir()
    legacy.write_bytes(b'legacy bytecode')
    cached = tmp_path / 'cache' / '__pycache__' / 'samplepkg.cpython-312.pyc'
    cached.parent.mkdir(parents=True); cached.write_bytes(b'cached bytecode')
    payload = tmp_path / 'payload.txt'; payload.write_text('cached source')
    flat = tmp_path / 'flat.tar'
    with tarfile.open(flat, 'w') as archive:
        archive.add(payload, arcname='./samplepkg/__init__.py')
    compressed = tmp_path / 'samplepkg-1.0.tar.gz'
    with tarfile.open(compressed, 'w:gz') as archive:
        archive.add(payload, arcname='samplepkg-1.0/samplepkg/__init__.py')
    result = audit_source_copies(root, ['samplepkg'], scan_root=tmp_path)
    assert {str(legacy), str(cached), str(flat), str(compressed)} <= set(result['external_copies'])
    assert not result['unreadable_paths']


def test_cleanup_failure_aborts_before_any_model_call():
    async def execute(*args, **kwargs):
        return ExecutionResult(exit_code=1, stderr='cannot prune objects')
    with pytest.raises(RuntimeError, match='hardening failed'):
        asyncio.run(PreAgentSetup(SimpleNamespace(execute=execute), '/task').remove_future_commits())


def test_successful_preparation_retains_the_audit_receipt():
    receipt = {'git_passed': True, 'repositories': [{'head': 'fixture'}]}
    async def execute(*args, **kwargs):
        return ExecutionResult(stdout=json.dumps(receipt))
    setup = PreAgentSetup(SimpleNamespace(execute=execute), '/task')
    asyncio.run(setup.remove_future_commits())
    assert setup.hardening_receipt == receipt
