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


def test_every_leak_channel_is_unreadable_after_cleanup_and_base_history_survives(tmp_path):
    """Planted future commit reachable through each mechanism must lose its object."""
    source, root, donor = tmp_path / 'source', tmp_path / 'task', tmp_path / 'donor'
    repository(source); base = repository(root); repository(donor)
    (root / 'code.py').write_text('before\nsecond\n')
    git(root, 'commit', '-qam', 'second'); base = text(git(root, 'rev-parse', 'HEAD'))
    git(root, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(source), 'nested')
    git(root, 'commit', '-qm', 'submodule'); head = text(git(root, 'rev-parse', 'HEAD'))
    nested = root / 'nested'
    for repo in (nested,):
        git(repo, 'config', 'user.name', 'Fixture'); git(repo, 'config', 'user.email', 'f@example.invalid')
        git(repo, 'config', 'commit.gpgsign', 'false')
    nested_future = add_future(nested)
    planted = {}
    def future(label):
        (root / 'code.py').write_text('answer ' + label + '\n')
        git(root, 'commit', '-qam', 'future ' + label)
        sha = text(git(root, 'rev-parse', 'HEAD'))
        git(root, 'reset', '-q', '--hard', head)
        planted[label] = sha
        return sha
    git(root, 'branch', 'upstream-fix', future('branch'))
    git(root, 'tag', '-a', 'v-next', '-m', 'release', future('tag'))
    git(root, 'notes', 'add', '-m', 'see ' + future('note-target'), head)
    git(root, 'replace', head, future('replace'))
    git(root, 'replace', '-d', head)  # replace ref recreated below on a non-HEAD object
    git(root, 'update-ref', 'refs/replace/' + base, planted['replace'])
    future('reflog')  # only reachable through HEAD's reflog
    (root / 'code.py').write_text('stashed answer\n'); git(root, 'stash', '-q')
    planted['stash'] = text(git(root, 'rev-parse', 'refs/stash'))
    common = root / '.git'
    (common / 'FETCH_HEAD').write_text(future('fetch-head') + '\t\tbranch main\n')
    (common / 'ORIG_HEAD').write_text(future('orig-head') + '\n')
    keep_sha = future('keep-pack')
    pack = subprocess.run(['git', '-C', str(root), 'pack-objects', str(common / 'objects/pack/pack')],
                          input=(keep_sha + '\n').encode(), capture_output=True, check=True).stdout.decode().strip()
    (common / 'objects/pack' / ('pack-' + pack + '.keep')).write_text('')
    wt = tmp_path / 'linked'
    git(root, 'worktree', 'add', '-q', '--detach', str(wt), future('worktree-head'))
    (donor / 'code.py').write_text('donor-only upstream answer\n')
    git(donor, 'commit', '-qam', 'donor answer')
    donor_sha = text(git(donor, 'rev-parse', 'HEAD'))
    (common / 'objects/info/alternates').write_text(str((donor / '.git/objects').resolve()) + '\n')
    old_module = common / 'modules' / 'retired'
    subprocess.run(['git', 'clone', '-q', '--bare', str(donor), str(old_module)], check=True)
    git(root, 'reflog', 'expire', '--expire=never', '--all', check=False)

    before = inspect_git(root, cutoff_commit=base)
    assert before['pseudo_refs'] and before['retained_packs'] and before['object_alternates']
    receipt = sanitize_git(root, base=base)
    assert receipt[-1]['removed_modules'] and receipt[-1]['removed_pseudo_refs']
    for label, sha in planted.items():
        assert git(root, 'cat-file', '-e', sha, check=False).returncode != 0, label
    assert git(root, 'cat-file', '-e', donor_sha, check=False).returncode != 0
    assert git(nested, 'cat-file', '-e', nested_future, check=False).returncode != 0
    assert not old_module.exists() and not (common / 'worktrees').exists()
    assert text(git(root, 'rev-parse', 'HEAD')) == head
    assert text(git(root, 'for-each-ref', '--format=%(refname)')) == 'refs/heads/' + text(
        git(root, 'symbolic-ref', '--short', 'HEAD'))
    assert 'second' in text(git(root, 'log', '--format=%s'))
    assert base in text(git(root, 'blame', '--porcelain', 'code.py'))


def test_base_must_be_head_or_ancestor(tmp_path):
    root = tmp_path / 'task'
    repository(root)
    future = add_future(root)
    with pytest.raises(RuntimeError, match='not an ancestor'):
        sanitize_git(root, base=future)
    with pytest.raises(RuntimeError, match='not a commit'):
        sanitize_git(root, base='0' * 40)


def test_linked_worktree_of_shared_repository_is_refused_untouched(tmp_path):
    root = tmp_path / 'shared'
    repository(root)
    future = add_future(root)
    linked = tmp_path / 'linked'
    git(root, 'worktree', 'add', '-q', '--detach', str(linked), 'HEAD')
    with pytest.raises(RuntimeError, match='linked worktree'):
        sanitize_git(linked)
    assert git(root, 'cat-file', '-e', future).returncode == 0
    assert (root / '.git' / 'worktrees').exists()


def test_dangling_remote_head_is_removed_and_legacy_timezone_history_is_kept_as_warning(tmp_path):
    # Shape of the SWE-bench requests image: origin/HEAD points at a missing branch and
    # an old upstream commit carries a malformed timezone that fsck reports as an error.
    root = tmp_path / 'task'
    repository(root)
    tree = text(git(root, 'rev-parse', 'HEAD^{tree}'))
    legacy = 'tree {}\nauthor Old <old@example.invalid> 1100000000 +05\ncommitter Old <old@example.invalid> 1100000000 +05\n\nlegacy\n'
    old = subprocess.run(['git', 'hash-object', '-t', 'commit', '-w', '--literally', '--stdin'], cwd=root,
                         input=legacy.format(tree).encode(), capture_output=True, check=True).stdout.decode().strip()
    base = text(git(root, 'commit-tree', tree, '-p', old, '-m', 'base'))
    git(root, 'checkout', '-q', '--detach', base)
    future = add_future(root)
    (root / '.git' / 'refs' / 'remotes' / 'origin').mkdir(parents=True)
    (root / '.git' / 'refs' / 'remotes' / 'origin' / 'HEAD').write_text('ref: refs/remotes/origin/main\n')
    before = inspect_git(root, cutoff_commit=base)
    assert 'refs/remotes/origin/HEAD' in before['forbidden_refs']
    sanitize_git(root, base=base)
    after = inspect_git(root)
    assert after['fsck_exit_code'] == 0 and not after['forbidden_refs']
    assert any('badTimezone' in line for line in after['fsck_warnings'])
    assert not (root / '.git' / 'refs' / 'remotes' / 'origin' / 'HEAD').exists()
    assert git(root, 'cat-file', '-e', future, check=False).returncode != 0
    assert git(root, 'cat-file', '-e', old).returncode == 0


def test_installed_top_level_names_the_modules_so_dev_dirs_do_not_flag_stdlib(tmp_path):
    # Shape of the SWE-bench sphinx image: a dev-only top-level utils/ package next to the project.
    root = tmp_path / 'task'; root.mkdir()
    for name in ('sphinx', 'utils'):
        (root / name).mkdir(); (root / name / '__init__.py').write_text('')
    (root / 'Sphinx.egg-info').mkdir(); (root / 'Sphinx.egg-info' / 'top_level.txt').write_text('sphinx\n')
    stdlib = tmp_path / 'lib' / 'email'; stdlib.mkdir(parents=True); (stdlib / 'utils.py').write_text('')
    vendored = tmp_path / 'lib' / 'pip' / '_vendor' / 'sphinx'; vendored.mkdir(parents=True)
    (vendored / '__init__.py').write_text('newer release')
    modules = project_modules(root)
    assert modules == ['sphinx']
    assert audit_source_copies(root, modules, scan_root=tmp_path)['external_copies'] == [str(vendored)]


def package(base, *files):
    for name in files:
        path = base / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('x')
    return base


def test_tests_top_level_ignores_other_packages_tests_but_blocks_real_copies(tmp_path):
    # fugue-project/triad declares tests as an installed top-level next to triad.
    root = tmp_path / 'task'
    package(root / 'tests', '__init__.py', 'utils/__init__.py', 'utils/test_convert.py', 'collections/test_schema.py')
    package(root / 'triad', '__init__.py', 'utils/__init__.py', 'utils/convert.py')
    package(root / 'triad.egg-info', 'top_level.txt').joinpath('top_level.txt').write_text('tests\ntriad\n')
    site = tmp_path / 'site-packages'
    other = package(site / 'numpy', '__init__.py', 'tests/__init__.py', 'tests/test_numeric.py',
                    'lib/__init__.py', 'lib/tests/__init__.py', 'lib/tests/test_io.py')
    copied_tests = package(site / 'tests', '__init__.py', 'utils/test_convert.py')            # installed top-level
    nested_copy = package(site / 'fugue' / 'tests', 'utils/test_convert.py')                  # same members, nested
    (site / 'fugue' / '__init__.py').write_text('')
    old_triad = package(site / 'fugue' / 'triad', 'utils/convert.py')
    wheel = tmp_path / 'setuptools.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('setuptools/__init__.py', ''); archive.writestr('setuptools/tests/test_egg.py', '')
    future_wheel = tmp_path / 'triad-9.whl'
    with zipfile.ZipFile(future_wheel, 'w') as archive:
        archive.writestr('triad/utils/convert.py', 'future')
    result = audit_source_copies(root, project_modules(root), scan_root=tmp_path)
    assert set(result['external_copies']) == {str(copied_tests), str(nested_copy), str(old_triad), str(future_wheel)}
    assert {str(other / 'tests'), str(other / 'lib' / 'tests'), str(wheel)} <= set(result['name_collisions'])


def test_svg_submodules_of_pygments_are_collisions_but_svg_copies_are_blocked(tmp_path):
    # regebro/svg.path: src/svg/path is the project; pygments has formatters/svg.py.
    root = tmp_path / 'task'
    package(root / 'src' / 'svg', '__init__.py', 'path/__init__.py', 'path/parser.py', 'path/path.py')
    site = tmp_path / 'site-packages'
    package(site / 'pygments', '__init__.py', 'formatters/__init__.py', 'formatters/svg.py')
    pyc = site / 'pygments' / 'formatters' / '__pycache__' / 'svg.cpython-311.pyc'
    pyc.parent.mkdir(); pyc.write_bytes(b'')
    pip_wheel = tmp_path / 'pip.whl'
    with zipfile.ZipFile(pip_wheel, 'w') as archive:
        for name in ('pip/__init__.py', 'pip/_vendor/__init__.py', 'pip/_vendor/pygments/__init__.py',
                     'pip/_vendor/pygments/formatters/__init__.py', 'pip/_vendor/pygments/formatters/svg.py'):
            archive.writestr(name, '')
    installed = package(site / 'svg', 'path/parser.py')
    vendored = package(site / 'pip' / '_vendor' / 'svg', '__init__.py')
    (site / 'pip' / '__init__.py').write_text(''); (site / 'pip' / '_vendor' / '__init__.py').write_text('')
    nested = package(site / 'pygments' / 'svg', 'path/path.py')
    loose_bytecode = site / 'svg.pyc'; loose_bytecode.write_bytes(b'')
    sdist = tmp_path / 'svg.path-9.tar.gz'
    payload = tmp_path / 'payload.txt'; payload.write_text('future')
    with tarfile.open(sdist, 'w:gz') as archive:
        archive.add(payload, arcname='svg.path-9/src/svg/path/path.py')
    result = audit_source_copies(root, project_modules(root), scan_root=tmp_path)
    assert project_modules(root) == ['svg']
    assert set(result['external_copies']) == {str(installed), str(vendored), str(nested), str(loose_bytecode), str(sdist)}
    assert {str(site / 'pygments' / 'formatters' / 'svg.py'), str(pyc), str(pip_wheel)} <= set(result['name_collisions'])


def test_vendored_requests_inside_pip_stays_an_external_copy(tmp_path):
    root = tmp_path / 'task'
    package(root / 'requests', '__init__.py', 'api.py', 'sessions.py')
    site = tmp_path / 'site-packages'
    package(site / 'pip', '__init__.py', '_vendor/__init__.py')
    vendored = package(site / 'pip' / '_vendor' / 'requests', '__init__.py', 'api.py')
    old_layout = package(site / 'botocore', '__init__.py', 'vendored/__init__.py')
    old_copy = package(site / 'botocore' / 'vendored' / 'requests', '__init__.py')            # only generic names
    wheel = tmp_path / 'pip-24.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name in ('pip/__init__.py', 'pip/_vendor/__init__.py', 'pip/_vendor/requests/__init__.py'):
            archive.writestr(name, '')
    result = audit_source_copies(root, ['requests'], scan_root=tmp_path)
    assert set(result['external_copies']) == {str(vendored), str(old_copy), str(wheel)}
    assert not result['name_collisions']


def test_single_file_and_unknown_projects_cannot_be_compared_so_stay_blocked(tmp_path):
    root = tmp_path / 'task'; root.mkdir(); (root / 'six.py').write_text('')
    site = tmp_path / 'site-packages'
    inline = package(site / 'django', '__init__.py', 'utils/__init__.py', 'utils/six.py') / 'utils' / 'six.py'
    assert audit_source_copies(root, ['six'], scan_root=tmp_path)['external_copies'] == [str(inline)]
    unknown = tmp_path / 'other'; unknown.mkdir()
    nested = package(site / 'django', 'ghost/__init__.py') / 'ghost'
    assert str(nested) in audit_source_copies(unknown, ['ghost'], scan_root=tmp_path)['external_copies']


def test_same_named_package_with_changed_layout_under_plain_directory_is_a_copy(tmp_path):
    # A later/older release can share no file names with the checkout; it is still the project.
    root = tmp_path / 'task'
    package(root / 'src' / 'triad', '__init__.py', 'old_api.py')
    download = package(tmp_path / 'download' / 'triad', '__init__.py', 'new_api.py')
    payload = tmp_path / 'payload.txt'; payload.write_text('future')
    sdist = tmp_path / 'triad-9.0.tar.gz'
    with tarfile.open(sdist, 'w:gz') as archive:
        for name in ('triad-9.0/src/triad/__init__.py', 'triad-9.0/src/triad/new_api.py'):
            archive.add(payload, arcname=name)
    bundle = tmp_path / 'triad-9.0.zip'
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr('triad-9.0/src/triad/__init__.py', ''); archive.writestr('triad-9.0/src/triad/new_api.py', '')
    unrelated = package(tmp_path / 'share' / 'tool' / 'triad', 'README')   # not Python, no shared members
    result = audit_source_copies(root, ['triad'], scan_root=tmp_path, import_roots=[])
    assert set(result['external_copies']) == {str(download), str(sdist), str(bundle)}
    assert result['name_collisions'] == [str(unrelated)]
