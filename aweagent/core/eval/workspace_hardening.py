"""Standalone pre-solver Git cleanup and Python source-copy audit.

Run inside a disposable task container, before the model receives any tools.
No task solutions or hidden tests are read. File-copy checks inspect names only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile

# Files outside refs/ that can name or retain objects beyond the frozen base.
PSEUDO_REFS = ('FETCH_HEAD', 'ORIG_HEAD', 'MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD',
               'REBASE_HEAD', 'AUTO_MERGE', 'BISECT_HEAD', 'BISECT_EXPECTED_REV', 'MERGE_AUTOSTASH')
# Legacy object-format findings in permitted history (e.g. an old commit with a
# malformed timezone). They are reported as warnings, never as a pass for
# missing, corrupt or unreachable objects, which stay fatal.
BROKEN_REF = re.compile(r'error: (refs/\S+): invalid sha1 pointer ')
FSCK_LEGACY_FORMAT = ('badTimezone', 'badDate', 'badDateOverflow', 'missingSpaceBeforeDate',
                      'zeroPaddedDate', 'zeroPaddedFilemode')
STATE_PATHS = ('rebase-merge', 'rebase-apply', 'sequencer', 'BISECT_LOG', 'BISECT_START',
               'BISECT_TERMS', 'BISECT_NAMES', 'lost-found', 'worktrees')


def git(root, *args, check=True):
    result = subprocess.run(
        ['git', '-C', str(root), *args], capture_output=True,
        env={**os.environ, 'GIT_NO_REPLACE_OBJECTS': '1'},
    )
    if check and result.returncode:
        raise RuntimeError('git {} failed: {}'.format(
            ' '.join(args), result.stderr.decode('utf-8', errors='replace')[-1200:]))
    return result


def text(result):
    return result.stdout.decode('utf-8', errors='replace').strip()


def repositories(root):
    """Include initialized submodules and verify their pinned gitlink identity."""
    root = Path(root).resolve()
    result = [root]
    for entry in git(root, 'ls-files', '--stage', '-z').stdout.split(b'\0'):
        if not entry:
            continue
        meta, path = entry.split(b'\t', 1)
        mode, commit, stage = meta.split()
        if mode != b'160000':
            continue
        child = (root / os.fsdecode(path)).resolve()
        if root not in child.parents:
            raise RuntimeError('submodule escapes workspace: {}'.format(child))
        if not (child / '.git').exists():
            if child.exists() and any(child.iterdir()):
                raise RuntimeError('submodule has files but no Git identity: {}'.format(child))
            continue  # an absent submodule exposes no source or Git objects
        actual = text(git(child, 'rev-parse', 'HEAD'))
        if actual != commit.decode():
            raise RuntimeError('submodule HEAD differs from pinned gitlink: {}'.format(child))
        result.extend(repositories(child))
    return result


def common_dir(root):
    common = Path(text(git(root, 'rev-parse', '--git-common-dir')))
    return common if common.is_absolute() else Path(root) / common


def leftover_modules(root, repos):
    """Module Git dirs under .git/modules that no initialized submodule uses."""
    modules = common_dir(root) / 'modules'
    if not modules.is_dir():
        return []
    used = {common_dir(repo).resolve() for repo in repos}
    found = []
    for directory, dirs, _ in os.walk(modules):
        path = Path(directory)
        if (path / 'HEAD').is_file() and (path / 'objects').is_dir():
            if path.resolve() not in used:
                found.append(path)
            dirs[:] = [d for d in dirs if d == 'modules']
    return found


def inspect_git(root, cutoff_commit=None):
    head = text(git(root, 'rev-parse', 'HEAD'))
    index = git(root, 'diff-index', '--cached', '--quiet', 'HEAD', '--', check=False)
    if index.returncode:
        raise RuntimeError('Git index differs from HEAD: {}'.format(root))
    if cutoff_commit is None:
        saved = git(root, 'rev-parse', '--verify', 'refs/aweagent/pre-solver-base^{commit}', check=False)
        cutoff_commit = text(saved) if saved.returncode == 0 else head
    resolved = git(root, 'rev-parse', '--verify', '--quiet', cutoff_commit + '^{commit}', check=False)
    if resolved.returncode:
        raise RuntimeError('pre-solver base is not a commit in this repository: {}'.format(cutoff_commit))
    cutoff_commit = text(resolved)
    if git(root, 'merge-base', '--is-ancestor', cutoff_commit, head, check=False).returncode:
        raise RuntimeError('pre-solver base is not an ancestor of HEAD')
    cutoff_time = int(text(git(root, 'show', '-s', '--format=%ct', cutoff_commit)))
    branch = text(git(root, 'symbolic-ref', '-q', 'HEAD', check=False))
    listed = git(root, 'for-each-ref', '--format=%(refname)%09%(taggerdate:unix)')
    legacy = [arg for name in FSCK_LEGACY_FORMAT for arg in ('-c', 'fsck.{}=warn'.format(name))]
    fsck = git(root, *legacy, 'fsck', '--unreachable', '--no-reflogs', check=False)
    fsck_stderr = fsck.stderr.decode('utf-8', errors='replace').splitlines()
    # for-each-ref silently skips refs that resolve to nothing (e.g. a dangling origin/HEAD symref); fsck names them.
    broken = sorted({m.group(1) for m in map(BROKEN_REF.match, fsck_stderr) if m})
    refs = list(broken)
    forbidden = list(broken)
    for entry in text(listed).splitlines():
        ref, _, tag_time = entry.partition('\t')
        target = git(root, 'rev-parse', ref + '^{commit}', check=False)
        ancestor = target.returncode == 0 and git(
            root, 'merge-base', '--is-ancestor', text(target), head, check=False).returncode == 0
        allowed = (ref == branch and text(target) == head) or (
            ref.startswith('refs/tags/') and ancestor and (not tag_time or int(tag_time) <= cutoff_time))
        refs.append(ref)
        if not allowed:
            forbidden.append(ref)
    unreachable = [line for line in text(fsck).splitlines() if line.startswith(('unreachable ', 'dangling '))]
    fsck_warnings = [line for line in fsck_stderr if line.startswith('warning in ')]
    common = common_dir(root)
    grafts = common / 'info' / 'grafts'
    if grafts.is_file():
        lines = grafts.read_text(encoding='utf-8', errors='replace').splitlines()
        if any(line.strip() and not line.lstrip().startswith('#') for line in lines):
            raise RuntimeError('Git grafts can hide future objects: {}'.format(root))
    alternates = [common / 'objects' / 'info' / name for name in ('alternates', 'http-alternates')]
    pack = common / 'objects' / 'pack'
    retained = sorted(p.name for p in pack.glob('*') if p.suffix in ('.keep', '.promisor')) if pack.is_dir() else []
    private = Path(text(git(root, 'rev-parse', '--absolute-git-dir')))
    if private.resolve() != common.resolve():
        # Cleaning a shared common dir would touch other checkouts; only exclusive copies qualify.
        raise RuntimeError('task workspace is a linked worktree of a shared repository: {}'.format(root))
    return dict(head=head, cutoff_commit=cutoff_commit, ref_count=len(refs), forbidden_refs=forbidden,
                unreachable_objects=len(unreachable), fsck_exit_code=fsck.returncode,
                object_alternates=any(path.exists() for path in alternates),
                pseudo_refs=[name for name in PSEUDO_REFS if (common / name).exists() or (private / name).exists()],
                state_paths=[name for name in STATE_PATHS if (common / name).exists()],
                retained_packs=retained, fsck_warnings=fsck_warnings[:20])


def converged(receipt):
    return not (receipt['forbidden_refs'] or receipt['unreachable_objects'] or receipt['object_alternates']
                or receipt['fsck_exit_code'] or receipt['pseudo_refs'] or receipt['state_paths']
                or receipt['retained_packs'] or receipt.get('leftover_modules'))


def _force(func, path, _):
    os.chmod(path, stat.S_IWRITE)  # Git writes read-only object files
    func(path)


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(str(path), onerror=_force)
    elif path.exists() or path.is_symlink():
        path.unlink()


def sanitize_git(root, base=None):
    """Delete everything not reachable from the frozen checkout, recursively.

    ``base`` is the public task commit; it must be HEAD or one of its ancestors.
    Only the disposable task copy at ``root`` is modified.
    """
    repos = repositories(root)
    receipts = []
    snapshots = [(repo, inspect_git(repo, cutoff_commit=base if repo == repos[0] else None))
                 for repo in reversed(repos)]
    for repo, before in snapshots:
        for ref in before['forbidden_refs']:
            if git(repo, 'update-ref', '--no-deref', '-d', ref, check=False).returncode:
                remove(common_dir(repo) / ref)  # a broken loose symref update-ref cannot resolve
        common = common_dir(repo)
        private = Path(text(git(repo, 'rev-parse', '--absolute-git-dir')))
        for name in PSEUDO_REFS:
            for directory in {common, private}:
                remove(directory / name)
        for name in STATE_PATHS:
            remove(common / name)
        stale = leftover_modules(repo, repos)
        for module in stale:
            remove(module)
        alternates = common / 'objects' / 'info' / 'alternates'
        if alternates.exists() or (common / 'objects' / 'info' / 'http-alternates').exists():
            # Materialize reachable objects before severing the shared object store.
            git(repo, 'repack', '-a', '-d')
            remove(alternates)
            remove(common / 'objects' / 'info' / 'http-alternates')
        for marker in before['retained_packs']:
            # A .keep/.promisor pack survives gc with whatever unreachable objects it holds.
            remove(common / 'objects' / 'pack' / marker)
        git(repo, 'reflog', 'expire', '--expire=now', '--expire-unreachable=now', '--all')
        git(repo, 'gc', '--prune=now')
        after = inspect_git(repo, cutoff_commit=before['cutoff_commit'])
        after['leftover_modules'] = [str(path) for path in leftover_modules(repo, repos)]
        if after['head'] != before['head'] or not converged(after):
            raise RuntimeError('Git isolation did not converge: {}'.format(repo))
        receipts.append(dict(path=str(repo), removed_ref_count=len(before['forbidden_refs']),
                             removed_pseudo_refs=before['pseudo_refs'], removed_state_paths=before['state_paths'],
                             removed_modules=[str(path) for path in stale], **after))
    return receipts


def project_modules(root):
    """Python import roots the project installs; callers can supply explicit modules.

    An installed checkout's ``*.egg-info/top_level.txt`` is authoritative. Without it,
    fall back to conventional top-level packages, which can include dev-only dirs.
    """
    declared = {name.strip() for base in (Path(root), Path(root) / 'src', Path(root) / 'lib')
                for listing in base.glob('*.egg-info/top_level.txt')
                for name in listing.read_text(encoding='utf-8', errors='replace').splitlines()
                if name.strip().isidentifier()}
    if declared:
        return sorted(declared)
    names = set()
    for base in (Path(root), Path(root) / 'src'):
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            name = entry.stem if entry.is_file() else entry.name
            if name.startswith(('test', '.')) or name in {'setup', 'conftest', '__init__', 'docs', 'examples'}:
                continue
            if name.isidentifier() and ((entry.is_dir() and (entry / '__init__.py').is_file())
                                        or (entry.is_file() and entry.suffix == '.py')):
                names.add(name)
    return sorted(names)


def audit_source_copies(root, modules, scan_root='/'):
    """Look for the project's import namespaces and Python source/archive copies outside root.

    This is a named Python-source check, not a claim to recognize renamed code.
    Pseudo-filesystems are excluded. Archive entries are inspected without extraction.
    """
    root = Path(root).resolve()
    scan_root = Path(scan_root).resolve()
    modules = set(modules)
    hits, scanned, errors = [], 0, []
    excluded = {Path(p) for p in ('/proc', '/sys', '/dev', '/run')}
    tar_suffixes = ('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')
    def onerror(error):
        errors.append(str(error.filename))
    def namespaced(name):
        parts = [part for part in name.replace('\\', '/').lstrip('/').split('/')
                 if part and part != '.']
        if any(part in modules for part in parts):
            return True
        return bool(parts) and parts[-1].lower().endswith(('.py', '.pyc', '.pyo')) \
            and parts[-1].split('.')[0] in modules
    for directory, dirs, files in os.walk(scan_root, topdown=True, onerror=onerror):
        current = Path(directory)
        dirs[:] = [d for d in dirs if current / d not in excluded
                   and (current / d).resolve() != root]
        if current.resolve() == root or root in current.resolve().parents:
            dirs[:] = []
            continue
        for name in list(dirs):
            path = current / name
            if name in modules and path.resolve() != root and root not in path.resolve().parents:
                hits.append(str(path)); dirs.remove(name)
        for name in files:
            scanned += 1
            path = current / name
            if path.suffix.lower() in {'.py', '.pyc', '.pyo'} and namespaced(name):
                if path.resolve() != root and root not in path.resolve().parents:
                    hits.append(str(path))
            elif path.suffix.lower() in {'.whl', '.egg', '.zip'}:
                try:
                    with zipfile.ZipFile(path) as archive:
                        if any(namespaced(part) for part in archive.namelist()):
                            hits.append(str(path))
                except (OSError, zipfile.BadZipFile):
                    errors.append(str(path))
            elif path.name.lower().endswith(tar_suffixes):
                try:
                    with tarfile.open(path, 'r:*') as archive:
                        if any(namespaced(part) for part in archive.getnames()):
                            hits.append(str(path))
                except (OSError, tarfile.TarError, EOFError):
                    errors.append(str(path))
    return dict(modules=sorted(modules), external_copies=sorted(set(hits)),
                files_inspected=scanned, unreadable_paths=sorted(set(errors)),
                scope='named Python namespaces and py/pyc/wheel/egg/zip/tar entries; excludes proc/sys/dev/run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--sanitize-git', action='store_true')
    parser.add_argument('--audit-copies', action='store_true')
    parser.add_argument('--module', action='append', default=[])
    parser.add_argument('--base', help='public task base commit; must be HEAD or an ancestor of HEAD')
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(args.root).resolve()
    if args.sanitize_git:
        repos = sanitize_git(root, base=args.base)
    else:
        found = repositories(root)
        repos = [dict(path=str(r), **inspect_git(r, cutoff_commit=args.base if r == found[0] else None),
                      leftover_modules=[str(p) for p in leftover_modules(r, found)]) for r in found]
    result = dict(schema_version=2, root=str(root), base=args.base, repositories=repos)
    git_passed = all(converged(r) for r in repos)
    result['git_passed'] = git_passed
    if args.audit_copies:
        copies = audit_source_copies(root, args.module or project_modules(root))
        result['source_copies'] = copies
        result['source_copies_passed'] = bool(copies['modules']) and not copies['external_copies'] and not copies['unreadable_paths']
    result['elapsed_s'] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if git_passed and result.get('source_copies_passed', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
