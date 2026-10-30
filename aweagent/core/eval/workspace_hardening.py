"""Standalone pre-solver Git cleanup and Python source-copy audit.

Run inside a disposable task container, before the model receives any tools.
No task solutions or hidden tests are read. File-copy checks inspect names only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import zipfile


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


def inspect_git(root, cutoff_commit=None):
    head = text(git(root, 'rev-parse', 'HEAD'))
    index = git(root, 'diff-index', '--cached', '--quiet', 'HEAD', '--', check=False)
    if index.returncode:
        raise RuntimeError('Git index differs from HEAD: {}'.format(root))
    if cutoff_commit is None:
        saved = git(root, 'rev-parse', '--verify', 'refs/aweagent/pre-solver-base^{commit}', check=False)
        cutoff_commit = text(saved) if saved.returncode == 0 else head
    if git(root, 'merge-base', '--is-ancestor', cutoff_commit, head, check=False).returncode:
        raise RuntimeError('pre-solver base is not an ancestor of HEAD')
    cutoff_time = int(text(git(root, 'show', '-s', '--format=%ct', cutoff_commit)))
    branch = text(git(root, 'symbolic-ref', '-q', 'HEAD', check=False))
    refs = []
    forbidden = []
    for entry in text(git(root, 'for-each-ref', '--format=%(refname)%09%(taggerdate:unix)')).splitlines():
        ref, _, tag_time = entry.partition('\t')
        target = git(root, 'rev-parse', ref + '^{commit}', check=False)
        ancestor = target.returncode == 0 and git(
            root, 'merge-base', '--is-ancestor', text(target), head, check=False).returncode == 0
        allowed = (ref == branch and text(target) == head) or (
            ref.startswith('refs/tags/') and ancestor and (not tag_time or int(tag_time) <= cutoff_time))
        refs.append(ref)
        if not allowed:
            forbidden.append(ref)
    fsck = git(root, 'fsck', '--unreachable', '--no-reflogs', check=False)
    unreachable = [line for line in text(fsck).splitlines() if line.startswith(('unreachable ', 'dangling '))]
    common = Path(text(git(root, 'rev-parse', '--git-common-dir')))
    if not common.is_absolute():
        common = Path(root) / common
    grafts = common / 'info' / 'grafts'
    if grafts.is_file():
        lines = grafts.read_text(encoding='utf-8', errors='replace').splitlines()
        if any(line.strip() and not line.lstrip().startswith('#') for line in lines):
            raise RuntimeError('Git grafts can hide future objects: {}'.format(root))
    alternates = common / 'objects' / 'info' / 'alternates'
    return dict(head=head, cutoff_commit=cutoff_commit, ref_count=len(refs), forbidden_refs=forbidden,
                unreachable_objects=len(unreachable), fsck_exit_code=fsck.returncode,
                object_alternates=alternates.exists())


def sanitize_git(root):
    repos = repositories(root)
    receipts = []
    snapshots = [(repo, inspect_git(repo)) for repo in reversed(repos)]
    for repo, before in snapshots:
        for ref in before['forbidden_refs']:
            git(repo, 'update-ref', '--no-deref', '-d', ref)
        common = Path(text(git(repo, 'rev-parse', '--git-common-dir')))
        if not common.is_absolute():
            common = repo / common
        alternates = common / 'objects' / 'info' / 'alternates'
        if alternates.exists():
            # Materialize reachable objects before severing the shared object store.
            git(repo, 'repack', '-a', '-d')
            alternates.unlink()
        git(repo, 'reflog', 'expire', '--expire=now', '--expire-unreachable=now', '--all')
        git(repo, 'gc', '--prune=now')
        after = inspect_git(repo, cutoff_commit=before['cutoff_commit'])
        if (after['head'] != before['head'] or after['forbidden_refs'] or
                after['unreachable_objects'] or after['object_alternates'] or after['fsck_exit_code']):
            raise RuntimeError('Git isolation did not converge: {}'.format(repo))
        receipts.append(dict(path=str(repo), removed_ref_count=len(before['forbidden_refs']), **after))
    return receipts


def project_modules(root):
    """Conventional Python import roots; callers can supply explicit modules."""
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
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(args.root).resolve()
    repos = sanitize_git(root) if args.sanitize_git else [dict(path=str(r), **inspect_git(r)) for r in repositories(root)]
    result = dict(schema_version=1, root=str(root), repositories=repos)
    git_passed = all(not r['forbidden_refs'] and not r['unreachable_objects'] and
                     not r['object_alternates'] and not r['fsck_exit_code'] for r in repos)
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
