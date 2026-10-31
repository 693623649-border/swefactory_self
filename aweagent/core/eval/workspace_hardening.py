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
import sys
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


PY_SUFFIXES = ('.py', '.pyc', '.pyo')
# Directories whose direct children are import roots: installs, eggs and vendored copies.
SITE_DIRS = {'site-packages', 'dist-packages'}
VENDOR_DIRS = {'_vendor', 'vendor', '_vendored', 'vendored', 'extern', '_extern', 'externals',
               'third_party', '_third_party', 'packages', '_packages'}
# Names every package can have; they say nothing about which project a directory copies.
GENERIC_MEMBERS = {'__init__.py', '__main__.py', 'conftest.py'}


def _source_name(relative):
    """``a/__pycache__/b.cpython-311.pyc`` -> ``a/b.py``; None for non-Python names."""
    parts = relative.split('/')
    if not parts[-1].lower().endswith(PY_SUFFIXES):
        return None
    folders = parts[:-1]
    if folders and folders[-1] == '__pycache__':
        folders = folders[:-1]
    return '/'.join([*folders, parts[-1].split('.')[0] + '.py'])


def _members(relatives):
    names = {_source_name(r) for r in relatives}
    return {n for n in names if n and n.rsplit('/', 1)[-1] not in GENERIC_MEMBERS}


def module_shapes(root, modules):
    """Each module's form in the checkout: its member set (package), 'module', or None (not found)."""
    root = Path(root)
    shapes = {}
    for name in modules:
        shapes[name] = None
        for base in (root, root / 'src', root / 'lib'):
            if (base / name).is_dir():
                shapes[name] = frozenset(_members(p.relative_to(base / name).as_posix()
                                                  for p in (base / name).rglob('*') if p.is_file()))
                break
            if (base / f'{name}.py').is_file():
                shapes[name] = 'module'
                break
    return shapes


def classify_named_path(is_dir, parent_is_package, import_root, members, shape, is_package=False):
    """'copy' or 'collision' for one path outside root named like a project module.

    At an import root (sys.path entry, site/dist-packages, egg, vendoring directory,
    archive top level) the name shadows or ships the project: always a copy. Under an
    ordinary directory a same-named package is a copy too: a different layout can be an
    older or later release. Only a non-package directory, or a submodule of another
    package, is compared by members; a file is a collision only inside a package.
    Anything that cannot be compared (project form unknown or a single file, no
    non-generic members) is a copy.
    """
    if import_root or not isinstance(shape, frozenset):
        return 'copy'
    if not is_dir:
        return 'collision' if parent_is_package else 'copy'
    if is_package and not parent_is_package:
        return 'copy'
    if not shape:
        return 'copy'
    return 'copy' if members & shape else 'collision'


def _import_root_name(name):
    return name in SITE_DIRS or name in VENDOR_DIRS or name.endswith('.egg')


def _is_package(directory):
    return (directory / '__init__.py').is_file() or (directory / '__init__.pyc').is_file() \
        or any((directory / '__pycache__').glob('__init__.*.pyc'))


def _archive_verdicts(names, modules, shapes):
    """Classify same-named entries of one archive listing, without extraction."""
    entries = {'/'.join(p for p in n.replace('\\', '/').split('/') if p and p != '.') for n in names}
    entries.discard('')
    packages = set()
    for entry in entries:
        source = _source_name(entry)
        if source and source.rsplit('/', 1)[-1] == '__init__.py':
            packages.add(source.rpartition('/')[0])
    verdicts = {}
    for entry in entries:
        parts = entry.split('/')
        for index, part in enumerate(parts):
            is_dir = index < len(parts) - 1
            if not is_dir and not part.lower().endswith(PY_SUFFIXES):
                continue
            name = part if is_dir else part.split('.')[0]
            if name not in modules:
                continue
            path = '/'.join(parts[:index + 1])
            if (path, is_dir) in verdicts:
                continue
            parent = parts[:index]
            if parent and parent[-1] == '__pycache__':
                parent = parent[:-1]
            members = _members(e[len(path) + 1:] for e in entries if e.startswith(path + '/')) if is_dir else set()
            verdicts[path, is_dir] = classify_named_path(is_dir, '/'.join(parent) in packages,
                                                         not parent or _import_root_name(parent[-1]),
                                                         members, shapes.get(name), path in packages)
    return verdicts


def audit_source_copies(root, modules, scan_root='/', import_roots=None):
    """Look for the project's import namespaces and Python source/archive copies outside root.

    This is a named Python-source check, not a claim to recognize renamed code.
    Same-named submodules of unrelated packages (``pygments/formatters/svg.py``,
    ``numpy/tests``) are reported as name collisions, see ``classify_named_path``.
    ``import_roots`` defaults to this interpreter's sys.path (the container's Python).
    Pseudo-filesystems are excluded. Archive entries are inspected without extraction.
    """
    root = Path(root).resolve()
    scan_root = Path(scan_root).resolve()
    modules = set(modules)
    roots = {Path(p).resolve() for p in (sys.path if import_roots is None else import_roots) if p}
    roots -= {root, Path('/').resolve()}
    shapes = module_shapes(root, modules)
    hits, collisions, scanned, errors = [], [], 0, []
    excluded = {Path(p) for p in ('/proc', '/sys', '/dev', '/run')}
    tar_suffixes = ('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')
    def onerror(error):
        errors.append(str(error.filename))
    def outside(path):
        return path.resolve() != root and root not in path.resolve().parents
    def judge(path, name, is_dir):
        parent = path.parent.parent if path.parent.name == '__pycache__' else path.parent
        members = set()
        if is_dir:
            try:
                members = _members(p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file())
            except OSError:
                errors.append(str(path))
        verdict = classify_named_path(is_dir, _is_package(parent),
                                      _import_root_name(parent.name) or parent.resolve() in roots,
                                      members, shapes.get(name), is_dir and _is_package(path))
        (hits if verdict == 'copy' else collisions).append(str(path))
        return verdict
    def archive(path, names):
        verdicts = _archive_verdicts(names, modules, shapes).values()
        if 'copy' in verdicts:
            hits.append(str(path))
        elif verdicts:
            collisions.append(str(path))
    for directory, dirs, files in os.walk(scan_root, topdown=True, onerror=onerror):
        current = Path(directory)
        dirs[:] = [d for d in dirs if current / d not in excluded
                   and (current / d).resolve() != root]
        if current.resolve() == root or root in current.resolve().parents:
            dirs[:] = []
            continue
        for name in list(dirs):
            path = current / name
            # A collision is still walked: it may contain a vendored copy further down.
            if name in modules and outside(path) and judge(path, name, True) == 'copy':
                dirs.remove(name)
        for name in files:
            scanned += 1
            path = current / name
            if path.suffix.lower() in PY_SUFFIXES:
                if name.split('.')[0] in modules and outside(path):
                    judge(path, name.split('.')[0], False)
            elif path.suffix.lower() in {'.whl', '.egg', '.zip'}:
                try:
                    with zipfile.ZipFile(path) as opened:
                        archive(path, opened.namelist())
                except (OSError, zipfile.BadZipFile):
                    errors.append(str(path))
            elif path.name.lower().endswith(tar_suffixes):
                try:
                    with tarfile.open(path, 'r:*') as opened:
                        archive(path, opened.getnames())
                except (OSError, tarfile.TarError, EOFError):
                    errors.append(str(path))
    return dict(modules=sorted(modules), external_copies=sorted(set(hits)),
                name_collisions=sorted(set(collisions)),
                files_inspected=scanned, unreadable_paths=sorted(set(errors)),
                scope='named Python namespaces and py/pyc/wheel/egg/zip/tar entries; excludes proc/sys/dev/run; '
                      'outside import roots, same-named paths sharing no project members are collisions, not copies')


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
