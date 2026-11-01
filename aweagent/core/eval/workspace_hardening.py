"""Standalone pre-solver Git cleanup and Python source-copy audit.

Run inside a disposable task container, before the model receives any tools.
No task solutions or hidden tests are read. File-copy checks inspect names only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import tomllib
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

    Prefer installed egg metadata or Poetry's package selection. Only projects
    without either declaration use conventional top-level package discovery.
    """
    declared = {name.strip() for base in (Path(root), Path(root) / 'src', Path(root) / 'lib')
                for listing in base.glob('*.egg-info/top_level.txt')
                for name in listing.read_text(encoding='utf-8', errors='replace').splitlines()
                if name.strip().isidentifier()}
    if declared:
        return sorted(declared)
    root = Path(root)
    pyproject = root / 'pyproject.toml'
    if pyproject.is_file():
        poetry = tomllib.loads(pyproject.read_text(encoding='utf-8')).get('tool', {}).get('poetry', {})
        if 'packages' in poetry or poetry.get('name'):
            packages = poetry.get('packages')
            if packages is None:
                # Poetry installs its normalized project name, not sibling dev packages.
                name = re.sub(r'[-_.]+', '_', poetry['name']).lower()
                packages = [{'include': include, 'from': base} for base in ('', 'src')
                            for include in (name, name + '.py')]
            declared = set()
            for package in packages:
                base = root / package.get('from', '')
                for entry in base.glob(package['include']):
                    top = entry.relative_to(base).parts[0]
                    name = Path(top).stem if entry.is_file() else top
                    if name.isidentifier() and (entry.is_dir() or entry.suffix == '.py'):
                        declared.add(name)
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


def _importable(filename):
    """A Python source/bytecode file name that ``import`` can resolve (not ``run-tests.py``)."""
    return filename.lower().endswith(PY_SUFFIXES) and filename.split('.')[0].isidentifier()


def classify_named_path(is_dir, in_package, import_root, members, shape, python_bearing=True):
    """'copy' or 'collision' for one path outside root named like a project module.

    At an import root (sys.path entry, site/dist-packages, egg, vendoring directory,
    archive top level) the name shadows or ships the project: always a copy. Outside
    any other package's tree, a same-named file, or a directory holding any importable
    Python source/bytecode (regular or PEP 420 namespace, even only ``conftest.py``), is
    a copy whatever its layout: it can be an older or later release. Only inside another
    package's tree (an ancestor up to the import root is a package) is a directory
    compared by members, and a file there is a collision. Same-named directories without
    importable Python are collisions. Anything that cannot be compared (project form
    unknown or a single file, no non-generic members) is a copy.
    """
    if import_root or not isinstance(shape, frozenset):
        return 'copy'
    if not in_package:
        return 'copy' if not is_dir or python_bearing else 'collision'
    if not is_dir:
        return 'collision'
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
            inner = [e[len(path) + 1:] for e in entries if e.startswith(path + '/')] if is_dir else []
            in_package = any('/'.join(parent[:end]) in packages for end in range(1, len(parent) + 1))
            verdicts[path, is_dir] = classify_named_path(
                is_dir, in_package, not parent or _import_root_name(parent[-1]), _members(inner),
                shapes.get(name), any(_importable(e.rsplit('/', 1)[-1]) for e in inner))
    return verdicts


def system_package_sources(path):
    """Checksums of OS-owned Python stdlib sources, or no ownership evidence."""
    query = shutil.which('dpkg-query')
    if not query:
        return {}
    try:
        owner = subprocess.run([query, '-S', str(path)], capture_output=True, text=True, timeout=10)
        if owner.returncode or len(owner.stdout.splitlines()) != 1:
            return {}
        package = owner.stdout.partition(': ')[0]
        source = subprocess.run([query, '-W', '-f=${source:Package}', package],
                                capture_output=True, text=True, timeout=10)
        if source.returncode or not re.fullmatch(r'python\d+(?:\.\d+)*', source.stdout.strip()):
            return {}
        listing = subprocess.run([query, '--control-path', package, 'md5sums'],
                                 capture_output=True, text=True, timeout=10)
        if listing.returncode:
            return {}
        records = (line.split(None, 1) for line in Path(listing.stdout.strip()).read_text().splitlines())
        return {Path('/') / name: checksum for checksum, name in records}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def verified_stdlib_collision(path, members, shape, compiler_evidence=None):
    """An unrelated OS stdlib namespace stays a collision only while all Python bytes verify."""
    if not isinstance(shape, frozenset) or not shape or members & shape:
        return False
    records = system_package_sources(path)
    if not records:
        return False
    try:
        for file in path.rglob('*'):
            if not file.is_file() or not file.name.lower().endswith(PY_SUFFIXES):
                continue
            source = file if file.suffix == '.py' else path / _source_name(file.relative_to(path).as_posix())
            if source not in records or hashlib.md5(source.read_bytes()).hexdigest() != records[source]:
                return False
            if file.suffix != '.py':
                data = file.read_bytes()
                optimized = re.search(r'\.opt-([12])\.pyc$', file.name)
                expected = compile(source.read_bytes(), str(source), 'exec', dont_inherit=True,
                                   optimize=int(optimized[1]) if optimized else 0)
                payload = io.BytesIO(data[16:])
                actual = marshal.load(payload)
                if data[:4] != importlib.util.MAGIC_NUMBER or payload.tell() != len(data) - 16:
                    return False
                if actual != expected:
                    # A second Python patch version can emit different code with the same magic.
                    compiler = shutil.which(f'python{sys.version_info.major}.{sys.version_info.minor}', path=os.defpath)
                    if not compiler:
                        return False
                    compiler = Path(compiler).resolve()
                    binaries = system_package_sources(compiler)
                    if compiler not in binaries or hashlib.md5(compiler.read_bytes()).hexdigest() != binaries[compiler]:
                        return False
                    code = ("import marshal,pathlib,sys; s=pathlib.Path(sys.argv[1]); "
                            "actual=marshal.loads(pathlib.Path(sys.argv[2]).read_bytes()[16:]); "
                            "expected=compile(s.read_bytes(),str(s),'exec',dont_inherit=True,optimize=int(sys.argv[3])); "
                            "print('match' if actual==expected else 'mismatch')")
                    result = subprocess.run([str(compiler), '-I', '-B', '-c', code, str(source), str(file),
                                             optimized[1] if optimized else '0'],
                                            capture_output=True, text=True, timeout=10)
                    if result.returncode or result.stdout.strip() != 'match':
                        return False
                    if compiler_evidence is not None:
                        compiler_evidence.append(dict(bytecode=str(file), compiler=str(compiler)))
    except (OSError, ValueError, EOFError, TypeError, SyntaxError, subprocess.TimeoutExpired):
        return False
    return True


def audit_source_copies(root, modules, scan_root='/', import_roots=None):
    """Look for the project's import namespaces and Python source/archive copies outside root.

    This is a named Python-source check, not a claim to recognize renamed code.
    Same-named members of other packages' trees that share no project members
    (``pygments/formatters/svg.py``, ``numpy/tests``) and same-named directories without
    importable Python are reported as name collisions, see ``classify_named_path``.
    ``import_roots`` defaults to this interpreter's sys.path (the container's Python).
    Pseudo-filesystems are excluded. Archive entries are inspected without extraction.
    """
    root = Path(root).resolve()
    scan_root = Path(scan_root).resolve()
    modules = set(modules)
    roots = {Path(p).resolve() for p in (sys.path if import_roots is None else import_roots) if p}
    roots -= {root, Path('/').resolve()}
    shapes = module_shapes(root, modules)
    hits, collisions, stdlib_collisions, stdlib_compilers, scanned, errors = [], [], [], [], 0, []
    excluded = {Path(p) for p in ('/proc', '/sys', '/dev', '/run')}
    tar_suffixes = ('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')
    def onerror(error):
        errors.append(str(error.filename))
    def outside(path):
        return path.resolve() != root and root not in path.resolve().parents
    def import_root(directory):
        return _import_root_name(directory.name) or directory.resolve() in roots
    def in_package(directory):
        for current in (directory, *directory.parents):
            if _is_package(current):
                return True
            if import_root(current) or current == scan_root:
                return False
        return False
    def judge(path, name, is_dir):
        parent = path.parent.parent if path.parent.name == '__pycache__' else path.parent
        inner = []
        if is_dir:
            try:
                inner = [p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()]
            except OSError:
                errors.append(str(path))
        members = _members(inner)
        verdict = classify_named_path(is_dir, in_package(parent), import_root(parent), members,
                                      shapes.get(name), any(_importable(r.rsplit('/', 1)[-1]) for r in inner))
        if verdict == 'copy' and is_dir and verified_stdlib_collision(path, members, shapes.get(name), stdlib_compilers):
            verdict = 'collision'
            stdlib_collisions.append(str(path))
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
                verified_stdlib_collisions=sorted(set(stdlib_collisions)),
                verified_stdlib_compilers=stdlib_compilers,
                files_inspected=scanned, unreadable_paths=sorted(set(errors)),
                scope='named Python namespaces and py/pyc/wheel/egg/zip/tar entries; excludes proc/sys/dev/run; '
                      'same-named files and importable Python directories (incl. PEP 420) are copies unless inside '
                      'another package tree without shared project members; non-Python same-named dirs and '
                      'OS stdlib namespaces with disjoint members and verified source/bytecode are collisions')


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
