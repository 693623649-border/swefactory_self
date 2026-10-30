import hashlib
import json
import subprocess

import pytest

from aweagent.core.eval.production_gate import (
    GATES, SOURCE_DIGEST_RULE, digest, require_acceptance, source_digest,
)


def accepted_receipt(root):
    evidence = root / 'evidence.json'; evidence.write_text('{"verified":true}')
    source = root / 'solver.py'; source.write_text('# frozen fixture\n')
    receipt = dict(schema_version=1, solver_id='dsh-native-v1',
                   gates={gate: {'passed': True, 'evidence': [
                       {'path': evidence.name, 'sha256': digest(evidence)}]} for gate in GATES},
                   sources=[{'path': source.name, 'sha256': digest(source)}])
    path = root / 'acceptance.json'
    path.write_text(json.dumps(receipt))
    return path, receipt


def test_missing_receipt_and_each_unfinished_gate_block_production(tmp_path):
    with pytest.raises(ValueError, match='required'):
        require_acceptance(None, solver_id='dsh-native-v1', project_root=tmp_path)
    path, receipt = accepted_receipt(tmp_path)
    for gate in GATES:
        receipt['gates'][gate]['passed'] = False
        path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match=gate):
            require_acceptance(path, solver_id='dsh-native-v1', project_root=tmp_path)
        receipt['gates'][gate]['passed'] = True


def test_native_acceptance_never_authorizes_legacy_or_changed_code(tmp_path):
    path, receipt = accepted_receipt(tmp_path)
    require_acceptance(path, solver_id='dsh-native-v1', project_root=tmp_path)
    with pytest.raises(ValueError, match='entrypoint'):
        require_acceptance(path, solver_id='python-dsh-compat', project_root=tmp_path)
    (tmp_path / 'solver.py').write_text('# changed implementation')
    with pytest.raises(ValueError, match='source changed'):
        require_acceptance(path, solver_id='dsh-native-v1', project_root=tmp_path)


def test_modified_evidence_is_not_accepted(tmp_path):
    path, receipt = accepted_receipt(tmp_path)
    (tmp_path / 'evidence.json').write_text('{"verified":false}')
    with pytest.raises(ValueError, match='evidence or source changed'):
        require_acceptance(path, solver_id='dsh-native-v1', project_root=tmp_path)


def test_official_low_v1_receipt_never_unlocks_the_schema2_high_chain(tmp_path):
    path, receipt = accepted_receipt(tmp_path)
    with pytest.raises(ValueError, match='entrypoint'):
        require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)
    receipt['solver_id'] = 'dsh-native-v2'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='entrypoint'):
        require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)
    receipt['schema_version'] = 2
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='source digest rule'):
        require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)
    receipt['source_digest_rule'] = SOURCE_DIGEST_RULE
    receipt['sources'] = [{'path': 'solver.py', 'sha256': source_digest(tmp_path / 'solver.py')}]
    path.write_text(json.dumps(receipt))
    assert require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)


SOLVER = b'def solve(task):\n    return task.strip()\n'
BINARY = b'\x7fELF\x00\x01\r\nbinary payload\r\n'


def checkout(origin, target, autocrlf):
    subprocess.run(['git', '-c', 'core.autocrlf=' + autocrlf, 'clone', '-q', str(origin), str(target)], check=True)
    return target


def schema2_receipt(sealed_root):
    # Sealed from one checkout; evidence stays raw bytes, sources follow the versioned rule.
    evidence = sealed_root / 'evidence.json'; evidence.write_bytes(b'{"verified":true}\n')
    receipt = dict(schema_version=2, solver_id='dsh-native-v2', source_digest_rule=SOURCE_DIGEST_RULE,
                   gates={gate: {'passed': True, 'evidence': [
                       {'path': evidence.name, 'sha256': digest(evidence)}]} for gate in GATES},
                   sources=[{'path': name, 'sha256': source_digest(sealed_root / name)}
                            for name in ('solver.py', 'tool.bin')])
    return receipt


def write_receipt(root, receipt):
    path = root / 'acceptance.json'
    path.write_text(json.dumps(receipt))
    (root / 'evidence.json').write_bytes(b'{"verified":true}\n')
    return path


def test_schema2_source_identity_is_the_committed_text_under_any_autocrlf_checkout(tmp_path):
    # Field case: the same submodule commit checked out with core.autocrlf=true turned the
    # LF blob into CRLF on disk and a raw-byte receipt reported "source changed".
    origin = tmp_path / 'origin'; origin.mkdir()
    subprocess.run(['git', 'init', '-q', str(origin)], check=True)
    for key, value in (('user.name', 'Fixture'), ('user.email', 'fixture@example.invalid'),
                       ('core.autocrlf', 'false'), ('commit.gpgsign', 'false')):
        subprocess.run(['git', '-C', str(origin), 'config', key, value], check=True)
    (origin / 'solver.py').write_bytes(SOLVER)
    (origin / 'tool.bin').write_bytes(BINARY)
    subprocess.run(['git', '-C', str(origin), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(origin), 'commit', '-qm', 'frozen'], check=True)
    lf, crlf = checkout(origin, tmp_path / 'lf', 'false'), checkout(origin, tmp_path / 'crlf', 'true')
    assert (lf / 'solver.py').read_bytes() == SOLVER
    assert (crlf / 'solver.py').read_bytes() == SOLVER.replace(b'\n', b'\r\n')
    assert (crlf / 'tool.bin').read_bytes() == BINARY  # Git leaves binary bytes alone
    blob = subprocess.run(['git', '-C', str(origin), 'cat-file', 'blob', 'HEAD:solver.py'],
                          capture_output=True, check=True).stdout
    assert source_digest(crlf / 'solver.py') == hashlib.sha256(blob).hexdigest()

    receipt = schema2_receipt(lf)
    for root in (lf, crlf):
        path = write_receipt(root, receipt)
        assert require_acceptance(path, solver_id='dsh-native-v2', project_root=root, schema_version=2)

    # The same frozen sources under the old raw-byte rule still fail on the CRLF checkout:
    # v1 semantics (and v1 receipts) are not reinterpreted.
    legacy = dict(receipt, schema_version=1, solver_id='dsh-native-v1',
                  sources=[{'path': row['path'], 'sha256': digest(lf / row['path'])} for row in receipt['sources']])
    legacy.pop('source_digest_rule')
    with pytest.raises(ValueError, match='source changed: solver.py'):
        require_acceptance(write_receipt(crlf, legacy), solver_id='dsh-native-v1', project_root=crlf)


@pytest.mark.parametrize('name, edit', [
    ('solver.py', lambda data: data.replace(b'strip()', b'lower()')),        # substantive code change
    ('solver.py', lambda data: data.replace(b'task):', b'task):  ')),        # whitespace outside EOL
    ('solver.py', lambda data: data.replace(b'return', b'\rreturn', 1)),    # lone CR is content
    ('tool.bin', lambda data: data.replace(b'\r\n', b'\n')),                 # binary keeps raw bytes
])
def test_schema2_rule_rejects_every_change_except_crlf_line_endings(tmp_path, name, edit):
    (tmp_path / 'solver.py').write_bytes(SOLVER.replace(b'\n', b'\r\n'))
    (tmp_path / 'tool.bin').write_bytes(BINARY)
    path = write_receipt(tmp_path, schema2_receipt(tmp_path))
    require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)
    target = tmp_path / name
    target.write_bytes(edit(target.read_bytes()))
    with pytest.raises(ValueError, match='source changed: ' + name):
        require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)


def test_schema2_gate_evidence_is_never_line_ending_normalized(tmp_path):
    (tmp_path / 'solver.py').write_bytes(SOLVER)
    (tmp_path / 'tool.bin').write_bytes(BINARY)
    receipt = schema2_receipt(tmp_path)
    path = write_receipt(tmp_path, receipt)
    (tmp_path / 'evidence.json').write_bytes(b'{"verified":true}\r\n')
    with pytest.raises(ValueError, match='evidence or source changed: evidence.json'):
        require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)
    receipt['source_digest_rule'] = 'sha256-git-text-lf-v0'
    path.write_text(json.dumps(receipt))
    (tmp_path / 'evidence.json').write_bytes(b'{"verified":true}\n')
    with pytest.raises(ValueError, match='source digest rule'):
        require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)
