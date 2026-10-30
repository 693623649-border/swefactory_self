import json

import pytest

from aweagent.core.eval.production_gate import GATES, digest, require_acceptance


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
    assert require_acceptance(path, solver_id='dsh-native-v2', project_root=tmp_path, schema_version=2)
