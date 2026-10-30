"""The parity check must catch runtime schema drift, not compare with itself."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'results'))
from verify_dsh_alignment import check_schema
from aweagent.scaffold.dsh_compat.agent import DSHCompatAgent


def test_runtime_parameter_and_prompt_drift_fails_alignment():
    tools = [t.schema for t in DSHCompatAgent('/testbed').get_tools()]
    assert check_schema(tools)['schema_equal']
    live = copy.deepcopy(tools)
    live[0]['function']['parameters']['properties']['workdir'] = {'type': 'string'}
    live[1]['function']['description'] = 'Changed model-visible contract'
    result = check_schema(live)
    assert not result['schema_equal']
    assert result['differences'] == ['bash', 'read']
    assert result['factory_schema_sha256'] != result['eval_schema_sha256']
    with pytest.raises(ValueError, match='no tool schemas'):
        check_schema([])
