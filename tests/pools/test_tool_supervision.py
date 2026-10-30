"""Failed tool contracts stay as context but cannot become positive targets."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'results'))
import segment_review as sr
from to_sft_qwen import Rendered
from reviewed_export import render_a_row


def pair(name, result, call_id='call', **args):
    return [
        {'role': 'assistant', 'tool_calls': [{'id': call_id, 'function': {
            'name': name, 'arguments': json.dumps(args)}}]},
        {'role': 'tool', 'tool_call_id': call_id, 'content': result},
    ]


class CharacterRenderer:
    """One token per character exposes ownership without loading model weights."""
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: SimpleNamespace(
        ids=list(map(ord, text)), offsets=[(i, i + 1) for i in range(len(text))]))

    def prefix_text(self, messages, before_last_query):
        return ''.join(json.dumps(m, sort_keys=True) + '\n' for m in messages)


def test_error_call_masked_but_recovery_and_test_failure_preserved():
    messages = [{'role': 'user', 'content': 'Fix a.py'}]
    messages += pair('read', 'Error: path is outside the session workspace', 'bad',
                     file_path='/wrong/a.py')
    messages += pair('bash', '1 failed in 0.1s\n[exit code: 1]', 'test', command='pytest -q')
    messages += pair('edit', 'edited /testbed/a.py: 1 replacement(s)', 'good',
                     file_path='a.py', old_string='1', new_string='2')
    messages += [{'role': 'assistant', 'content': 'Run verification next.'}]
    diag = sr.diagnose({'messages': messages})
    assert diag.duplicate_msg == {1: 'TOOL_CONTRACT_ERROR'}
    assert diag.first_mut_msg == 5
    rendered = Rendered(CharacterRenderer(), messages)
    assert set(rendered.spans()) == {3, 5, 7}
    assert set(rendered.spans(rendered.mask([1, 5]))) == {5}
    assert 'Error: path' in rendered.text  # observations and bad prefix are not deleted
    with pytest.raises(ValueError, match='rebuild rule ledger'):
        render_a_row({'keep_msgs': [1, 5]}, {'instance_id': 'x', 'messages': messages},
                     CharacterRenderer())


def test_failed_edit_cannot_supply_a_correction_or_first_mutation():
    messages = pair('bash', '1 failed in 0.1s', 'fail', command='pytest -q')
    messages += pair('edit', 'Error: old_string not found', 'bad',
                     file_path='a.py', old_string='missing', new_string='new')
    messages += pair('bash', '1 passed in 0.1s', 'pass', command='pytest -q')
    diag = sr.diagnose({'messages': messages})
    assert diag.first_mut_msg is None
    assert diag.failure_then_pass == []
    assert sr.tool_error_messages(messages) == {2: 'TOOL_CONTRACT_ERROR'}


@pytest.mark.parametrize('variant', ['missing', 'before', 'duplicate', 'late'])
def test_invalid_tool_result_link_never_supplies_success(variant):
    messages = pair('edit', 'edited a.py: 1 replacement(s)')
    if variant == 'missing':
        messages.pop()
    elif variant == 'before':
        messages.reverse()
    elif variant == 'duplicate':
        messages.append(dict(messages[-1]))
    else:
        messages.insert(1, {'role': 'assistant', 'content': 'Another step'})
    diag = sr.diagnose({'messages': messages})
    assert diag.first_mut_msg is None
    assert 'MISSING_TOOL_RESULT' in diag.duplicate_msg.values()


def test_source_code_containing_error_is_a_valid_read():
    assert sr.tool_error_messages(pair('read', '     1\tError: sample text\n\n[1 lines total]')) == {}
