import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'results'
sys.path.insert(0, str(RESULTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, RESULTS/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeReleaseContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.renderer = load('to_sft_qwen')
        self.index = load('build_index_db')

    def test_grep_miss_or_expected_red_is_not_tool_contract_error(self):
        messages = [{'role': 'user', 'content': 'fix'}]
        tools = [{'type': 'function', 'function': {'name': 'bash', 'parameters': {'type': 'object'}}}]
        for command, observation in (('grep x .', 'grep: no matches found'),
                                     ('pytest -q', '1 failed in 0.20s'),
                                     ('grep x .', 'source text')):
            assistant = {'role': 'assistant', 'tool_calls': [{'id': 'call', 'type': 'function',
                'function': {'name': 'bash', 'arguments': {'command': command}}}]}
            # Exercise the pure builder only when the local tokenizer is available.
            if (Path('E:/Deeplearning/local_train/model/Qwen3.5-4B/tokenizer.json').exists()
                    and importlib.util.find_spec('tokenizers') and importlib.util.find_spec('jinja2')):
                value = self.renderer.build_decision(self.renderer.Renderer(),
                    {'messages': messages, 'tools': tools}, assistant,
                    [{'role': 'tool', 'tool_call_id': 'call', 'content': observation}],
                    sample_id='fixture', response_status='completed')
                self.assertNotEqual(value['masked_response_reason'], 'TOOL_CONTRACT_ERROR')
                # The assistant decision remains useful; only the failed observation is zero-loss context.
                self.assertGreater(sum(value['mask']), 0)
        assistant = {'role': 'assistant', 'tool_calls': [{'id': 'call', 'type': 'function',
            'function': {'name': 'read', 'arguments': {'file_path': 'missing'}}}]}
        tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]
        if (Path('E:/Deeplearning/local_train/model/Qwen3.5-4B/tokenizer.json').exists()
                and importlib.util.find_spec('tokenizers') and importlib.util.find_spec('jinja2')):
            value = self.renderer.build_decision(self.renderer.Renderer(),
                {'messages': messages, 'tools': tools}, assistant,
                [{'role': 'tool', 'tool_call_id': 'call', 'content': 'Error: missing'}],
                sample_id='fixture', response_status='completed')
            self.assertEqual(value['masked_response_reason'], 'TOOL_CONTRACT_ERROR')
            self.assertFalse(any(value['mask']))

    def test_legacy_index_refuses_rebuild_when_sources_are_incomplete(self):
        with self.assertRaisesRegex(ValueError, 'sources incomplete'):
            self.index.build()

if __name__ == '__main__':
    unittest.main()
