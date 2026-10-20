"""Opt-in contract review regression checks, with no teacher calls."""
import unittest
from types import SimpleNamespace

from aweagent.scaffold.dsh_compat.agent import DATA_CONTRACT, DSH_PERSONA, DSHCompatAgent
from aweagent.scaffold.dsh_compat.contract_review import CONTRACT_REVIEW


class ContractReviewTests(unittest.TestCase):
    def test_default_is_byte_identical_to_old_prompt(self):
        self.assertEqual(DSHCompatAgent('/workspace').get_system_prompt({}),
                         DSH_PERSONA + DATA_CONTRACT)

    def test_only_prompt_changes_not_tool_schemas_or_stop(self):
        baseline = DSHCompatAgent('/workspace')
        contract = DSHCompatAgent('/workspace', contract_review=True)
        self.assertEqual([t.schema for t in baseline.get_tools()],
                         [t.schema for t in contract.get_tools()])
        self.assertEqual([t.name for t in contract.get_tools()], ['bash', 'read', 'edit'])
        self.assertIsNone(contract.get_no_tool_call_prompt())
        self.assertEqual(contract.get_system_prompt({}),
                         baseline.get_system_prompt({}) + CONTRACT_REVIEW)

    def test_metadata_cannot_inject_hidden_eval_instructions(self):
        agent = DSHCompatAgent('/workspace', contract_review=True)
        sentinel = 'HIDDEN_TEST_OR_GOLD_PATCH_SENTINEL'
        metadata = dict.fromkeys(['f2p_patch', 'gold_patch', 'raw', 'matched',
                                  'eval_result', 'FAIL_TO_PASS'], sentinel)
        self.assertEqual(agent.get_system_prompt(metadata), agent.get_system_prompt({}))
        self.assertNotIn(sentinel, agent.get_system_prompt(metadata))

    def test_task_user_prompt_excludes_grading_and_gold_fields(self):
        from aweagent.tasks.scale_swe.task import ScaleSWETask
        sentinel = 'HIDDEN_TEST_OR_GOLD_PATCH_SENTINEL'
        task = ScaleSWETask(instances=[{
            'instance_id': 'example', 'workdir': '/workspace/example',
            'problem_statement': 'PUBLIC_ISSUE', 'patch': sentinel,
            'f2p_patch': sentinel, 'f2p_script': sentinel,
            'FAIL_TO_PASS': sentinel, 'PASS_TO_PASS': sentinel}])
        instance = task.get_instances()[0]
        self.assertIn('PUBLIC_ISSUE', task.get_prompt(instance))
        self.assertNotIn(sentinel, task.get_prompt(instance))
        self.assertNotIn(sentinel, str(task.get_task_info(instance)))

    def test_report_separates_target_tests_from_regressions(self):
        from recipes.scale_swe.contract_experiment import outcome
        raw = {'FAIL_TO_PASS': '["target"]', 'PASS_TO_PASS': '["existing"]'}
        record = {'instance_id': 'example', 'success': False,
                  'eval_result': {'details': {'matched': {'target': 'passed', 'existing': 'failed'}}}}
        result = outcome(record, raw)
        self.assertEqual(result['checks']['FAIL_TO_PASS']['statuses'], {'passed': 1})
        self.assertEqual(result['checks']['PASS_TO_PASS']['statuses'], {'failed': 1})
        accepted = outcome({'instance_id': 'example', 'success': True}, raw)
        self.assertEqual(accepted['checks']['FAIL_TO_PASS']['evidence'],
                         'inferred_from_evaluator_acceptance')

    def test_opt_in_via_existing_extra_configuration(self):
        config = SimpleNamespace(
            runtime=SimpleNamespace(workdir='/workspace/example'),
            agent=SimpleNamespace(bash_timeout=120, bash_max_timeout=600),
            extra={'dsh_contract_review': True})
        agent = DSHCompatAgent.from_config_with_constraints(config, None)
        self.assertTrue(agent.get_system_prompt({}).endswith(CONTRACT_REVIEW))
        config.extra = {}
        self.assertEqual(DSHCompatAgent.from_config(config).get_system_prompt({}),
                         DSH_PERSONA + DATA_CONTRACT)


if __name__ == '__main__':
    unittest.main()
