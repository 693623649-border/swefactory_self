"""Exercise the per-instance tools, not just their advertised parameter names."""
import asyncio
from types import SimpleNamespace

from aweagent.core.agent.context import AgentContext
from aweagent.scaffold.dsh_compat.agent import DSHCompatAgent


def test_batch_instance_root_and_concurrent_contexts(mock_session):
    async def exercise():
        agent = DSHCompatAgent('/testbed')  # global runtime default
        contexts = []
        for root in ('/workspace/one', '/workspace/two'):
            mock_session.files[root + '/a.py'] = b'answer = 1\n'
            ctx = AgentContext(llm=SimpleNamespace(), session=mock_session,
                               task_info={'workdir': root})
            agent.create_loop(ctx)
            contexts.append(ctx)
            prompt = agent.get_system_prompt(ctx.task_info)
            assert root in prompt and '/testbed' not in prompt
        for ctx in contexts:
            root = ctx.task_info['workdir']
            read = ctx.get_tool('read')
            edit = ctx.get_tool('edit')
            relative = await read.execute({'file_path': 'a.py'}, mock_session)
            absolute = await read.execute({'file_path': root + '/a.py'}, mock_session)
            assert relative == absolute and 'answer = 1' in relative
            result = await edit.execute({'file_path': 'a.py', 'old_string': '1',
                                         'new_string': '2'}, mock_session)
            assert not result.startswith('Error:')
            assert mock_session.files[root + '/a.py'] == b'answer = 2\n'
            assert (await read.execute({'file_path': '/workspace/outside/a.py'},
                                       mock_session)).startswith('Error:')
        # Later contexts must not rebind the tools of the first context.
        assert contexts[0].get_tool('read') is not contexts[1].get_tool('read')
    asyncio.run(exercise())


def test_noop_edit_is_not_reported_as_a_success(mock_session):
    async def exercise():
        mock_session.files['/testbed/a.py'] = b'original'
        agent = DSHCompatAgent('/testbed')
        edit = next(t for t in agent.get_tools() if t.name == 'edit')
        result = await edit.execute({'file_path': 'a.py', 'old_string': 'original',
                                     'new_string': 'original'}, mock_session)
        assert result == 'Error: old_string and new_string must differ'
        assert mock_session.files['/testbed/a.py'] == b'original'
    asyncio.run(exercise())
