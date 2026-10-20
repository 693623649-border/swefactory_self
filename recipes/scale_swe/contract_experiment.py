"""Reproducible failure-only control/contract experiment on the factory runner.

No test failures, hidden test names, gold patches or reference diffs are added to
teacher messages. The only treatment is a task-independent system suffix.
Post-hoc diagnostics and grading inputs are kept separate from teacher prompts.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / 'results/pool_a50/deepseek-flash_20261020_235140'
DEFAULT_OUT = ROOT / 'results/contract_retry_v1'


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared(value):
    if isinstance(value, str):
        value = json.loads(value or '[]')
    return value or []


def outcome(record, raw):
    detail = (record.get('eval_result') or {}).get('details') or {}
    matched = detail.get('matched') or {}
    files = [line.split(' b/', 1)[-1] for line in (record.get('patch') or '').splitlines()
             if line.startswith('diff --git ')]
    checks = {}
    for name in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
        ids = declared(raw.get(name))
        statuses = dict(Counter(matched.get(node, 'unmatched') for node in ids))
        evidence = 'matched_node_ids'
        if not matched and record.get('success'):
            # Some native successes contain only the evaluator's success marker.
            # Do not present the following aggregate as individually observed nodes.
            statuses = {'passed': len(ids)}
            evidence = 'inferred_from_evaluator_acceptance'
        checks[name] = {
            'declared': len(ids), 'statuses': statuses, 'evidence': evidence,
            'failed_nodes': [node for node in ids if matched.get(node) in ('failed', 'error')],
        }
    return {
        'instance_id': record['instance_id'], 'success': record['success'],
        'finish_reason': record.get('finish_reason'), 'stats': record.get('stats') or {},
        'duration': record.get('duration'), 'patch_chars': len(record.get('patch') or ''),
        'files': files,
        'test_paths_in_patch': [p for p in files if '/tests/' in '/' + p or
                                '/test/' in '/' + p or Path(p).name.startswith('test_') or
                                Path(p).name.endswith('_test.py')],
        'eval_error': detail.get('error'), 'xml_errors': detail.get('xml_errors'),
        'all_matched_statuses': dict(Counter(matched.values())), 'checks': checks,
    }


def prepare(args):
    from aweagent.scaffold.dsh_compat.agent import DSHCompatAgent
    from aweagent.scaffold.dsh_compat.contract_review import CONTRACT_REVIEW_VERSION
    source = Path(args.data)
    rows = load_rows(BASELINE / 'trajectories.jsonl')
    selected = [r for r in rows if not r['success']]
    assert len(rows) == 50 and len(selected) == 25
    raw = {r['instance_id']: r for r in load_rows(source)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('frozen manifest already exists; do not silently replace it')
    ids = [r['instance_id'] for r in selected]
    # Frozen dataset is for Task/Evaluator loading only; get_prompt uses only
    # problem_statement and workdir, not grading metadata.
    dataset = out / 'frozen_tasks.jsonl'
    dataset.write_text(''.join(json.dumps(raw[i], ensure_ascii=False) + '\n' for i in ids), encoding='utf-8')
    issues = out / 'public_issues.json'
    save(issues, [{'instance_id': i, 'workdir': raw[i].get('workdir'),
                   'problem_statement': raw[i].get('problem_statement', '')} for i in ids])
    save(out / 'baseline_audit.json', [outcome(r, raw[r['instance_id']]) for r in selected])
    old = DSHCompatAgent('/workspace')
    new = DSHCompatAgent('/workspace', contract_review=True)
    assert [t.schema for t in old.get_tools()] == [t.schema for t in new.get_tools()]
    for record in selected:
        assert record['messages'][0]['content'] == old.get_system_prompt({}), record['instance_id']
    (out / 'control_system.txt').write_text(old.get_system_prompt({}), encoding='utf-8')
    (out / 'contract_system.txt').write_text(new.get_system_prompt({}), encoding='utf-8')
    manifest = {
        'baseline': str(BASELINE), 'baseline_sha256': digest(BASELINE / 'trajectories.jsonl'),
        'source_sha256': digest(source), 'dataset_sha256': digest(dataset),
        'instance_ids': ids, 'protocol': CONTRACT_REVIEW_VERSION,
        'control_prompt_sha256': digest(out / 'control_system.txt'),
        'contract_prompt_sha256': digest(out / 'contract_system.txt'),
        'tool_schema_sha256': hashlib.sha256(json.dumps([t.schema for t in old.get_tools()], sort_keys=True).encode()).hexdigest(),
        'length_gate': 81920, 'max_steps': 60, 'model': 'deepseek-flash',
        'reasoning_effort': 'low', 'per_call_output_cap': 8192,
        'max_concurrent': 6, 'max_retries': 3,
        'design': 'fresh independent attempts; no baseline patch or evaluator feedback replayed',
        'selection_bias': 'old failures only; recovery on control is resampling, not treatment gain',
    }
    save(out / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if 'sha256' not in k}, ensure_ascii=False, indent=2))


async def run_arm(args):
    # Set proxy exceptions before constructing httpx clients.
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = 'localhost,127.0.0.1'
    from aweagent.core.config.schema import AweAgentConfig
    from aweagent.tasks.scale_swe.task import ScaleSWETask
    from aweagent.scaffold.dsh_compat.agent import DSHCompatAgent
    from recipes.scale_swe.run import _mode_batch
    out = Path(args.out)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert digest(BASELINE / 'trajectories.jsonl') == manifest['baseline_sha256']
    assert digest(out / 'frozen_tasks.jsonl') == manifest['dataset_sha256']
    arm_root = out / args.arm
    if arm_root.exists():
        raise RuntimeError(f'arm already exists: {arm_root}; inspect/resume explicitly, do not rerun')
    enabled = args.arm == 'contract'
    agent = DSHCompatAgent('/workspace', contract_review=enabled)
    expected = manifest['contract_prompt_sha256' if enabled else 'control_prompt_sha256']
    prompt_file = out / ('contract_system.txt' if enabled else 'control_system.txt')
    assert digest(prompt_file) == expected
    assert agent.get_system_prompt({}) == prompt_file.read_text(encoding='utf-8')
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('DEEPSEEK_API_KEY must be provided through the process environment')
    snapshot = json.loads((BASELINE / 'run_config.json').read_text(encoding='utf-8'))
    raw_config = snapshot['config']
    raw_config['llm']['api_key'] = key
    raw_config['task']['data_file'] = str(out / 'frozen_tasks.jsonl')
    raw_config['execution']['output_path'] = str(arm_root)
    raw_config['execution']['max_concurrent'] = manifest['max_concurrent']
    raw_config['extra']['dsh_contract_review'] = enabled
    cfg = AweAgentConfig(**raw_config)
    assert cfg.agent.max_steps == 60
    assert cfg.llm.model == manifest['model']
    assert cfg.llm.params['reasoning_effort'] == 'low'
    assert cfg.llm.params['max_completion_tokens'] == 8192
    task = ScaleSWETask(data_file=str(out / 'frozen_tasks.jsonl'))
    await _mode_batch(cfg, task, manifest['instance_ids'], False)
    runs = list(arm_root.glob('*/trajectories.jsonl'))
    if len(runs) != 1:
        raise RuntimeError(f'expected exactly one native run under {arm_root}')
    save(out / f'{args.arm}_complete.json', {
        'run_dir': str(runs[0].parent), 'trajectories_sha256': digest(runs[0]),
        'result_rows': len(load_rows(runs[0].parent / 'results.jsonl')),
        'trajectory_rows': len(load_rows(runs[0])),
    })


def report(args):
    out = Path(args.out)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    raw = {r['instance_id']: r for r in load_rows(out / 'frozen_tasks.jsonl')}
    arms = {}
    for arm in ('control', 'contract'):
        receipt = json.loads((out / f'{arm}_complete.json').read_text(encoding='utf-8'))
        path = Path(receipt['run_dir'])
        records = load_rows(path / 'trajectories.jsonl')
        summaries = load_rows(path / 'results.jsonl')
        assert {r['instance_id'] for r in summaries} == set(manifest['instance_ids'])
        byid = {r['instance_id']: r for r in records}
        arms[arm] = {'run_dir': str(path), 'instances': {}, 'tokens': {}, 'successes': 0,
                     'missing_trajectories': [], 'infra': []}
        for i in manifest['instance_ids']:
            r = byid.get(i)
            if r is None:
                arms[arm]['missing_trajectories'].append(i)
                continue
            info = outcome(r, raw[i])
            arms[arm]['instances'][i] = info
            arms[arm]['successes'] += bool(r['success'])
            if info['eval_error']:
                arms[arm]['infra'].append(i)
        for field in ('total_prompt_tokens', 'total_completion_tokens', 'llm_calls', 'steps'):
            arms[arm]['tokens'][field] = sum(r['stats'].get(field, 0) for r in arms[arm]['instances'].values())
    both = []
    for i in manifest['instance_ids']:
        c, t = arms['control']['instances'].get(i), arms['contract']['instances'].get(i)
        state = 'infra_or_missing' if c is None or t is None or c['eval_error'] or t['eval_error'] else (
            'both_pass' if c['success'] and t['success'] else 'gain' if t['success'] else
            'loss' if c['success'] else 'both_fail')
        both.append({'instance_id': i, 'comparison': state})
    save(out / 'comparison.json', {'arms': arms, 'paired': both,
                                   'counts': dict(Counter(x['comparison'] for x in both))})
    print(json.dumps({'counts': dict(Counter(x['comparison'] for x in both)),
                      'arms': {k: {f: v[f] for f in ('run_dir', 'successes', 'tokens', 'infra', 'missing_trajectories')}
                               for k, v in arms.items()}}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'report'])
    parser.add_argument('--data', default=str(Path(os.environ.get('TEMP', '.')) / 'scaleswe/factory_sample.jsonl'))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--arm', choices=['control', 'contract'])
    args = parser.parse_args()
    log_path = str(Path(args.out) / f'{args.arm}.native.log') if args.stage == 'run' else None
    logging.basicConfig(level=logging.INFO, filename=log_path, encoding='utf-8',
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'report':
        report(args)
    else:
        if not args.arm:
            parser.error('--arm is required for run')
        asyncio.run(run_arm(args))


if __name__ == '__main__':
    main()
