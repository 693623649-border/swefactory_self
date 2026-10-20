"""Post-hoc paired/cost analysis. Never used to construct teacher messages."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def totals(instances):
    sums = {'F2P': Counter(), 'P2P': Counter()}
    inferred = 0
    for value in instances.values():
        for label, key in [('F2P', 'FAIL_TO_PASS'), ('P2P', 'PASS_TO_PASS')]:
            check = value['checks'][key]
            sums[label].update(check['statuses'])
            inferred += check.get('evidence') == 'inferred_from_evaluator_acceptance'
    return {k: dict(v) for k, v in sums.items()}, inferred


def logged_cost(path):
    text = path.read_text(encoding='utf-8')
    calls = re.findall(r'LLM call done:.*?tokens=(\d+)\+(\d+)', text)
    attempts = re.findall(r'Instance (\S+) attempt (\d+)/(\d+) failed', text)
    timestamps = re.findall(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)', text, re.M)
    duration = ((datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S,%f') -
                 datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S,%f')).total_seconds()
                if len(timestamps) > 1 else None)
    return {'successful_llm_calls_including_retried_attempts': len(calls),
            'prompt_tokens_including_retried_attempts': sum(int(p) for p, _ in calls),
            'completion_tokens_including_retried_attempts': sum(int(c) for _, c in calls),
            'failed_instance_attempts': attempts, 'log_span_seconds': duration}


def fraction(instance, key):
    if instance is None or instance.get('eval_error'):
        return 'infra/缺失'
    d = instance['checks'][key]
    return f"{d['statuses'].get('passed', 0)}/{d['declared']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/contract_retry_v1'))
    args = parser.parse_args()
    root = args.out
    comparison = json.loads((root / 'comparison.json').read_text(encoding='utf-8'))
    original = {r['instance_id']: r for r in read_rows(
        json.loads((root / 'manifest.json').read_text(encoding='utf-8'))['baseline'] + '/trajectories.jsonl')
                if not r['success']}
    audits = {r['instance_id']: r for r in json.loads((root / 'baseline_audit.json').read_text(encoding='utf-8'))}
    summary = {'counts': comparison['counts'], 'arms': {}}
    for arm in ['control', 'contract']:
        values = comparison['arms'][arm]
        stats, inferred = totals(values['instances'])
        data = {k: values[k] for k in ['successes', 'tokens', 'infra', 'missing_trajectories']}
        data['tests'] = stats
        data['aggregate_groups_inferred_from_evaluator_acceptance'] = inferred
        data['cost'] = logged_cost(root / f'{arm}.native.log')
        sft = Path(values['run_dir']) / 'sft_qwen.jsonl'
        if sft.exists():
            rows = read_rows(sft)
            seqs = sorted(len(row['ids']) for row in rows)
            data['export'] = {'records': len(rows), 'tokens': sum(seqs),
                'supervised_tokens': sum(sum(row['mask']) for row in rows),
                'min_length': min(seqs) if seqs else None,
                'median_length': statistics.median(seqs) if seqs else None,
                'max_length': max(seqs) if seqs else None}
        success = data['successes']
        data['cost']['prompt_tokens_per_acceptance'] = (
            data['cost']['prompt_tokens_including_retried_attempts'] / success if success else None)
        data['cost']['completion_tokens_per_acceptance'] = (
            data['cost']['completion_tokens_including_retried_attempts'] / success if success else None)
        # Never quietly substitute retained trajectory cost for all-attempt cost.
        if data['cost']['successful_llm_calls_including_retried_attempts'] == 0:
            raise ValueError('native log does not contain trace token receipts')
        summary['arms'][arm] = data
    pairs = comparison['paired']
    assert len(pairs) == len(original) == 25
    header = '| 完整instance_id | 原F2P | 对照F2P/P2P | 契约F2P/P2P | 两臂判分 | 步数 对照→契约 |\n|---|---:|---|---|---|---:|'
    lines = [header]
    names = {'both_pass': '均通过', 'gain': '仅契约通过', 'loss': '仅对照通过',
             'both_fail': '均失败', 'infra_or_missing': 'infra/缺失'}
    for pair in pairs:
        i = pair['instance_id']
        c = comparison['arms']['control']['instances'].get(i)
        t = comparison['arms']['contract']['instances'].get(i)
        fc = fraction(c, 'FAIL_TO_PASS') + ' / ' + fraction(c, 'PASS_TO_PASS')
        ft = fraction(t, 'FAIL_TO_PASS') + ' / ' + fraction(t, 'PASS_TO_PASS')
        cs = c['stats'].get('steps', '?') if c else '?'
        ts = t['stats'].get('steps', '?') if t else '?'
        lines.append(f"| {i} | {fraction(audits[i], 'FAIL_TO_PASS')} | {fc} | {ft} | {names[pair['comparison']]} | {cs}→{ts} |")
    gains = comparison['counts'].get('gain', 0)
    losses = comparison['counts'].get('loss', 0)
    discordant = gains + losses
    summary['exploratory_exact_paired_two_sided_p'] = (
        min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses)+1)) /
            (2 ** discordant)) if discordant else 1.0)
    (root / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (root / 'paired_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
