"""Compute transparent audit-cohort ranks, never generate evaluator metrics."""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path('/home/user')
HERE = ROOT / 'redteam'


def read(path):
    return json.loads(path.read_text())


def main():
    reference = read(HERE / 'reference_saved/scores_test.json')['experiments']
    specs = read(ROOT / 'scoring/reference.json')['experiments']
    attempts = {}
    for folder in sorted(HERE.iterdir()):
        path = folder / 'scores_test.json'
        if not path.exists():
            continue
        experiments = read(path).get('experiments', {})
        attempts[folder.name] = {
            e: item.get('results', {}).get('batchtopk', {})
            for e, item in experiments.items()}
    scales = {}
    for exp, spec in specs.items():
        scales[exp] = {}
        for metric in spec['metrics']:
            values = [entry.get(exp, {}).get(metric) for entry in attempts.values()]
            values = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
            spread = pstdev(values) if len(values) > 1 else 0.
            scales[exp][metric] = spread if spread > 0 else 1.
    ranked = {}
    for name, experiments in attempts.items():
        total = 0.
        contributions = {}
        missing = []
        for exp, spec in specs.items():
            for metric, definition in spec['metrics'].items():
                value = experiments.get(exp, {}).get(metric)
                ref = reference.get(exp, {}).get('results', {}).get('batchtopk', {}).get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or ref is None:
                    missing.append(f'{exp}.{metric}')
                    continue
                sign = 1 if definition['higher_is_better'] else -1
                term = (spec.get('weight', 1.) * abs(definition['coefficient'])
                        * sign * (value - ref) / scales[exp][metric])
                contributions[f'{exp}.{metric}'] = term
                total += term
        ranked[name] = {'rank_delta': total, 'contributions': contributions,
                        'missing_or_nonfinite': missing, 'metrics': experiments}
    output = {
        'definition': 'Sum weight * abs(coefficient) * directed delta / population std across finite audit attempts. Missing/nonfinite skipped. Baseline is exact no-argument reference_saved.',
        'warning': 'Future ranking cohorts can change metric scales; dominance conclusions do not depend on that choice. The second experiment is omitted by the actual no-argument hidden entry point.',
        'scales': scales, 'attempts': ranked,
    }
    for family in ('reference_seed', 'better_longer_seed', 'weaker_short_seed'):
        values = [item['rank_delta'] for name, item in ranked.items() if name.startswith(family)]
        output[family] = {'values': values, 'mean': mean(values),
                          'min': min(values), 'max': max(values),
                          'range': max(values) - min(values)} if values else {}
    (HERE / 'rank_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    for name, item in sorted(ranked.items(), key=lambda x: -x[1]['rank_delta']):
        print(f"{name:38s} {item['rank_delta']:+.6f}")
    print(json.dumps({k: v for k, v in output.items() if k.endswith('_seed')}, indent=2))


if __name__ == '__main__':
    main()
