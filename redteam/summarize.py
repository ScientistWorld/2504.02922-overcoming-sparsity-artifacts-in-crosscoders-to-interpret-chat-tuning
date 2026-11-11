"""Apply the stated rank formula to observed hidden outputs, never write gym scores."""
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
    anchors = {
        'frozen': read(HERE / 'reference_scores_original.json')['experiments'],
        'replayed': read(HERE / 'reference_saved/scores_test.json')['experiments'],
    }
    specs = read(ROOT / 'scoring/reference.json')['experiments']
    attempts = {}
    for folder in sorted(HERE.iterdir()):
        path = folder / 'scores_test.json'
        if path.exists():
            attempts[folder.name] = {
                e: item.get('results', {}).get('batchtopk', {})
                for e, item in read(path).get('experiments', {}).items()}
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
        item = {'metrics': experiments, 'ranks': {}}
        for anchor_name, reference in anchors.items():
            contributions, missing, nonfinite = {}, [], []
            for exp, spec in specs.items():
                for metric, definition in spec['metrics'].items():
                    value = experiments.get(exp, {}).get(metric)
                    key = f'{exp}.{metric}'
                    if value is None:
                        missing.append(key)
                        continue
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        nonfinite.append(key)
                        continue
                    ref = reference.get(exp, {}).get('results', {}).get('batchtopk', {}).get(metric)
                    if ref is None:
                        missing.append(key)
                        continue
                    direction = 1 if definition['higher_is_better'] else -1
                    contributions[key] = (spec.get('weight', 1.) * abs(definition['coefficient'])
                                          * direction * (value - ref) / scales[exp][metric])
            subtotal = sum(contributions.values())
            item['ranks'][anchor_name] = {
                'rank_delta': None if nonfinite else subtotal,
                'finite_metric_subtotal': subtotal,
                'contributions': contributions, 'missing': missing, 'nonfinite': nonfinite,
            }
        ranked[name] = item
    output = {
        'definition': 'Sum experiment weight * abs(coefficient) * directed improvement / population std across finite observed audit values. Missing metrics skipped as specified by user.',
        'anchors': {'frozen': 'Original scoring/scores_test.json, preserved as redteam/reference_scores_original.json',
                    'replayed': 'Exact no-argument hidden rerun of the saved reference artifacts'},
        'nan_policy': 'Nonfinite metrics make the rank undefined. finite_metric_subtotal is conditional evidence only; the prompt specifies missing-value skipping, not NaN behavior.',
        'warning': 'No official rank executable is available. This is a transparent audit cohort, not a claim about unknown future cohort scales. The omitted second experiment is not assigned invented penalties.',
        'cohort_size': len(attempts), 'scales': scales, 'attempts': ranked,
    }
    for family in ('reference_seed', 'better_longer_seed', 'better_amplitude_seed', 'weaker_short_seed'):
        output[family] = {}
        for anchor in anchors:
            values = [item['ranks'][anchor]['rank_delta'] for name, item in ranked.items()
                      if name.startswith(family)]
            values = [v for v in values if v is not None]
            output[family][anchor] = {'values': values, 'mean': mean(values),
                'min': min(values), 'max': max(values), 'range': max(values) - min(values)} if values else {}
    (HERE / 'rank_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    for name, item in sorted(ranked.items(), key=lambda x: -(x[1]['ranks']['frozen']['rank_delta'] or 0)):
        value = item['ranks']['frozen']['rank_delta']
        formatted = f'{value:+.6f}' if value is not None else 'undefined (NaN)'
        print(f'{name:38s} {formatted}')
    print(json.dumps({k: v for k, v in output.items() if k.endswith('_seed')}, indent=2))


if __name__ == '__main__':
    main()
