"""Finite processing sweep. Produce comparisons and a Pareto report, never auto-select."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from .common import read_rows, write_json, write_tables
from .evaluate import QUALITY_METRICS

BACKENDS = ['identity', 'deepfilternet3', 'sidon', 'clearervoice',
            'clearervoice_sidon', 'clearervoice_sidon_deepfilternet3']


def run(command):
    subprocess.run([sys.executable, '-m', 'training.quality', *map(str, command)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('training/conf/quality.yaml'))
    parser.add_argument('--backends', nargs='+', default=BACKENDS)
    parser.add_argument('--wet', nargs='+', type=float, default=[1.0, .5])
    parser.add_argument('--model-cache', type=Path, default=Path('training/vendor/enhancement-models'))
    parser.add_argument('--rnnoise-binary', type=Path, default=Path('training/vendor/rnnoise/examples/rnnoise_demo'))
    args = parser.parse_args()
    baseline = args.output / 'original'
    run(['evaluate', '--panel', args.panel, '--label', 'original', '--output', baseline,
         '--config', args.config, '--resume'])
    rows = []
    for backend in args.backends:
        for wet in ([1.] if backend == 'identity' else args.wet):
            name = f'{backend}_wet{wet:g}'
            destination = args.output / name
            profile_path = destination / 'profile.yaml'
            destination.mkdir(parents=True, exist_ok=True)
            profile = {'backend': backend, 'wet': wet, 'model_cache': str(args.model_cache),
                       'rnnoise_binary': str(args.rnnoise_binary)}
            if profile_path.exists() and yaml.safe_load(profile_path.read_text()) != profile:
                raise ValueError('An existing processing profile changed')
            profile_path.write_text(yaml.safe_dump(profile))
            run(['process', '--panel', args.panel, '--profile', profile_path,
                 '--output', destination / 'wav', '--resume'])
            run(['evaluate', '--panel', args.panel, '--wav-dir', destination / 'wav', '--label', name,
                 '--output', destination / 'quality', '--config', args.config, '--resume'])
            run(['compare', '--candidate', destination / 'quality', '--original', baseline,
                 '--output', destination / 'quality'])
            aggregates = read_rows(destination / 'quality/aggregate.jsonl')
            row = {'profile': name, 'backend': backend, 'wet': wet,
                   **{r['metric']: r['mean'] for r in aggregates if r['source_id'] == 'source_macro'}}
            rows.append(row)
            write_tables(args.output, 'combinations', rows)
    # Show non-dominated observed profiles without hiding quality/content tradeoffs.
    higher = QUALITY_METRICS + ['ecapa_similarity']
    lower = ['cer', 'wer', 'clipping_fraction', 'hf_burst_count']
    if yaml.safe_load(args.config.read_text())['models'].get('parakeet'):
        lower += ['parakeet_cer', 'parakeet_wer']
    def dominates(a, b):
        deltas = [a[k] - b[k] for k in higher] + [b[k] - a[k] for k in lower]
        return all(x >= -1e-6 for x in deltas) and any(x > 1e-6 for x in deltas)
    frontier = [r['profile'] for r in rows if not any(dominates(other, r) for other in rows)]
    write_json(args.output / 'search_report.json', {
        'status': 'finite_sweep_complete', 'selection_mode': 'report_only',
        'profiles': len(rows), 'pareto_profiles': frontier, 'selected_profile': None,
        'stopping_rule': 'All declared combinations evaluated; extend around the reviewed frontier before freezing.',
        'global_optimum_claimed': False,
    })


if __name__ == '__main__':
    main()
