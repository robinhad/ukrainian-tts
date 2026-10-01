from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import read_rows, write_json, write_tables
from .evaluate import COMPARE_METRICS, QUALITY_METRICS


def compare(candidate, baselines, output):
    candidate, output = Path(candidate), Path(output)
    meta = json.loads((candidate / 'run.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('Cannot compare an incomplete evaluation')
    current = {r['utterance_id']: r for r in read_rows(candidate / 'per_file.jsonl')}
    details, summaries = [], []
    for role, path in baselines.items():
        path = Path(path)
        other_meta = json.loads((path / 'run.json').read_text())
        for key in ('panel_sha256', 'config', 'models'):
            if meta[key] != other_meta[key]:
                raise ValueError(f'{role}: incompatible {key}')
        if other_meta['status'] != 'complete':
            raise ValueError(f'{role}: incomplete baseline')
        other = {r['utterance_id']: r for r in read_rows(path / 'per_file.jsonl')}
        if current.keys() != other.keys():
            raise ValueError(f'{role}: unmatched file coverage')
        for identifier, row in current.items():
            baseline = other[identifier]
            if row['reference_sha256'] != baseline['reference_sha256']:
                raise ValueError('Reference audio mismatch')
            for metric in COMPARE_METRICS:
                a, b = row.get(metric), baseline.get(metric)
                if a is not None and b is not None:
                    details.append({'utterance_id': identifier, 'source_id': row['source_id'],
                                    'candidate': meta['label'], 'baseline_role': role,
                                    'baseline': other_meta['label'], 'metric': metric,
                                    'candidate_score': a, 'baseline_score': b, 'delta': a - b})
        for metric in COMPARE_METRICS:
            items = [r for r in details if r['baseline_role'] == role and r['metric'] == metric]
            for source in ['all', *sorted({r['source_id'] for r in items})]:
                selected = [r['delta'] for r in items if source == 'all' or r['source_id'] == source]
                if selected:
                    summaries.append({'baseline_role': role, 'metric': metric, 'source_id': source,
                                      'count': len(selected), 'mean_delta': float(np.mean(selected)),
                                      'median_delta': float(np.median(selected)),
                                      'p05_delta': float(np.percentile(selected, 5)),
                                      'p95_delta': float(np.percentile(selected, 95)),
                                      'direction': 'higher' if metric in QUALITY_METRICS + ['ecapa_similarity']
                                                   else 'near_one' if metric == 'duration_ratio' else 'lower'})
            source_means = [r['mean_delta'] for r in summaries if r['baseline_role'] == role
                            and r['metric'] == metric and r['source_id'] not in {'all', 'source_macro'}]
            if source_means:
                summaries.append({'baseline_role': role, 'metric': metric, 'source_id': 'source_macro',
                                  'count': len(source_means), 'mean_delta': float(np.mean(source_means))})
    write_tables(output, 'comparison_per_file', details)
    write_tables(output, 'comparison_aggregate', summaries)
    write_json(output / 'comparison.json', {'selection_mode': 'report_only', 'candidate': meta['label'],
                                          'baselines': {k: str(v) for k, v in baselines.items()},
                                          'automatic_promotion': False})
    return summaries
