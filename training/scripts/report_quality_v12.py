"""Export portable full-corpus quality/retention findings without local paths."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from training.quality.common import file_hash, read_rows, write_json, write_tables


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0}
    return {'count': len(values), 'mean': float(values.mean()),
            'std_population': float(values.std()),
            **{f'p{q:02d}': float(np.percentile(values, q)) for q in [0, 5, 25, 50, 75, 95, 100]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('training/data/quality_v12'))
    parser.add_argument('--output', type=Path, default=Path('training/reports'))
    args = parser.parse_args()
    complete = json.loads((args.input / 'complete.json').read_text())
    assert complete['status'] == 'complete'
    rows = read_rows(args.input / 'scores.jsonl')
    assert len(rows) == complete['evaluated']
    boundaries = {}
    for path in (args.input / 'decisions').glob('*.json'):
        decision = json.loads(path.read_text())
        assert decision['run_key'] == complete['run_key']
        audit = decision['mfa']
        boundaries[decision['sample_id']] = {
            'input_duration_seconds': audit['input_frames'] / 24000,
            'removed_seconds': (audit['input_frames'] - audit['output_frames']) / 24000,
            'mfa_flags': audit['flags'],
        }
    assert len(boundaries) == len(rows)
    for row in rows:
        row.update(boundaries[row['sample_id']])
    sources = ['all'] + sorted({r['source_id'] for r in rows})
    summaries = []
    for source in sources:
        items = [r for r in rows if source == 'all' or r['source_id'] == source]
        for label, subset in [('all_processed', items),
                              ('train_dev_before_filter', [r for r in items if not r['heldout']]),
                              ('train_dev_ge3.5', [r for r in items if not r['heldout'] and r['passed_threshold']]),
                              ('heldout_unfiltered', [r for r in items if r['heldout']])]:
            summaries.append({'source_id': source, 'population': label, 'items': len(subset),
                              'hours': sum(r['duration_seconds'] for r in subset) / 3600,
                              'input_hours': sum(r['input_duration_seconds'] for r in subset) / 3600,
                              'sigmos_overall': distribution([r['sigmos_overall'] for r in subset]),
                              'duration_seconds': distribution([r['duration_seconds'] for r in subset]),
                              'input_duration_seconds': distribution([r['input_duration_seconds'] for r in subset]),
                              'removed_seconds': distribution([r['removed_seconds'] for r in subset]),
                              'mfa_status': dict(Counter(r['mfa_status'] for r in subset)),
                              'mfa_flags': dict(Counter(flag for r in subset for flag in r['mfa_flags'])),
                              'sigmos_dimensions': {name: distribution([r['processed_scores'][name] for r in subset])
                                                    for name in sorted(items[0]['processed_scores'])}})
    write_tables(args.output, 'quality_v12_scores', rows)
    write_tables(args.output, 'quality_v12_distribution', summaries)
    write_json(args.output / 'quality_v12_summary.json', {
        'status': 'complete', 'run_key': complete['run_key'],
        'profile': 'MFA + ClearVoice → Sidon → DeepFilterNet3 (100% wet)',
        'threshold': 3.5, 'threshold_scope': 'train/dev only; fixed heldout unfiltered',
        'scores_sha256': file_hash(args.input / 'scores.jsonl'),
        'coverage': json.loads((args.input / 'coverage.json').read_text()),
        'retention': read_rows(args.input / 'summary.jsonl')})
    print(json.dumps([r for r in summaries if r['source_id'] == 'all'], ensure_ascii=False))


if __name__ == '__main__':
    main()
