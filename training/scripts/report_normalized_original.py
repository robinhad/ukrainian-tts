"""Export sanitized scores and source-wise summaries for normalized selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from training.quality.common import read_rows, write_tables
from training.scripts.evaluate_normalized_original import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('training/quality_runs/normalized_original'))
    parser.add_argument('--processed', type=Path, default=Path('training/quality_runs/v10/search/clearervoice_sidon_deepfilternet3_wet1/quality/per_file.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('training/reports'))
    args = parser.parse_args()
    decisions = read_rows(args.run / 'selection/selection.jsonl')
    def keyed(rows):
        return {hashlib.sha256(json.dumps((r['source_id'], r['utterance_id'])).encode()).hexdigest()[:16]: r for r in rows}
    normalized = keyed(read_rows(args.run / 'per_file.jsonl'))
    processed = keyed(read_rows(args.processed))
    if set(normalized) != set(processed) or set(normalized) != {r['sample_id'] for r in decisions}:
        raise ValueError('All score tables must contain the same recordings')
    metrics = ['sigmos_' + value for value in METRICS.values()]
    exported = []
    for decision in decisions:
        key = decision['sample_id']
        selected = processed[key] if decision['selected_variant'] == 'processed' else normalized[key]
        categories = [('normalized_original', normalized[key]), ('full_cascade', processed[key]),
                      ('normalized_per_item_best', selected)]
        if decision['retained']:
            categories.append(('normalized_per_item_best_ge3_5', selected))
        for category, row in categories:
            exported.append({'sample_id': key, 'source_id': decision['source_id'], 'profile': category,
                             'duration_seconds': decision['duration_seconds'],
                             **{metric: row[metric] for metric in metrics}})
    summaries = []
    for category in sorted({r['profile'] for r in exported}):
        subset = [r for r in exported if r['profile'] == category]
        for source in ['all'] + sorted({r['source_id'] for r in exported}):
            rows = [r for r in subset if source == 'all' or r['source_id'] == source]
            for metric in metrics:
                values = [r[metric] for r in rows]
                summaries.append({'profile': category, 'source_id': source, 'metric': metric,
                                  'count': len(rows), 'hours': sum(r['duration_seconds'] for r in rows) / 3600,
                                  'median': float(np.median(values)) if values else None,
                                  'mean': float(np.mean(values)) if values else None,
                                  **{f'p{q}': float(np.percentile(values, q)) if values else None for q in [5, 25, 75, 95]}})
    write_tables(args.output, 'quality_normalized_original_per_file', exported)
    write_tables(args.output, 'quality_normalized_original_summary', summaries)
    write_tables(args.output, 'quality_normalized_original_selection', decisions)
    # Public reports use repository-standard LF rather than csv's default CRLF.
    for name in ['per_file', 'summary', 'selection']:
        path = args.output / f'quality_normalized_original_{name}.csv'
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))


if __name__ == '__main__':
    main()
