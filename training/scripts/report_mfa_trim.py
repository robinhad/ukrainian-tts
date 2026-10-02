"""Sanitized paired MFA/current-trim metrics and alignment diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from training.quality.common import file_hash, read_rows, write_json, write_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('training/quality_runs/mfa_trim'))
    parser.add_argument('--output', type=Path, default=Path('training/reports'))
    args = parser.parse_args()
    candidates = {'mfa_trim': args.run / 'quality/per_file.jsonl',
                  'energy_trim': Path('training/quality_runs/v10/search/identity_wet1/quality/per_file.jsonl'),
                  'native_original': Path('training/quality_runs/v10/search/original/per_file.jsonl')}
    panels = {name: {(r['source_id'], r['utterance_id']): r for r in read_rows(path)} for name, path in candidates.items()}
    if any(len(panel) != 800 or panel.keys() != panels['native_original'].keys() for panel in panels.values()):
        raise ValueError('Expected complete matched 800-item scores for all comparisons')
    metadata = {(r['source_id'], r['utterance_id']): r for r in read_rows(args.run / 'processing.jsonl')}
    names = [name for name in next(iter(panels['mfa_trim'].values()))
             if name.startswith('sigmos_') or name in {'audiobox_pq', 'cer', 'wer', 'parakeet_cer', 'parakeet_wer',
                  'ecapa_similarity', 'duration_seconds', 'clipping_fraction', 'hf_burst_count'}]
    public, summaries = [], []
    for key in sorted(panels['mfa_trim']):
        row, meta = panels['mfa_trim'][key], metadata[key]
        if row['audio_sha256'] != meta['output_sha256']:
            raise ValueError('MFA scored hash does not match rendered audio')
        if any(panel[key]['reference_sha256'] != row['reference_sha256'] for panel in panels.values()):
            raise ValueError('Comparison reference hashes differ')
        public.append({'sample_id': hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16],
                       'source_id': key[0], 'status': meta['status'], 'alignment_flags': meta['flags'],
                       'removed_seconds': meta['removed_seconds'],
                       'trim_start_seconds': meta['start_frame'] / 24000,
                       'trim_end_seconds': meta['end_frame'] / 24000,
                       'audio_sha256': row['audio_sha256'], **{name: row[name] for name in names},
                       'energy_sigmos_overall': panels['energy_trim'][key]['sigmos_overall'],
                       'native_sigmos_overall': panels['native_original'][key]['sigmos_overall']})
    for label, panel in panels.items():
        for source in ['all'] + sorted({key[0] for key in panel}):
            items = [r for key, r in panel.items() if source == 'all' or key[0] == source]
            for metric in names:
                values = [r[metric] for r in items]
                summaries.append({'profile': label, 'source_id': source, 'metric': metric,
                                  'count': len(values), 'mean': float(np.mean(values)),
                                  'median': float(np.median(values)),
                                  'p05': float(np.percentile(values, 5)), 'p95': float(np.percentile(values, 95))})
    write_tables(args.output, 'quality_mfa_per_file', public)
    write_tables(args.output, 'quality_mfa_summary', summaries)
    for name in ['per_file', 'summary']:
        path = args.output / f'quality_mfa_{name}.csv'
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    diagnostics = json.loads((args.run / 'render_summary.json').read_text())
    diagnostics.update({'input_metrics_sha256': {label: file_hash(path) for label, path in candidates.items()},
                        'improved_vs_energy': sum(r['sigmos_overall'] > r['energy_sigmos_overall'] for r in public),
                        'improved_vs_native': sum(r['sigmos_overall'] > r['native_sigmos_overall'] for r in public),
                        'retained_ge3_5': sum(r['sigmos_overall'] >= 3.5 for r in public)})
    for label, panel in panels.items():
        diagnostics[label + '_corpus_errors'] = {
            metric: sum(r[numerator] for r in panel.values()) / sum(r[denominator] for r in panel.values())
            for metric, numerator, denominator in [
                ('cer', 'character_errors', 'reference_characters'), ('wer', 'word_errors', 'reference_words'),
                ('parakeet_cer', 'parakeet_character_errors', 'parakeet_reference_characters'),
                ('parakeet_wer', 'parakeet_word_errors', 'parakeet_reference_words')]}
    write_json(args.output / 'quality_mfa_diagnostics.json', diagnostics)
    print(json.dumps(diagnostics), flush=True)


if __name__ == '__main__':
    main()
