"""Select each panel item's maximum overall SigMOS across every scored variant."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from training.quality.common import file_hash, read_rows, write_json, write_tables
from training.scripts.plot_processing_sigmos import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', type=Path, default=Path('training/quality_runs/v10/search'))
    parser.add_argument('--normalized', type=Path, default=Path('training/quality_runs/normalized_original/per_file.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('training/reports'))
    args = parser.parse_args()
    paths = {name: args.search / name / ('per_file.jsonl' if name == 'original' else 'quality/per_file.jsonl')
             for name in LABELS}
    paths['normalized_original'] = args.normalized
    panels = {}
    for name, path in paths.items():
        rows = read_rows(path)
        panel = {(r['source_id'], r['utterance_id']): r for r in rows}
        if len(rows) != 800 or len(panel) != 800 or sorted(Counter(r['source_id'] for r in rows).values()) != [100] * 8:
            raise ValueError(f'Incomplete or unbalanced panel: {name}')
        if any(not math.isfinite(r['sigmos_overall']) or not 1 <= r['sigmos_overall'] <= 5 for r in rows):
            raise ValueError(f'Invalid scores: {name}')
        panels[name] = panel
    reference = panels['original']
    if any(panel.keys() != reference.keys() for panel in panels.values()):
        raise ValueError('Variant recording keys differ')
    # Stable tie breaking: normalized, native, then alphabetical processing name.
    order = ['normalized_original', 'original'] + sorted(set(panels) - {'normalized_original', 'original'})
    decisions = []
    for key in sorted(reference):
        if any(panel[key]['reference_sha256'] != reference[key]['reference_sha256'] for panel in panels.values()):
            raise ValueError('Native reference hashes differ')
        scores = {name: panels[name][key]['sigmos_overall'] for name in order}
        winner = max(order, key=scores.get)
        selected = panels[winner][key]
        decisions.append({'sample_id': hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16],
                          'source_id': key[0], 'selected_variant': winner,
                          'selected_sigmos_overall': scores[winner], 'retained': scores[winner] >= 3.5,
                          'selected_audio_sha256': selected['audio_sha256'],
                          'duration_seconds': selected['duration_seconds'],
                          **{name + '_sigmos_overall': score for name, score in scores.items()}})
    write_tables(args.output, 'quality_bruteforce_selection', decisions)
    csv = args.output / 'quality_bruteforce_selection.csv'
    csv.write_bytes(csv.read_bytes().replace(b'\r\n', b'\n'))
    retained = [r for r in decisions if r['retained']]
    report = {'variants': order, 'variant_count': len(order), 'count': len(decisions),
              'median_sigmos_overall': median(r['selected_sigmos_overall'] for r in decisions),
              'threshold': 3.5, 'retained': len(retained),
              'filtered_median_sigmos_overall': median(r['selected_sigmos_overall'] for r in retained),
              'retained_hours': sum(r['duration_seconds'] for r in retained) / 3600,
              'winners': dict(Counter(r['selected_variant'] for r in decisions)),
              'retained_winners': dict(Counter(r['selected_variant'] for r in retained)),
              'retained_sources': dict(Counter(r['source_id'] for r in retained)),
              'tie_break_order': order, 'metrics_sha256': {name: file_hash(path) for name, path in paths.items()},
              'scope': 'fixed source-balanced panel; saved-score comparison, no new audio processing'}
    write_json(args.output / 'quality_bruteforce_summary.json', report)
    print(json.dumps({k: report[k] for k in ['count', 'variant_count', 'median_sigmos_overall', 'retained', 'filtered_median_sigmos_overall', 'retained_hours']}))


if __name__ == '__main__':
    main()
