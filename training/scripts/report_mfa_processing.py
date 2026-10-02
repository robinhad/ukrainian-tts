"""Audit the complete MFA sweep and plot median SigMOS with population SD."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import yaml

from training.quality.common import file_hash, read_rows, write_json, write_tables
from training.scripts.plot_processing_sigmos import LABELS


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite scores')
    return {'count': len(values), 'median': float(np.median(values)),
            'mean': float(np.mean(values)), 'stddev_population': float(np.std(values, ddof=0)),
            'p05': float(np.percentile(values, 5)), 'p95': float(np.percentile(values, 95))}


def selections(profiles, names, best_method, threshold):
    ids = profiles['normalized_original'].keys()
    policies = {'normalized_best': ['normalized_original', best_method],
                'brute_force': ['normalized_original'] + sorted(set(names) - {'normalized_original'})}
    selected = []
    for policy, candidates in policies.items():
        for identifier in ids:
            # Stable exact ties favor normalized original, then alphabetical name.
            winner = max(candidates, key=lambda name: profiles[name][identifier]['sigmos_overall'])
            row = profiles[winner][identifier]
            selected.append({'sample_id': row['sample_id'], 'source_id': row['source_id'],
                             'policy': policy, 'selected_profile': winner,
                             'audio_sha256': row['audio_sha256'], 'retained': row['sigmos_overall'] >= threshold,
                             'duration_seconds': row['duration_seconds'],
                             **{k: v for k, v in row.items() if k.startswith('sigmos_')}})
    return selected


def report(run, config_path, native_path, output):
    config = yaml.safe_load(Path(config_path).read_text())
    names = list(config['profiles'])
    completed = json.loads((run / 'complete.json').read_text())
    if completed['status'] != 'complete' or completed['profiles'] != names:
        raise ValueError('Sweep is not complete')
    panel = read_rows(run / 'panel.jsonl')
    expected = {r['utterance_id']: r for r in panel}
    if (len(expected) != 800 or len(panel) != 800
            or sorted(Counter(r['source_id'] for r in panel).values()) != [100] * 8):
        raise ValueError('Expected the same balanced 800-item panel')
    profiles, public = {}, []
    for name in ['original'] + names:
        rows = read_rows(native_path if name == 'original' else run / name / 'per_file.jsonl')
        indexed = {r['utterance_id']: r for r in rows}
        if len(indexed) != len(rows) or indexed.keys() != expected.keys():
            raise ValueError('Score coverage differs between variants')
        for identifier, row in indexed.items():
            source = expected[identifier]
            if row['reference_sha256'] != source['reference_sha256'] or row['source_id'] != source['source_id']:
                raise ValueError('Score references do not match')
            path = source['audio_path'] if name == 'original' else run / name / 'wav' / (identifier + '.wav')
            if file_hash(path) != row['audio_sha256']:
                raise ValueError('Scored audio changed')
            if name != 'original' and (row['processing_input_sha256'] != source['processing_input_sha256']
                                      or row['boundary_method'] != 'mfa'):
                raise ValueError('Methods did not use the same MFA input')
            row['sample_id'] = source['sample_id']
            metrics = {k: float(v) for k, v in row.items() if k.startswith('sigmos_')}
            if len(metrics) != 7 or not all(np.isfinite(v) and 1 <= v <= 5 for v in metrics.values()):
                raise ValueError('Expected seven valid SigMOS scores per recording')
            public.append({'profile': name, 'sample_id': source['sample_id'], 'source_id': source['source_id'],
                           'audio_sha256': row['audio_sha256'], 'duration_seconds': row['duration_seconds'],
                           'boundary_method': 'native_untrimmed_reference' if name == 'original' else 'mfa',
                           'mfa_status': source['mfa_status'] if name != 'original' else None, **metrics})
        profiles[name] = indexed
    enhancement_names = [n for n in names if config['profiles'][n]['backend'] != 'identity']
    best_method = max(enhancement_names, key=lambda n: np.median([r['sigmos_overall'] for r in profiles[n].values()]))
    selected = selections(profiles, names, best_method, 3.5)
    groups = {name: [r for r in public if r['profile'] == name] for name in profiles}
    for policy in ['normalized_best', 'brute_force']:
        groups[policy] = [r for r in selected if r['policy'] == policy]
        groups[policy + '_ge3_5'] = [r for r in groups[policy] if r['retained']]
    aggregates = []
    for name, rows in groups.items():
        for source in ['all'] + sorted({r['source_id'] for r in rows}):
            items = [r for r in rows if source == 'all' or r['source_id'] == source]
            for metric in sorted(k for k in rows[0] if k.startswith('sigmos_')):
                aggregates.append({'profile': name, 'source_id': source, 'metric': metric,
                                   **summarize([r[metric] for r in items])})
    output.mkdir(parents=True, exist_ok=True)
    write_tables(output, 'quality_mfa_processing_per_file', public)
    write_tables(output, 'quality_mfa_processing_selection', selected)
    write_tables(output, 'quality_mfa_processing_summary', aggregates)
    diagnostics = {'selection_mode': 'report_only', 'best_single_enhancement': best_method,
                   'raw_variants': len(names), 'items_per_raw_variant': 800,
                   'standard_deviation': 'population SD across recordings, ddof=0; not a confidence interval',
                   'selection_candidates': names, 'native_reference_excluded_from_selectors': True,
                   'mfa_status_counts': dict(Counter(r['mfa_status'] for r in panel)),
                   'profile_config_sha256': file_hash(config_path),
                   'panel_sha256': file_hash(run / 'panel.jsonl'),
                   'policies': {}}
    for policy in ['normalized_best', 'brute_force']:
        items = groups[policy]
        retained = groups[policy + '_ge3_5']
        diagnostics['policies'][policy] = {
            'retained_ge3_5': len(retained), 'retained_hours': sum(r['duration_seconds'] for r in retained) / 3600,
            'winner_counts_all': dict(Counter(r['selected_profile'] for r in items)),
            'winner_counts_retained': dict(Counter(r['selected_profile'] for r in retained))}
    write_json(output / 'quality_mfa_processing_diagnostics.json', diagnostics)
    for path in output.glob('quality_mfa_processing_*.csv'):
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    chart = [r for r in aggregates if r['source_id'] == 'all' and r['metric'] == 'sigmos_overall']
    plot(chart, best_method, output / 'quality_mfa_processing_median_sigmos')
    print(json.dumps({'diagnostics': diagnostics, 'overall': chart}, indent=2))


def plot(rows, best_method, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {**LABELS, 'identity_wet1': 'MFA crop only',
              'normalized_original': 'Normalized original · MFA',
              'original': 'Native original · untrimmed reference',
              'legacy_cascade': 'Legacy cascade · MFA input',
              'normalized_best': 'Best per item: normalized / best method',
              'normalized_best_ge3_5': 'Best per item + ≥3.5 filter',
              'brute_force': 'Brute force: best of all 16 MFA variants',
              'brute_force_ge3_5': 'Brute force + ≥3.5 filter'}
    labels = {k: v.replace('  [V10 training]', '').replace('  [V11 cascade]', '') for k, v in labels.items()}
    raw = sorted([r for r in rows if r['profile'] not in
                  {'normalized_best', 'normalized_best_ge3_5', 'brute_force', 'brute_force_ge3_5'}],
                 key=lambda r: r['median'], reverse=True)
    policies = [next(r for r in rows if r['profile'] == name) for name in
                ['normalized_best', 'normalized_best_ge3_5', 'brute_force', 'brute_force_ge3_5']]
    ordered = raw + policies
    positions = list(range(len(raw))) + list(np.arange(len(raw), len(ordered)) + .8)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11,
                         'figure.facecolor': '#fffff8', 'axes.facecolor': '#fffff8',
                         'savefig.facecolor': '#fffff8', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': False})
    fig, ax = plt.subplots(figsize=(16, 10.5))
    fig.subplots_adjust(left=.405, right=.84, bottom=.11, top=.85)
    for row, y in zip(ordered, positions):
        color = '#4e79a7' if row['profile'] in {best_method, 'brute_force_ge3_5'} else '#666666'
        ax.errorbar(row['median'], y, xerr=row['stddev_population'], fmt='o', markersize=4,
                    color=color, elinewidth=1.1, capsize=2)
        ax.text(1.03, y, f"{row['median']:.3f}    {row['stddev_population']:.3f}    {row['count']:>3}",
                transform=ax.get_yaxis_transform(), va='center', color=color, fontsize=11,
                fontfamily='DejaVu Sans Mono')
    ax.set_yticks(positions, [labels[r['profile']] for r in ordered])
    ax.tick_params(axis='y', length=0, pad=12, colors='#333333')
    ax.invert_yaxis()
    limits = [r['median'] + sign * r['stddev_population'] for r in rows for sign in [-1, 1]]
    ax.set_xlim(min(limits) - .08, max(limits) + .08)
    ax.spines['bottom'].set_bounds(min(limits), max(limits))
    ax.spines['bottom'].set_color('#aaaaaa')
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', colors='#666666')
    ax.set_xlabel('Overall SigMOS · higher is better', labelpad=12, color='#555555')
    ax.axvline(3.5, color='#aaaaaa', linestyle=':', linewidth=.8, zorder=0)
    ax.text(1.03, 1.035, 'Median     SD      n', transform=ax.transAxes,
            fontfamily='DejaVu Sans Mono', fontsize=11, color='#555555')
    best = next(r for r in rows if r['profile'] == best_method)
    fig.text(.055, .953, f"Best single MFA method: {labels[best_method]} · median {best['median']:.3f}", fontsize=19)
    fig.text(.055, .917, 'Same 800 recordings · 100 per source · all processing uses shared MFA boundaries', fontsize=13, color='#555555')
    fig.text(.055, .887, 'Dots: medians. Whiskers: median ± population SD across recordings; these are not confidence intervals.', fontsize=12, color='#555555')
    fig.text(.055, .049, '710 aligned crops; 90 flagged alignments preserved untrimmed. Native reference is excluded from both selectors.', fontsize=11, color='#555555')
    fig.text(.055, .025, 'Bottom four rows are per-item selection policies. Filtered rows contain fewer, higher-scoring items; n is shown explicitly.', fontsize=11, color='#555555')
    for suffix in ['png', 'pdf']:
        fig.savefig(prefix.with_suffix('.' + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('training/quality_runs/mfa_processing'))
    parser.add_argument('--config', type=Path, default=Path('training/conf/quality_mfa_sweep.yaml'))
    parser.add_argument('--native', type=Path, default=Path('training/quality_runs/v10/search/original/per_file.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('training/reports'))
    args = parser.parse_args()
    report(args.run, args.config, args.native, args.output)


if __name__ == '__main__':
    main()
