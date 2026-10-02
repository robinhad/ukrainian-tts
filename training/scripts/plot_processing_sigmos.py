#!/usr/bin/env python3
"""Plot median overall SigMOS from the source-balanced processing report."""
import argparse
import csv
import math
from pathlib import Path
from statistics import median

LABELS = {
    'original': 'Native original',
    'identity_wet1': 'Base mono / 24 kHz / trim',
    'deepfilternet3_wet0.5': 'DF3 · 50%',
    'deepfilternet3_wet1': 'DF3 · 100%',
    'deepfilternet3_peakminus3_wet1': 'DF3 · −3 dBFS input · 100%',
    'deepfilternet3_peakminus3_wet0.75': 'DF3 · −3 dBFS input · 75%  [V10 training]',
    'clearervoice_wet0.5': 'ClearVoice · 50%',
    'clearervoice_wet0.75': 'ClearVoice · 75%',
    'clearervoice_wet1': 'ClearVoice · 100%',
    'sidon_wet0.5': 'Sidon · 50%',
    'sidon_wet1': 'Sidon · 100%',
    'clearervoice_sidon_wet0.5': 'ClearVoice → Sidon · 50%',
    'clearervoice_sidon_wet1': 'ClearVoice → Sidon · 100%',
    'clearervoice_sidon_deepfilternet3_wet0.5': 'ClearVoice → Sidon → DF3 · 50%',
    'clearervoice_sidon_deepfilternet3_wet1': 'ClearVoice → Sidon → DF3 · 100%  [V11 cascade]',
    'legacy_v8_exact': 'Historical v8/v9 · complete recipe',
}
POLICY_LABELS = {
    'per_item_best': 'Best: native / full cascade  [800]',
    'per_item_best_ge3_5': 'Best: native + ≥3.5 filter',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True,
                        help='Matched original/cascade per-item score CSV')
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--normalized-selection', type=Path,
                        help='Normalized-original/cascade selector CSV; adds three rows')
    parser.add_argument('--brute-force-selection', type=Path,
                        help='Per-item maximum across all 17 scored variants; adds two rows')
    parser.add_argument('--mfa-metrics', type=Path, help='Sanitized MFA per-file CSV; adds the boundary-trim variant')
    args = parser.parse_args()
    with args.input.open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if r['source_id'] == 'all' and r['metric'] == 'sigmos_overall']
    if len(rows) != len(LABELS) or {r['profile'] for r in rows} != set(LABELS):
        raise ValueError('Expected one overall SigMOS row for each V10 processing variant')
    for row in rows:
        row['count'] = int(row['count'])
        row['median'] = float(row['median'])
        if row['count'] != 800 or int(row['expected_count']) != 800:
            raise ValueError('Each variant must cover the full 800-item panel')
        if not math.isfinite(row['median']) or not 1 <= row['median'] <= 5:
            raise ValueError('Invalid median SigMOS')
    baseline = next(r['median'] for r in rows if r['profile'] == 'original')
    with args.selection.open() as stream:
        selected = list(csv.DictReader(stream))
    from collections import Counter
    if (len(selected) != 800 or len({r['sample_id'] for r in selected}) != 800
            or sorted(Counter(r['source_id'] for r in selected).values()) != [100] * 8):
        raise ValueError('Selection must cover the same balanced 800-item panel')
    before = [float(r['original_sigmos_overall']) for r in selected]
    after = [float(r['processed_sigmos_overall']) for r in selected]
    best = [max(a, b) for a, b in zip(before, after)]
    if not all(math.isfinite(x) and 1 <= x <= 5 for x in before + after):
        raise ValueError('Invalid per-item SigMOS')
    cascade = next(r['median'] for r in rows if r['profile'] == 'clearervoice_sidon_deepfilternet3_wet1')
    if not math.isclose(median(before), baseline, abs_tol=1e-10) or not math.isclose(median(after), cascade, abs_tol=1e-10):
        raise ValueError('Per-item scores do not match the plotted original/cascade medians')
    retained = [x for x in best if x >= 3.5]
    if not retained:
        raise ValueError('The filter retained no recordings')
    labels = {**LABELS, **POLICY_LABELS}
    labels['per_item_best_ge3_5'] = f'Best: native + ≥3.5  [{len(retained)} retained]'
    for profile, values in [('per_item_best', best), ('per_item_best_ge3_5', retained)]:
        rows.append({'profile': profile, 'count': len(values), 'median': median(values)})
    new_profiles = set()
    normalized_retained = []
    if args.normalized_selection:
        with args.normalized_selection.open() as stream:
            normalized_rows = list(csv.DictReader(stream))
        old_by_id = {r['sample_id']: r for r in selected}
        if (len(normalized_rows) != 800 or len({r['sample_id'] for r in normalized_rows}) != 800
                or {r['sample_id'] for r in normalized_rows} != set(old_by_id)):
            raise ValueError('Normalized selection must match the same 800 recordings')
        normalized_scores, normalized_best = [], []
        for row in normalized_rows:
            old = old_by_id[row['sample_id']]
            a, b = float(row['normalized_sigmos_overall']), float(row['processed_sigmos_overall'])
            if not all(math.isfinite(x) and 1 <= x <= 5 for x in (a, b)):
                raise ValueError('Invalid normalized selection score')
            if row['source_id'] != old['source_id'] or b != float(old['processed_sigmos_overall']):
                raise ValueError('Cascade scores or sources differ between comparisons')
            if float(row['selected_sigmos_overall']) != max(a, b):
                raise ValueError('Selected score is not the per-item maximum')
            normalized_scores.append(a)
            normalized_best.append(max(a, b))
        normalized_retained = [x for x in normalized_best if x >= 3.5]
        if not normalized_retained:
            raise ValueError('Normalized selection retains no recordings')
        for profile, label, values in [
            ('normalized_original', 'Normalized original', normalized_scores),
            ('normalized_per_item_best', 'Best: normalized / full cascade  [800]', normalized_best),
            ('normalized_per_item_best_ge3_5', f'Best: normalized + ≥3.5  [{len(normalized_retained)} retained]', normalized_retained),
        ]:
            labels[profile] = label
            new_profiles.add(profile)
            rows.append({'profile': profile, 'count': len(values), 'median': median(values)})
    brute_profiles = set()
    if args.brute_force_selection:
        with args.brute_force_selection.open() as stream:
            brute_rows = list(csv.DictReader(stream))
        old_by_id = {r['sample_id']: r for r in selected}
        if (len(brute_rows) != 800 or len({r['sample_id'] for r in brute_rows}) != 800
                or {r['sample_id'] for r in brute_rows} != set(old_by_id)):
            raise ValueError('Brute-force selection must match the 800-recording panel')
        variants = set(LABELS) | {'normalized_original'}
        for row in brute_rows:
            scores = {name: float(row[name + '_sigmos_overall']) for name in variants}
            if not all(math.isfinite(x) and 1 <= x <= 5 for x in scores.values()):
                raise ValueError('Invalid brute-force score')
            if (float(row['selected_sigmos_overall']) != max(scores.values())
                    or scores[row['selected_variant']] != max(scores.values())):
                raise ValueError('Brute-force selection is not the maximum')
            old = old_by_id[row['sample_id']]
            if row['source_id'] != old['source_id'] or scores['original'] != float(old['original_sigmos_overall']):
                raise ValueError('Brute-force reference mismatch')
        for variant in variants:
            expected = next((r['median'] for r in rows if r['profile'] == variant), None)
            if expected is not None and not math.isclose(median(float(r[variant + '_sigmos_overall']) for r in brute_rows), expected, abs_tol=1e-10):
                raise ValueError('Brute-force scores do not match plotted variant')
        brute_best = [float(r['selected_sigmos_overall']) for r in brute_rows]
        brute_retained = [x for x in brute_best if x >= 3.5]
        for profile, label, values in [
            ('brute_force_best', 'Brute-force best · 17 variants  [800]', brute_best),
            ('brute_force_best_ge3_5', f'Brute-force best + ≥3.5  [{len(brute_retained)} retained]', brute_retained),
        ]:
            labels[profile] = label
            brute_profiles.add(profile)
            rows.append({'profile': profile, 'count': len(values), 'median': median(values)})
    mfa_profiles = set()
    if args.mfa_metrics:
        with args.mfa_metrics.open() as stream:
            mfa_rows = list(csv.DictReader(stream))
        if (len(mfa_rows) != 800 or len({r['sample_id'] for r in mfa_rows}) != 800
                or {r['sample_id'] for r in mfa_rows} != {r['sample_id'] for r in selected}):
            raise ValueError('MFA results must match the same 800-recording panel')
        mfa_scores = [float(r['sigmos_overall']) for r in mfa_rows]
        if not all(math.isfinite(x) and 1 <= x <= 5 for x in mfa_scores):
            raise ValueError('Invalid MFA SigMOS')
        mfa_review = sum(r['status'] != 'mfa_aligned' for r in mfa_rows)
        labels['mfa_trim'] = 'MFA boundary trim / normalize · 100 ms'
        mfa_profiles.add('mfa_trim')
        rows.append({'profile': 'mfa_trim', 'count': len(mfa_scores), 'median': median(mfa_scores)})
    rows.sort(key=lambda r: r['median'], reverse=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'profile', 'label', 'count', 'median_sigmos_overall', 'difference_from_original'
        ], lineterminator='\n')
        writer.writeheader()
        writer.writerows({
            'profile': r['profile'], 'label': labels[r['profile']], 'count': r['count'],
            'median_sigmos_overall': r['median'],
            'difference_from_original': r['median'] - baseline,
        } for r in rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11,
        'figure.facecolor': '#fffff8', 'axes.facecolor': '#fffff8',
        'savefig.facecolor': '#fffff8', 'text.color': '#111111',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.grid': False,
        'axes.edgecolor': '#666666', 'axes.linewidth': .6,
        'xtick.color': '#555555', 'ytick.color': '#333333',
    })
    fig, ax = plt.subplots(figsize=(14, 10 if brute_profiles else 9 if new_profiles else 8))
    for index, row in enumerate(rows):
        selected = row['profile'] in (mfa_profiles or brute_profiles or new_profiles or POLICY_LABELS)
        original = row['profile'] == 'original'
        color = '#285b80' if selected else '#333333' if original else '#666666'
        marker = 's' if row['profile'].endswith('_ge3_5') else 'D' if original else 'o'
        ax.plot(row['median'], index, marker, color=color, markersize=6)
        ax.annotate(f"{row['median']:.3f}", (row['median'], index),
                    xytext=(9, 0), textcoords='offset points', va='center', color=color,
                    bbox={'facecolor': '#fffff8', 'edgecolor': 'none', 'pad': .5}, zorder=3)
    ax.vlines(baseline, -.5, len(rows) - .5, color='#666666', linewidth=.8, linestyle='--')
    ax.text(baseline, -1, f'Original {baseline:.3f}', ha='center', fontsize=10)
    ax.set_yticks(range(len(rows)), [labels[r['profile']] for r in rows])
    for label, row in zip(ax.get_yticklabels(), rows):
        if row['profile'] in (mfa_profiles or brute_profiles or new_profiles or POLICY_LABELS):
            label.set_color('#285b80')
    ax.tick_params(axis='y', length=0, pad=12)
    ax.tick_params(axis='x', direction='in', length=3)
    low, high = rows[-1]['median'], rows[0]['median']
    ax.set_xlim(low - .02, high + .065)
    ax.set_xticks([x / 10 for x in range(10, 51, 2) if low - .02 <= x / 10 <= high + .065])
    ax.set_ylim(len(rows) - .3, -1.6)
    ax.spines['bottom'].set_bounds(low, high)
    ax.set_xlabel('Median overall SigMOS · higher is better (1–5 scale; zoomed axis)', labelpad=12)
    title = (f'Median SigMOS: {median(normalized_scores):.3f} normalized → {median(normalized_best):.3f} best → '
             f'{median(normalized_retained):.3f} filtered' if new_profiles else
             f'Median SigMOS: {baseline:.3f} original → {median(best):.3f} best → {median(retained):.3f} filtered')
    if brute_profiles:
        title = f'Brute-force selection: {median(brute_best):.3f} median SigMOS → {median(brute_retained):.3f} after ≥3.5'
    if mfa_profiles:
        energy = next(r['median'] for r in rows if r['profile'] == 'identity_wet1')
        title = f'MFA boundary trim: {median(mfa_scores):.3f} median SigMOS vs. {energy:.3f} energy trim'
    fig.text(.04, .95, title, fontsize=19)
    fig.text(.04, .91, 'Fixed pilot · 100 recordings per source · 800 recordings per row except labeled filtered subsets', fontsize=11)
    fig.text(.04, .065, 'DF3 = DeepFilterNet3. Percentages indicate enhanced-audio blend; remainder is dry audio.', fontsize=10)
    foot = (f'≥3.5 retains {len(normalized_retained)}/800 with normalized originals; {len(retained)}/800 with native originals. '
            'Filtered medians describe smaller populations.' if new_profiles else
            f'Filtered row retains {len(retained)}/800 recordings ({len(retained) / 8:.1f}%). Its median describes a smaller population.')
    if brute_profiles:
        foot = f'Brute-force includes native and normalized originals; ≥3.5 retains {len(brute_retained)}/800 ({len(brute_retained) / 8:.1f}%). Filtered medians describe smaller populations.'
    if mfa_profiles:
        foot = f'MFA: {800 - mfa_review} alignments used; {mfa_review} review cases preserve untrimmed audio. Brute-force remains the earlier 17-variant comparison.'
    fig.text(.04, .038, foot, fontsize=10)
    fig.subplots_adjust(left=.39, right=.98, top=.86, bottom=.17)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output_prefix.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
