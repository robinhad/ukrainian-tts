#!/usr/bin/env python3
"""Plot median overall SigMOS from the source-balanced processing report."""
import argparse
import csv
import math
from pathlib import Path

LABELS = {
    'original': 'Native original',
    'identity_wet1': 'Base mono / 24 kHz / trim',
    'deepfilternet3_wet0.5': 'DF3 · 50%',
    'deepfilternet3_wet1': 'DF3 · 100%',
    'deepfilternet3_peakminus3_wet1': 'DF3 · −3 dBFS input · 100%',
    'deepfilternet3_peakminus3_wet0.75': 'DF3 · −3 dBFS input · 75%  [selected]',
    'clearervoice_wet0.5': 'ClearVoice · 50%',
    'clearervoice_wet0.75': 'ClearVoice · 75%',
    'clearervoice_wet1': 'ClearVoice · 100%',
    'sidon_wet0.5': 'Sidon · 50%',
    'sidon_wet1': 'Sidon · 100%',
    'clearervoice_sidon_wet0.5': 'ClearVoice → Sidon · 50%',
    'clearervoice_sidon_wet1': 'ClearVoice → Sidon · 100%',
    'clearervoice_sidon_deepfilternet3_wet0.5': 'ClearVoice → Sidon → DF3 · 50%',
    'clearervoice_sidon_deepfilternet3_wet1': 'ClearVoice → Sidon → DF3 · 100%',
    'legacy_v8_exact': 'Historical v8/v9 · complete recipe',
}
SELECTED = 'deepfilternet3_peakminus3_wet0.75'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
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
    rows.sort(key=lambda r: r['median'], reverse=True)
    baseline = next(r['median'] for r in rows if r['profile'] == 'original')
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'profile', 'label', 'count', 'median_sigmos_overall', 'difference_from_original'
        ], lineterminator='\n')
        writer.writeheader()
        writer.writerows({
            'profile': r['profile'], 'label': LABELS[r['profile']], 'count': r['count'],
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
    fig, ax = plt.subplots(figsize=(12, 8))
    for index, row in enumerate(rows):
        selected = row['profile'] == SELECTED
        original = row['profile'] == 'original'
        color = '#285b80' if selected else '#333333' if original else '#666666'
        marker = 's' if selected else 'D' if original else 'o'
        ax.plot(row['median'], index, marker, color=color, markersize=6)
        ax.annotate(f"{row['median']:.3f}", (row['median'], index),
                    xytext=(9, 0), textcoords='offset points', va='center', color=color)
    ax.vlines(baseline, -.5, len(rows) - .5, color='#666666', linewidth=.8, linestyle='--')
    ax.text(baseline, -1, f'Original {baseline:.3f}', ha='center', fontsize=10)
    ax.set_yticks(range(len(rows)), [LABELS[r['profile']] for r in rows])
    for label, row in zip(ax.get_yticklabels(), rows):
        if row['profile'] == SELECTED:
            label.set_color('#285b80')
    ax.tick_params(axis='y', length=0, pad=12)
    ax.tick_params(axis='x', direction='in', length=3)
    low, high = rows[-1]['median'], rows[0]['median']
    ax.set_xlim(low - .02, high + .065)
    ax.set_xticks([2.95, 3.00, 3.05, 3.10, 3.15, 3.20])
    ax.set_ylim(len(rows) - .3, -1.6)
    ax.spines['bottom'].set_bounds(low, high)
    ax.set_xlabel('Median overall SigMOS · higher is better (1–5 scale; zoomed axis)', labelpad=12)
    fig.text(.04, .95, f"Full cascade leads median overall SigMOS at {high:.3f}", fontsize=19)
    fig.text(.04, .91, 'Same 800 non-VOA recordings per variant · 100 per source · whole-file scores', fontsize=11)
    fig.text(.04, .065, 'DF3 = DeepFilterNet3. Percentages indicate enhanced-audio blend; remainder is dry audio.', fontsize=10)
    fig.text(.04, .038, 'Selected recipe also balances content and speaker preservation. Smaller refinement panels are excluded.', fontsize=10)
    fig.subplots_adjust(left=.39, right=.98, top=.86, bottom=.17)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output_prefix.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
