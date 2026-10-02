#!/usr/bin/env python3
"""Visualize matched per-recording SigMOS changes, exporting only score data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from report_source_quality_distribution import tables

SOURCES = {
    'common_voice_available_uk': 'Common Voice', 'fleurs_uk': 'FLEURS',
    'opentts_lada': 'Lada', 'opentts_mykyta': 'Mykyta',
    'opentts_tetiana': 'Tetiana', 'tg_voices_uk': 'TG voices',
    'ua_ser': 'UA-SER', 'ukr_dialects': 'Ukrainian dialects',
}


def read_panel(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {(r['source_id'], r['utterance_id']): r for r in rows}
    counts = Counter(r['source_id'] for r in rows)
    if len(rows) != 800 or len(keyed) != 800 or counts != dict.fromkeys(SOURCES, 100):
        raise ValueError('Expected 800 unique recordings, 100 per known source')
    values = np.array([r['sigmos_overall'] for r in rows])
    if not np.isfinite(values).all() or np.any((values < 1) | (values > 5)):
        raise ValueError('Scores must be finite and in [1, 5]')
    return keyed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    original, processed = read_panel(args.original), read_panel(args.processed)
    if original.keys() != processed.keys():
        raise ValueError('Panels must contain the same recordings')
    records = []
    for key in sorted(original):
        before, after = original[key]['sigmos_overall'], processed[key]['sigmos_overall']
        records.append({
            'sample_id': hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16],
            'source_id': key[0], 'original_sigmos_overall': before,
            'processed_sigmos_overall': after, 'change': after - before,
        })
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Sample identifier collision')
    tables(args.output_prefix, records)
    summaries = []
    for source in ['all', *SOURCES]:
        values = np.array([r['change'] for r in records if source == 'all' or r['source_id'] == source])
        summaries.append({
            'source_id': source, 'count': len(values),
            'improved': int((values > 0).sum()), 'worsened': int((values < 0).sum()),
            'unchanged': int((values == 0).sum()), 'mean_change': float(values.mean()),
            'median_change': float(np.median(values)),
            'p05_change': float(np.quantile(values, .05)),
            'p95_change': float(np.quantile(values, .95)),
        })
    tables(args.output_prefix.with_name(args.output_prefix.name + '_summary'), summaries)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11,
        'figure.facecolor': '#fffff8', 'axes.facecolor': '#fffff8',
        'savefig.facecolor': '#fffff8', 'text.color': '#111111',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'axes.edgecolor': '#666666', 'axes.linewidth': .6,
        'xtick.color': '#555555', 'ytick.color': '#333333',
    })
    before = np.array([r['original_sigmos_overall'] for r in records])
    after = np.array([r['processed_sigmos_overall'] for r in records])
    delta = after - before
    colors = np.where(delta > 0, '#285b80', '#666666')
    fig = plt.figure(figsize=(14, 8.5))
    paired = fig.add_axes([.07, .2, .38, .62])
    by_source = fig.add_axes([.63, .2, .32, .62])
    paired.plot([1, 5], [1, 5], '--', linewidth=1, color='#555555')
    paired.scatter(before, after, c=colors, s=15, alpha=.75, linewidths=0)
    paired.set_aspect('equal', adjustable='box')
    paired.set_xlim(1, 5)
    paired.set_ylim(1, 5)
    paired.set_xticks([1, 2, 3, 4, 5])
    paired.set_yticks([1, 2, 3, 4, 5])
    paired.set_xlabel('Original overall SigMOS', labelpad=10)
    paired.set_ylabel('Processed overall SigMOS', labelpad=10)
    paired.set_title('Each dot pairs one recording', loc='left', fontsize=14, pad=18)
    paired.text(.04, .92, f"{summaries[0]['improved']} higher scores", transform=paired.transAxes, color='#285b80')
    paired.text(.55, .1, f"{summaries[0]['worsened']} lower scores", transform=paired.transAxes, color='#555555')
    paired.text(4.07, 4.46, 'No change', rotation=45, fontsize=10, color='#555555')
    paired.spines['bottom'].set_bounds(before.min(), before.max())
    paired.spines['left'].set_bounds(after.min(), after.max())

    order = sorted(summaries[1:], key=lambda r: r['median_change'], reverse=True)
    rng = np.random.default_rng(777)
    by_source.axvline(0, linestyle='--', linewidth=1, color='#555555')
    for index, summary in enumerate(order):
        values = np.array([r['change'] for r in records if r['source_id'] == summary['source_id']])
        jitter = rng.uniform(-.23, .23, len(values))
        by_source.scatter(values, index + jitter, s=14, linewidths=0, alpha=.75,
                          c=np.where(values > 0, '#285b80', '#666666'))
        by_source.plot(summary['median_change'], index, '|', markersize=20,
                       markeredgewidth=2, color='#111111')
    by_source.set_yticks(range(len(order)), [SOURCES[r['source_id']] for r in order])
    by_source.set_ylim(len(order) - .4, -.6)
    by_source.set_xlim(-1.5, 2.3)
    by_source.set_xticks([-1, 0, 1, 2], ['−1', '0', '+1', '+2'])
    by_source.set_xlabel('Individual change: processed − original', labelpad=10)
    by_source.set_title('Changes by source · black ticks mark medians', loc='left', fontsize=13, pad=18)
    by_source.spines['left'].set_visible(False)
    by_source.spines['bottom'].set_bounds(delta.min(), delta.max())
    by_source.tick_params(axis='y', length=0, pad=8)
    for ax in [paired, by_source]:
        ax.tick_params(axis='x', direction='in', length=3)
    overall = summaries[0]
    fig.text(.07, .94, f"{overall['improved']} of 800 recordings gain SigMOS; {overall['worsened']} lose score", fontsize=21)
    fig.text(.07, .895, 'Original → ClearVoice → Sidon → DeepFilterNet3, full strength · same 800 recordings, 100 per source', fontsize=12)
    fig.text(.07, .11, f"Median individual change: {overall['median_change']:+.3f} · mean: {overall['mean_change']:+.3f} · "
             f"middle 90%: {overall['p05_change']:+.3f} to {overall['p95_change']:+.3f}", fontsize=12)
    fig.text(.07, .065, 'Whole-file model estimates; any nonzero change is counted, without a calibrated meaningful-change threshold.', fontsize=10)
    fig.text(.07, .035, 'Vertical jitter separates dots within each source. Equal-source sample, not full-corpus proportions.', fontsize=10)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output_prefix.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
