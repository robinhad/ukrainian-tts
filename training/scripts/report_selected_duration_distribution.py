#!/usr/bin/env python3
"""Compare native and selected audio durations on the completed balanced panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from plot_individual_sigmos_changes import read_panel
from report_source_quality_distribution import tables


def summary(group, source, values):
    x = np.asarray(values, dtype=float)
    return {'group': group, 'source_id': source, 'count': len(x),
            'hours': float(x.sum() / 3600), 'mean_seconds': float(x.mean()),
            **{name + '_seconds': float(np.quantile(x, q)) for name, q in
               [('minimum', 0), ('p05', .05), ('p25', .25), ('median', .5),
                ('p75', .75), ('p95', .95), ('maximum', 1)]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=3.5)
    args = parser.parse_args()
    original, processed = read_panel(args.original), read_panel(args.processed)
    if original.keys() != processed.keys():
        raise ValueError('Expected identical original and processed panels')
    records = []
    for key in sorted(original):
        before, candidate = original[key], processed[key]
        selected = candidate if candidate['sigmos_overall'] > before['sigmos_overall'] else before
        records.append({
            'sample_id': hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16],
            'source_id': key[0], 'original_seconds': before['duration_seconds'],
            'selected_seconds': selected['duration_seconds'],
            'retained': selected['sigmos_overall'] >= args.threshold,
            'selected_variant': 'processed' if selected is candidate else 'original',
        })
    if not all(np.isfinite(r[k]) and r[k] > 0 for r in records
               for k in ('original_seconds', 'selected_seconds')):
        raise ValueError('Expected positive finite durations')
    sources = ['all', *sorted({r['source_id'] for r in records})]
    summaries = []
    for source in sources:
        rows = [r for r in records if source == 'all' or r['source_id'] == source]
        kept = [r for r in rows if r['retained']]
        for group, values in [
            ('before_all', [r['original_seconds'] for r in rows]),
            ('retained_before_processing', [r['original_seconds'] for r in kept]),
            ('after_selection', [r['selected_seconds'] for r in kept]),
        ]:
            if values:
                summaries.append(summary(group, source, values))
    prefix = args.output_prefix
    tables(prefix.with_name(prefix.name + '_items'), records)
    tables(prefix, summaries)
    groups = [
        ('Before · native originals', np.array([r['original_seconds'] for r in records])),
        ('After · selected score ≥' + str(args.threshold),
         np.array([r['selected_seconds'] for r in records if r['retained']])),
    ]
    edges = np.array([0, 3, 5, 10, 15, 20, np.inf])
    bins = []
    for label, values in groups:
        counts = np.histogram(values, bins=edges)[0]
        assert counts.sum() == len(values)
        for low, high, count in zip(edges[:-1], edges[1:], counts):
            bins.append({'group': label, 'lower_seconds_inclusive': float(low),
                         'upper_seconds_exclusive': float(high) if np.isfinite(high) else None,
                         'count': int(count), 'percent': float(100 * count / len(values))})
    tables(prefix.with_name(prefix.name + '_bins'), bins)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11,
        'figure.facecolor': '#fffff8', 'axes.facecolor': '#fffff8',
        'savefig.facecolor': '#fffff8', 'text.color': '#111111',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'axes.edgecolor': '#666666', 'axes.linewidth': .6,
    })
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    histogram_edges = np.arange(0, np.ceil(max(x.max() for _, x in groups)) + 1)
    heights = [np.histogram(x, histogram_edges)[0] * 100 / len(x) for _, x in groups]
    ymax = np.ceil(max(h.max() for h in heights) / 5) * 5 + 5
    for ax, (label, values), height, color in zip(axes, groups, heights, ['#666666', '#285b80']):
        ax.bar(histogram_edges[:-1], height, width=.9, align='edge', color=color)
        median = np.median(values)
        ax.vlines(median, 0, ymax * .76, colors='#333333', linestyles=':', linewidth=1)
        ax.text(median + .2, ymax * .79, f'Median {median:.2f} s', color=color)
        ax.text(.62, .85, label, transform=ax.transAxes, fontsize=15, color=color)
        ax.text(.62, .65, f'{len(values):,} recordings\nMean {values.mean():.2f} s',
                transform=ax.transAxes, linespacing=1.6)
        ax.set_ylim(0, ymax)
        ticks = np.arange(0, ymax, 5)
        ax.set_yticks(ticks, [f'{t:g}%' for t in ticks])
        ax.set_ylabel('Share of recordings')
        ax.spines['left'].set_bounds(0, ticks[-1])
        ax.spines['bottom'].set_bounds(2, histogram_edges[-1])
        ax.tick_params(direction='in', length=3)
    axes[-1].set_xlim(1.5, histogram_edges[-1])
    axes[-1].set_xticks(np.arange(2, histogram_edges[-1] + 1, 2))
    axes[-1].set_xlabel('Recording duration (seconds) · identical one-second bins')
    fig.text(.09, .95, f'Median duration: {np.median(groups[0][1]):.2f} → '
             f'{np.median(groups[1][1]):.2f} seconds after selection', fontsize=19)
    fig.text(.09, .90, 'Completed 800-recording pilot · 100 per source before filtering', fontsize=12)
    fig.text(.09, .055, 'Choose full-cascade audio only if SigMOS improves; otherwise keep native original. '
             'Then retain ≥3.5.', fontsize=10)
    fig.text(.09, .026, 'Sample results, not a full-corpus selection. '
             'Equal source sampling differs from full-corpus proportions.', fontsize=10)
    fig.subplots_adjust(left=.1, right=.97, top=.84, bottom=.14, hspace=.16)
    for extension in ('png', 'pdf'):
        fig.savefig(prefix.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
