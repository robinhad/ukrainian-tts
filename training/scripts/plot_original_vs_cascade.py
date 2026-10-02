#!/usr/bin/env python3
"""Compare overall SigMOS distributions on a matched original/cascade panel."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from report_source_quality_distribution import tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    groups = []
    panel = None
    for label, path in [('Original', args.original), ('Full cascade', args.processed)]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        identifiers = {(r['utterance_id'], r['source_id']) for r in rows}
        sources = Counter(r['source_id'] for r in rows)
        if len(rows) != 800 or len(identifiers) != 800 or sorted(sources.values()) != [100] * 8:
            raise ValueError('Expected 800 unique recordings, 100 per source')
        if panel is not None and identifiers != panel:
            raise ValueError('The groups must contain the same recordings')
        panel = identifiers
        values = np.asarray([r['sigmos_overall'] for r in rows], dtype=float)
        if not np.isfinite(values).all() or np.any((values < 1) | (values > 5)):
            raise ValueError('Expected finite overall SigMOS scores in [1, 5]')
        groups.append((label, values))

    edges = np.linspace(1, 5, 21)
    bins, summaries = [], []
    for label, values in groups:
        counts, _ = np.histogram(values, bins=edges)
        assert counts.sum() == len(values)
        bins.extend({'group': label, 'lower_inclusive': float(lo), 'upper': float(hi),
                     'upper_inclusive': index == len(counts) - 1,
                     'count': int(count), 'percent': float(100 * count / len(values))}
                    for index, (lo, hi, count) in enumerate(zip(edges[:-1], edges[1:], counts)))
        summaries.append({'group': label, 'count': len(values), 'mean': float(values.mean()),
                          'median': float(np.median(values)),
                          'p05': float(np.quantile(values, .05)),
                          'p95': float(np.quantile(values, .95)),
                          'count_ge_3_5': int((values >= 3.5).sum()),
                          'count_ge_4': int((values >= 4).sum())})
    tables(args.output_prefix, summaries)
    tables(args.output_prefix.with_name(args.output_prefix.name + '_bins'), bins)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 12,
        'figure.facecolor': '#fffff8', 'axes.facecolor': '#fffff8',
        'savefig.facecolor': '#fffff8', 'text.color': '#111111',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'axes.edgecolor': '#666666', 'axes.linewidth': .6,
        'xtick.color': '#555555', 'ytick.color': '#555555',
    })
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (label, values), summary, color in zip(
            axes, groups, summaries, ['#666666', '#285b80']):
        counts, _ = np.histogram(values, bins=edges)
        ax.bar(edges[:-1], counts * 100 / len(values), width=.19, align='edge', color=color)
        median = summary['median']
        ax.vlines(median, 0, 20, color='#333333', linewidth=1, linestyle=':')
        ax.text(median + .04, 21, f'Median {median:.3f}', color=color, fontsize=11)
        for threshold in [3.5, 4.0]:
            ax.vlines(threshold, 0, 19.5, color='#555555', linewidth=.7, linestyle='--')
            ax.text(threshold + .04, 18.5, f'{threshold:.1f}', fontsize=10)
        ax.text(.015, .88, label, transform=ax.transAxes, fontsize=16, color=color)
        ax.text(.015, .73,
                f"≥3.5: {summary['count_ge_3_5'] / 8:.1f}%\n"
                f"≥4.0: {summary['count_ge_4'] / 8:.1f}%",
                transform=ax.transAxes, va='top', fontsize=12, linespacing=1.5)
        ax.set_ylim(0, 24)
        ax.set_yticks([0, 5, 10, 15, 20], ['0%', '5%', '10%', '15%', '20%'])
        ax.set_ylabel('Recordings per score bin')
        ax.spines['left'].set_bounds(0, 20)
        ax.spines['bottom'].set_bounds(1.2, 4.8)
        ax.tick_params(direction='in', length=3)
    axes[-1].set_xlim(1, 5)
    axes[-1].set_xticks([1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    axes[-1].set_xlabel('Overall SigMOS · higher is better (1–5 scale)', labelpad=10)
    fig.text(.08, .95, 'The full cascade raises the median but reduces the ≥4.0 share', fontsize=19)
    fig.text(.08, .905, 'Original vs. ClearVoice → Sidon → DeepFilterNet3, full strength', fontsize=12)
    fig.text(.08, .87, 'Same 800 non-VOA recordings · 100 per source · identical 0.2-point bins and axes', fontsize=11)
    fig.text(.08, .035, 'Whole-file model estimates. Equal-source sample; not weighted by full-corpus source proportions.', fontsize=10)
    fig.subplots_adjust(left=.1, right=.97, top=.82, bottom=.14, hspace=.18)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output_prefix.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
