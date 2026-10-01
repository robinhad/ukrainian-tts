#!/usr/bin/env python3
"""Summarize existing original-audio quality measurements without rerunning models."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

METRICS = {
    'sigmos_overall': 'SigMOS overall',
    'sigmos_speech': 'SigMOS speech',
    'sigmos_noise': 'SigMOS noise',
    'sigmos_coloration': 'SigMOS coloration',
    'sigmos_discontinuity': 'SigMOS discontinuity',
    'audiobox_pq': 'Audiobox PQ',
}


def tables(prefix, records):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    prefix.with_suffix('.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in records)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows or any(row.get('label') != 'original' for row in rows):
        raise ValueError('Input must contain original-audio measurements only')
    sources = sorted({row['source_id'] for row in rows})
    records = []
    for source in ['all', *sources]:
        selected = rows if source == 'all' else [r for r in rows if r['source_id'] == source]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Missing/nonfinite measurements: {source}, {metric}')
            quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
            records.append({'source_id': source, 'metric': metric, 'count': len(values),
                            'mean': float(values.mean()),
                            **dict(zip(['min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'],
                                       map(float, quantiles)))})
    tables(args.output_prefix, records)
    bins = []
    for metric in METRICS:
        edges = np.arange(1, 5.01, .5) if metric.startswith('sigmos_') else np.arange(1, 10.01, 1)
        values = np.asarray([r[metric] for r in rows])
        counts, _ = np.histogram(values, bins=edges)
        if counts.sum() != len(values):
            raise ValueError(f'Values outside the declared score range: {metric}')
        bins.extend({'metric': metric, 'lower_inclusive': float(lo), 'upper': float(hi),
                     'upper_inclusive': i == len(counts) - 1, 'count': int(count),
                     'percent': float(100 * count / len(values))}
                    for i, (lo, hi, count) in enumerate(zip(edges[:-1], edges[1:], counts)))
    tables(args.output_prefix.with_name(args.output_prefix.name + '_bins'), bins)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 11, 'figure.facecolor': '#fffff8',
                         'axes.facecolor': '#fffff8', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': False,
                         'axes.edgecolor': '#666666', 'axes.linewidth': .6,
                         'text.color': '#111111', 'xtick.color': '#555555',
                         'ytick.color': '#555555', 'savefig.facecolor': '#fffff8'})
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        values = np.sort([row[metric] for row in rows])
        cumulative = np.arange(1, len(values) + 1) * 100 / len(values)
        median = float(np.median(values))
        ax.step(values, cumulative, where='post', color='#666666', linewidth=1.7)
        ax.plot([values[0], median], [50, 50], '--', color='#555555', linewidth=.8)
        ax.plot(median, 50, 'o', color='#285b80', markersize=4)
        ax.text(.04, .9, f'Median {median:.2f}', transform=ax.transAxes, color='#285b80')
        ax.set_title(label, loc='left', fontsize=13)
        ax.set_xlim((1, 5) if metric.startswith('sigmos_') else (1, 10))
        ax.set_ylim(0, 103)
        ax.set_xticks([1, 2, 3, 4, 5] if metric.startswith('sigmos_') else [1, 3, 5, 7, 9])
        ax.set_yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        ax.spines['bottom'].set_bounds(values.min(), values.max())
        ax.spines['left'].set_bounds(0, 100)
        ax.tick_params(direction='in', length=3)
        ax.set_xlabel('Predicted score (higher is better)')
    axes[0, 0].set_ylabel('Recordings at or below score')
    axes[1, 0].set_ylabel('Recordings at or below score')
    overall = next(r for r in records if r['source_id'] == 'all' and r['metric'] == 'sigmos_overall')
    fig.suptitle(f"Original-audio overall MOS has a median of {overall['median']:.2f}",
                 x=.07, ha='left', fontsize=18)
    counts = [sum(r['source_id'] == s for r in rows) for s in sources]
    sampling = f'{counts[0]} per source' if len(set(counts)) == 1 else 'unequal source counts'
    fig.text(.07, .922, f'{len(rows):,} original non-VOA recordings; {len(sources)} sources; {sampling}. '
             'Whole-file model estimates, before processing.', fontsize=10)
    fig.text(.07, .03, 'Empirical cumulative distributions. SigMOS uses 1–5; Audiobox PQ uses 1–10. '
             'The sample does not represent corpus source proportions.', fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, bottom=.13, top=.84, hspace=.48, wspace=.27)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output_prefix.with_suffix('.' + extension), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
