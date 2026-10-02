"""Compare SigMOS improvement counts with brute-force winner counts."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from plot_processing_sigmos import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=Path('training/reports/quality_bruteforce_selection.jsonl'))
    parser.add_argument('--output-prefix', type=Path, default=Path('training/reports/quality_processing_improvement_counts'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.selection.read_text().splitlines()]
    if len(rows) != 800 or len({r['sample_id'] for r in rows}) != 800 or sorted(Counter(r['source_id'] for r in rows).values()) != [100] * 8:
        raise ValueError('Expected the matched 800-item source-balanced panel')
    labels = {**LABELS, 'normalized_original': 'Normalized original'}
    labels = {k: v.split('  [')[0] for k, v in labels.items()}
    for row in rows:
        scores = [row[name + '_sigmos_overall'] for name in labels]
        if (not all(math.isfinite(x) and 1 <= x <= 5 for x in scores)
                or row['selected_sigmos_overall'] != max(scores)
                or row[row['selected_variant'] + '_sigmos_overall'] != max(scores)
                or row['retained'] != (max(scores) >= 3.5)):
            raise ValueError('Invalid scores, winner, or cutoff decision')
    counts = []
    for name in labels:
        counts.append({'profile': name, 'label': labels[name], 'evaluated': len(rows),
                       'improved_vs_native': sum(r[name + '_sigmos_overall'] > r['original_sigmos_overall'] for r in rows),
                       'equal_to_native': sum(r[name + '_sigmos_overall'] == r['original_sigmos_overall'] for r in rows),
                       'worse_vs_native': sum(r[name + '_sigmos_overall'] < r['original_sigmos_overall'] for r in rows),
                       'improved_vs_normalized': sum(r[name + '_sigmos_overall'] > r['normalized_original_sigmos_overall'] for r in rows),
                       'selected_all': sum(r['selected_variant'] == name for r in rows),
                       'selected_ge3_5': sum(r['selected_variant'] == name and r['retained'] for r in rows)})
    counts.sort(key=lambda r: (-r['improved_vs_native'], r['profile']))
    assert sum(r['selected_all'] for r in counts) == 800
    retained = sum(r['retained'] for r in rows)
    assert sum(r['selected_ge3_5'] for r in counts) == retained
    from report_source_quality_distribution import tables
    tables(args.output_prefix, counts)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11,
                         'figure.facecolor': '#fffff8', 'axes.facecolor': '#fffff8',
                         'savefig.facecolor': '#fffff8', 'text.color': '#111111',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.grid': False,
                         'axes.edgecolor': '#666666', 'axes.linewidth': .6})
    fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
    winner_maximum = math.ceil(max(r['selected_all'] for r in counts) / 20) * 20
    specs = [('improved_vs_native', 'Higher SigMOS than native original', 800),
             ('selected_all', 'Best variant · all 800 recordings', winner_maximum),
             ('selected_ge3_5', f'Best variant · {retained} retained (≥3.5)', winner_maximum)]
    for ax, (metric, title, maximum) in zip(axes, specs):
        values = [r[metric] for r in counts]
        colors = ['#285b80' if v == max(values) else '#777777' for v in values]
        ax.barh(range(len(counts)), values, height=.55, color=colors)
        for y, value in enumerate(values):
            ax.annotate(str(value), (value, y), xytext=(5, 0), textcoords='offset points', va='center')
        ax.set_xlim(0, maximum * 1.16)
        ax.spines['bottom'].set_bounds(0, maximum)
        ax.set_title(title, fontsize=11, pad=18)
        ax.set_xlabel('Recordings', labelpad=10)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.tick_params(axis='x', direction='in')
        ax.set_xticks([0, 200, 400, 600, 800] if metric == 'improved_vs_native' else [0, maximum // 2, maximum])
    axes[0].set_yticks(range(len(counts)), [r['label'] for r in counts])
    axes[0].invert_yaxis()
    leader = counts[0]
    fig.text(.035, .955, f"{leader['label']} improves SigMOS on {leader['improved_vs_native']}/800 recordings", fontsize=18)
    fig.text(.035, .91, 'Fixed source-balanced panel · improvement means strictly higher overall SigMOS, not a listening-test judgment', fontsize=11)
    fig.text(.035, .075, 'Improvement counts overlap: several methods can improve the same recording. Winner counts assign exactly one method per recording.', fontsize=10)
    fig.text(.035, .045, 'Native original is its own baseline (0 improvements). Ties: normalized original, native original, then alphabetical method order.', fontsize=10)
    fig.subplots_adjust(left=.30, right=.97, top=.83, bottom=.15, wspace=.24)
    for extension in ['png', 'pdf']:
        fig.savefig(args.output_prefix.with_suffix('.' + extension), dpi=170)
    plt.close(fig)
    print(json.dumps(counts[:3]))


if __name__ == '__main__':
    main()
