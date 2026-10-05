"""Conditional step extrapolations from a fixed listening subset's SigMOS history."""
from __future__ import annotations

import html
import math

import numpy as np


def forecast(comparison):
    points = sorted([(int(row['label'].removesuffix('K')) * 1000, float(row['sigmos_10']))
                     for row in comparison['rows']
                     if row['train_mel'] is not None and row['sigmos_10'] is not None])
    if len(points) < 4 or len({step for step, _ in points}) != len(points):
        raise ValueError('Forecast needs at least four distinct scored checkpoints')
    if not all(step > 0 and math.isfinite(score) for step, score in points):
        raise ValueError('Forecast inputs must be finite and steps positive')
    targets = {name: next(row['sigmos_10'] for row in comparison['rows'] if row['label'] == label)
               for name, label in [('original', 'Original source audio'), ('processed', 'Processed source audio')]}
    if not all(math.isfinite(value) for value in targets.values()):
        raise ValueError('Nonfinite forecast target')
    latest = points[-1][0]
    scenarios = []
    for label, subset, logarithmic in [('Linear, all checkpoints', points, False),
                                       ('Linear, latest four', points[-4:], False),
                                       ('Logarithmic, all checkpoints', points, True)]:
        steps, values = np.array(subset, dtype=float).T
        x = np.log(steps / 1000) if logarithmic else steps / 1000
        slope, intercept = np.polyfit(x, values, 1)
        estimates = {}
        for name, target in targets.items():
            observed = [step for step, value in points if value >= target]
            if observed:
                estimates[name] = {'status': 'observed', 'total_steps': min(observed), 'additional_steps': 0}
            elif slope <= 1e-12:
                estimates[name] = {'status': 'no_positive_trend', 'total_steps': None, 'additional_steps': None}
            else:
                crossing = (target - intercept) / slope
                if logarithmic:
                    crossing = math.exp(crossing) if crossing < 700 else math.inf
                total = crossing * 1000
                if not math.isfinite(total) or total <= latest:
                    estimates[name] = {'status': 'unresolved', 'total_steps': None, 'additional_steps': None}
                else:
                    estimates[name] = {'status': 'extrapolated', 'total_steps': float(total),
                                       'additional_steps': float(total - latest)}
        scenarios.append({'method': label, 'checkpoint_count': len(subset),
                          'first_step': int(steps[0]), 'last_step': int(steps[-1]),
                          'slope': float(slope), 'intercept': float(intercept), 'targets': estimates})
    return {'metric': 'sigmos_overall', 'sample_count': len(comparison['listening_ids']),
            'latest_scored_step': latest, 'targets': targets, 'scenarios': scenarios,
            'selection_mode': 'report_only', 'uncertainty': 'Model sensitivity, not confidence intervals; '
            'checkpoint errors are correlated. Assumes continued improvement; a plateau may prevent reaching either target.'}


def forecast_html(result):
    def estimate(value):
        if value['status'] == 'observed':
            return f"Observed at {value['total_steps'] / 1000:,.0f}K"
        if value['total_steps'] is None:
            return 'No supported crossing'
        return (f"≈{value['total_steps'] / 1000:,.0f}K total "
                f"(+{value['additional_steps'] / 1000:,.0f}K)")
    body = ''.join('<tr><th scope="row">' + html.escape(row['method']) + '</th>' + ''.join(
        '<td>' + estimate(row['targets'][name]) + '</td>' for name in ('original', 'processed')) + '</tr>'
        for row in result['scenarios'])
    targets = result['targets']
    return ('<section aria-label="Conditional SigMOS forecast"><h2>Estimated steps to reference quality</h2>'
            f'<p>Conditional extrapolations from the same {result["sample_count"]} listening items, '
            f'through {result["latest_scored_step"] / 1000:,.0f}K steps. Additional steps are measured from that scored checkpoint. '
            'Fits use training steps directly, not mel loss.</p><div style="overflow-x:auto">'
            '<table class="scores comparison"><caption>Model sensitivity — not confidence intervals</caption>'
            '<thead><tr><th scope="col">Trend assumption</th>'
            f'<th scope="col">Original median {targets["original"]:.3f}</th>'
            f'<th scope="col">Processed median {targets["processed"]:.3f}</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>'
            '<p>Linear fits assume a constant average gain per step; the logarithmic fit assumes diminishing gains. '
            'These models are unvalidated extrapolations of a small, fluctuating subset, not a guarantee or a scheduling recommendation. '
            'The differences between fits are not statistical confidence bounds; quality could plateau before either target. '
            'Changing the training dataset requires a new trend assessment. '
            'Confirm progress on the full 88-item panel. No training settings are changed.</p></section>')
