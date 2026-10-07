"""Conditional step extrapolations from a fixed listening subset's SigMOS history."""
from __future__ import annotations

import html
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np


def forecast(comparison, statistic='median'):
    if statistic not in ('median', 'p05'):
        raise ValueError('Unknown forecast statistic')
    def score(row):
        if statistic == 'median':
            return float(row['sigmos_10'])
        from training.scripts.listening_metric_correlation import score_interval
        value = score_interval(row, comparison['listening_ids'])['sigmos_p05']
        if value is None:
            raise ValueError('P05 forecast requires per-item scores at every checkpoint and reference')
        return value
    points = sorted([(int(row['label'].removesuffix('K')) * 1000, score(row))
                     for row in comparison['rows']
                     if row['train_mel'] is not None and row['sigmos_10'] is not None])
    if len(points) < 4 or len({step for step, _ in points}) != len(points):
        raise ValueError('Forecast needs at least four distinct scored checkpoints')
    if not all(step > 0 and math.isfinite(score) for step, score in points):
        raise ValueError('Forecast inputs must be finite and steps positive')
    targets = {name: next(score(row) for row in comparison['rows'] if row['label'] == label)
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
    result = {'metric': 'sigmos_overall', 'statistic': statistic,
            'current_score': points[-1][1], 'sample_count': len(comparison['listening_ids']),
            'latest_scored_step': latest, 'targets': targets, 'scenarios': scenarios,
            'selection_mode': 'report_only', 'uncertainty': 'Model sensitivity, not confidence intervals; '
            'checkpoint errors are correlated. Assumes continued improvement; a plateau may prevent reaching either target.'}
    if statistic == 'median':
        result['lower_bound'] = (forecast(comparison, 'p05') if all(
            row.get('sigmos_10_values') is not None for row in comparison['rows']) else None)
    return result


def forecast_html(result):
    def eta_text(value):
        if value.get('status') == 'step_reached':
            return 'Step already reached; quality still needs evaluation'
        if value.get('status') != 'projected':
            return 'ETA unavailable'
        stamp = datetime.fromisoformat(value['eta_kyiv'])
        label = f'{stamp:%d %b %Y, %H:%M} Kyiv · {value["remaining_hours"]:.1f} h remaining'
        if value.get('beyond_schedule'):
            label += ' · hypothetical, beyond scheduled training'
        return label

    def estimate(value):
        if value['status'] == 'observed':
            return f"Observed at {value['total_steps'] / 1000:,.0f}K"
        if value['total_steps'] is None:
            return 'No supported crossing'
        text = (f"≈{value['total_steps'] / 1000:,.0f}K total "
                f"(+{value['additional_steps'] / 1000:,.0f}K)")
        if 'eta' in value:
            text += '<br><small>' + eta_text(value['eta']) + '</small>'
        return text
    def table(data, label):
        body = ''.join('<tr><th scope="row">' + html.escape(row['method']) + '</th>' + ''.join(
            '<td>' + estimate(row['targets'][name]) + '</td>' for name in ('original', 'processed')) + '</tr>'
            for row in data['scenarios'])
        targets = data['targets']
        return ('<div style="overflow-x:auto"><table class="scores comparison">'
                f'<caption>{label} · sensitivity to trend assumption</caption>'
                '<thead><tr><th scope="col">Trend assumption</th>'
                f'<th scope="col">Original {targets["original"]:.3f}</th>'
                f'<th scope="col">Processed {targets["processed"]:.3f}</th></tr></thead>'
                f'<tbody>{body}</tbody></table></div>')
    lower = result.get('lower_bound')
    metrics = [('Median', result)] + ([('P05 lower bound', lower)] if lower else [])
    summary = ''.join(f'<tr><th scope="row">{label}</th><td>{data["current_score"]:.3f}</td>'
                      f'<td>{data["targets"]["processed"]:.3f}</td>'
                      f'<td>{estimate(data["scenarios"][0]["targets"]["processed"])}</td></tr>'
                      for label, data in metrics)
    timing_note = ''
    timing = result.get('timing')
    if timing:
        stamp = datetime.fromisoformat(timing['captured_at']).astimezone(ZoneInfo('Europe/Kyiv'))
        timing_note = f'<p><strong>ETA snapshot: {stamp:%d %b %Y, %H:%M} Kyiv.</strong> '
        if timing['status'] == 'measured':
            timing_note += (f'Training was at {timing["current_step"]:,} steps, '
                            f'averaging {timing["steps_per_hour"]:,.0f} steps/hour over the latest log window. ')
        else:
            timing_note += 'Fresh training progress and throughput are unavailable; calendar ETAs are omitted. '
        timing_note += ('Dates and remaining hours are fixed at this snapshot, not a live countdown. '
                        'They assume uninterrupted training and exclude preparation/evaluation overhead.</p>')
        if timing.get('scheduled_end_step'):
            timing_note += (f'<p><strong>Scheduled finish: {timing["scheduled_end_step"]:,} steps</strong> · '
                            f'{eta_text(result.get("scheduled_finish", {}))}.</p>')
    return ('<section id="source-quality-forecast" aria-label="Conditional SigMOS forecast"><h2>When will SigMOS reach source quality?</h2>'
            f'<p>Conditional extrapolations from the same {result["sample_count"]} listening items, '
            f'through {result["latest_scored_step"] / 1000:,.0f}K steps. Additional steps are measured from that scored checkpoint. '
            'Fits use training steps directly, not mel loss. Targets describe the matched listening references, not the full source corpus.</p>' + timing_note +
            '<div style="overflow-x:auto"><table class="scores comparison"><caption>Processed-source target · linear fit across all checkpoints</caption>'
            '<thead><tr><th scope="col">Statistic</th><th scope="col">Current</th><th scope="col">Reference</th>'
            f'<th scope="col">Predicted training step</th></tr></thead><tbody>{summary}</tbody></table></div>' +
            table(result, 'Median SigMOS') + (table(lower, 'P05 lower bound') if lower else
            '<p>P05 prediction unavailable: complete per-item scores are required for all checkpoints and references.</p>') +
            '<p>P05 is the 5th percentile, the lower edge of the central 90% score range, using linear interpolation. '
            'Its trend is fitted separately from the median. An observed crossing does not imply later checkpoints stay above the target.</p>'
            '<p>Linear fits assume a constant average gain per step; the logarithmic fit assumes diminishing gains. '
            'These models are unvalidated extrapolations of a small, fluctuating subset, not a guarantee or a scheduling recommendation. '
            'The differences between fits are not statistical confidence bounds; quality could plateau before either target. '
            'Changing the training dataset requires a new trend assessment. '
            'Confirm progress on the full 88-item panel. No training settings are changed.</p></section>')
