"""Rank logged losses against matched-checkpoint SigMOS; report-only analysis."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze(comparison, epochs):
    checkpoints = [r for r in comparison['rows'] if r['train_mel'] is not None and r['sigmos_10'] is not None]
    steps = [int(r['label'].removesuffix('K')) * 1000 for r in checkpoints]
    lookup = {r['training_step']: r for r in epochs}
    if len(steps) < 4 or len(set(steps)) != len(steps) or len(lookup) != len(epochs):
        raise ValueError('Need at least four unique matched checkpoints and unique epoch records')
    if not set(steps).issubset(lookup):
        raise ValueError('Missing epoch metrics for scored checkpoints')
    y = np.array([r['sigmos_10'] for r in checkpoints])
    if not np.isfinite(y).all() or np.ptp(y) < 1e-12:
        raise ValueError('SigMOS must be finite and nonconstant')
    matched = [lookup[s] for s in steps]
    candidates = sorted({k for r in matched for k in r if k.startswith(('train_', 'valid_')) and k.endswith('loss')})
    design = np.c_[np.ones(len(steps)), np.array(steps) / 1000]
    residual = lambda a: a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    ranking, excluded = [], []
    for key in candidates:
        x = np.array([r.get(key, np.nan) for r in matched])
        pearson = correlation(x, y)
        if pearson is None:
            excluded.append(key)
            continue
        ranking.append({'metric': key, 'count': len(steps), 'pearson_r': pearson,
                        'spearman_rho': correlation(rankdata(x), rankdata(y)),
                        'partial_r_controlling_steps': correlation(residual(x), residual(y))})
    if not ranking:
        raise ValueError('No finite, nonconstant loss metric covers every checkpoint')
    ranking.sort(key=lambda r: (-abs(r['pearson_r']), r['metric']))
    winner = ranking[0]['metric']
    points = [{'step': s, 'label': r['label'], 'metric_value': lookup[s][winner],
               'train_mel': r['train_mel'], 'sigmos_10': r['sigmos_10'], 'sigmos_88': r.get('sigmos_88')}
              for s, r in zip(steps, checkpoints)]
    panel = [r for r in points if r['sigmos_88'] is not None]
    return {'through_step': max(steps), 'listening_ids': comparison['listening_ids'],
            'comparison_sha256': hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest(),
            'target': 'Median synthesized SigMOS overall on the same 10 listening items',
            'selection': 'Largest absolute Pearson correlation; complete matched-checkpoint cases only',
            'candidate_scope': 'All logged training/validation losses; excludes time, LR, memory, step counters and SigMOS components',
            'candidate_count': len(candidates), 'ranked_count': len(ranking), 'excluded_metrics': excluded,
            'selected_metric': winner, 'ranking': ranking, 'points': points,
            'step_pearson_r': correlation(steps, y),
            'panel88': {'count': len(panel), 'pearson_r': correlation([r['metric_value'] for r in panel], [r['sigmos_88'] for r in panel])},
            'selection_mode': 'report_only',
            'caveat': 'Exploratory maximum over multiple metrics on correlated checkpoints from one run. '
                      'The same observations select and assess the winner. Not an independently validated predictor. '
                      'Partial r removes a linear step trend only; the 88-item panel overlaps the listening subset.'}


def collect_epochs(events, comparison):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    steps = {int(r['label'].removesuffix('K')) * 1000 for r in comparison['rows'] if r['train_mel'] is not None}
    rows = {s: {'training_step': s} for s in sorted(steps)}
    for phase in ('train', 'valid'):
        accumulator = EventAccumulator(str(Path(events) / phase), size_guidance={'scalars': 0})
        accumulator.Reload()
        for tag in accumulator.Tags()['scalars']:
            if not tag.endswith('loss'):
                continue
            # ESPnet emits the final batch then the epoch mean at the same step.
            # The latest event at that step is the completed epoch summary.
            for event in sorted(accumulator.Scalars(tag), key=lambda e: e.wall_time):
                if event.step in steps:
                    rows[event.step][f'{phase}_{tag}'] = event.value
    for r in comparison['rows']:
        if r['train_mel'] is not None:
            s = int(r['label'].removesuffix('K')) * 1000
            if not np.isclose(rows[s].get('train_generator_g_mel_loss', np.nan), r['train_mel'], atol=1e-6, rtol=0):
                raise ValueError('Epoch mel loss disagrees with the measured comparison snapshot')
    return list(rows.values())


def plot(report):
    points = report['points']
    xmin, xmax = min(r['metric_value'] for r in points), max(r['metric_value'] for r in points)
    ymin, ymax = min(r['sigmos_10'] for r in points), max(r['sigmos_10'] for r in points)
    x = lambda v: 85 + (v - xmin) / (xmax - xmin) * 570
    y = lambda v: 365 - (v - ymin) / (ymax - ymin) * 275
    variance = report['selected_metric'] == 'valid_generator_var_loss'
    title = 'Lower validation variance loss tracks higher SigMOS' if variance else 'The strongest observed loss–SigMOS relationship'
    axis = 'Validation variance loss (epoch mean)' if variance else 'Selected loss (epoch mean)'
    parts = ['<svg class="correlation-plot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 460" '
             'role="img" aria-label="Selected loss versus median SigMOS; exact values in the checkpoint table below.">',
             f'<text x="85" y="35" class="corr-title">{title}</text>',
             '<path d="M85 90V365H655" fill="none" class="plot-axis"/>']
    for tick in np.linspace(xmin, xmax, 4):
        parts.append(f'<text x="{x(tick):.2f}" y="392" text-anchor="middle">{tick:.3f}</text>')
    for tick in np.linspace(ymin, ymax, 4):
        parts.append(f'<text x="72" y="{y(tick)+5:.2f}" text-anchor="end">{tick:.2f}</text>')
    slope, intercept = np.polyfit([r['metric_value'] for r in points], [r['sigmos_10'] for r in points], 1)
    # Clip the fitted line to the observed range frame.
    parts.extend(['<defs><clipPath id="correlation-fit-clip"><rect x="85" y="90" width="570" height="275"/></clipPath></defs>',
                  f'<path d="M85 {y(slope*xmin+intercept):.2f}L655 {y(slope*xmax+intercept):.2f}" '
                  'class="plot-reference" stroke-dasharray="5 5" clip-path="url(#correlation-fit-clip)"/>'])
    labeled = {min(p['step'] for p in points), max(p['step'] for p in points), max(points, key=lambda r:r['sigmos_10'])['step']}
    for p in points:
        px, py = x(p['metric_value']), y(p['sigmos_10'])
        latest = p['step'] == report['through_step']
        accent = ' corr-latest' if latest else ''
        title = html.escape(f"{p['label']}: loss {p['metric_value']:.5f}; SigMOS {p['sigmos_10']:.5f}")
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" class="corr-point{accent}"><title>{title}</title></circle>')
        if p['step'] in labeled:
            dx = -45 if px > 580 else 10
            dy = 25 if latest else -14
            parts.append(f'<text x="{px+dx:.2f}" y="{py+dy:.2f}" class="corr-label{accent}">{html.escape(p["label"])}</text>')
    parts.extend([f'<text x="370" y="435" text-anchor="middle">{axis}</text>',
                  '<text transform="translate(28 230) rotate(-90)" text-anchor="middle">Median synthesized SigMOS</text>', '</svg>'])
    return ''.join(parts)


def correlation_html(output, comparison, embedded=False):
    path = Path(output) / 'metric_correlation.json'
    if not path.exists():
        return ''
    report = json.loads(path.read_text())
    digest = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()
    if report['comparison_sha256'] != digest:
        raise ValueError('Metric correlation does not match the comparison snapshot; refresh it')
    selected = report['ranking'][0]
    fmt = lambda v: '—' if v is None else f'{v:.3f}'
    name = html.escape(report['selected_metric'])
    mel = next((r['pearson_r'] for r in report['ranking'] if r['metric'] == 'train_generator_g_mel_loss'), None)
    ranking = ''.join(f'<tr><th scope="row">{html.escape(r["metric"])}</th><td>{r["pearson_r"]:.3f}</td>'
                      f'<td>{fmt(r["spearman_rho"])}</td><td>{fmt(r["partial_r_controlling_steps"])}</td></tr>' for r in report['ranking'])
    values = ''.join(f'<tr><th scope="row">{r["label"]}</th><td>{r["metric_value"]:.5f}</td>'
                     f'<td>{r["train_mel"]:.3f}</td><td>{r["sigmos_10"]:.3f}</td></tr>' for r in report['points'])
    links = []
    for filename in ['metric_correlation.json', 'metric_correlation_rankings.csv', 'metric_correlation_points.csv']:
        url = filename
        if embedded:
            url = 'data:application/octet-stream;base64,' + base64.b64encode((Path(output)/filename).read_bytes()).decode()
        links.append(f'<a download="{filename}" href="{url}">{filename}</a>')
    meaning = ('This validation loss combines duration, pitch and energy prediction losses. '
               if report['selected_metric'] == 'valid_generator_var_loss' else '')
    return f'''<section aria-label="Metric correlation with SigMOS"><h2>Which logged loss tracks SigMOS most closely?</h2>
<p><strong>{name}</strong>: Pearson <strong>r = {selected['pearson_r']:.3f}</strong>, Spearman ρ = {fmt(selected['spearman_rho'])},
across {len(report['points'])} checkpoints. {meaning}Selected by largest absolute Pearson correlation among {report['ranked_count']} logged training/validation losses,
using the same {len(report['listening_ids'])} synthesized listening items at every checkpoint. Training mel: r = {fmt(mel)}.
Negative correlation means lower loss accompanies higher SigMOS. Latest selected-loss value: {report['points'][-1]['metric_value']:.5f}.</p>
<style>.correlation-plot{{width:100%;max-width:900px;background:#fffff8;color:#111;font-family:Palatino,Georgia,serif}}.correlation-plot text{{fill:currentColor;font-size:18px}}.correlation-plot .corr-title{{font-size:23px}}.corr-point{{fill:#666}}.correlation-plot .corr-latest{{fill:#a63e25}}@media(prefers-color-scheme:dark){{.correlation-plot{{background:#151515;color:#ddd}}.corr-point{{fill:#aaa}}.correlation-plot .corr-latest{{fill:#e5a084}}}}@media(max-width:600px){{.correlation-plot text{{font-size:24px}}.correlation-plot .corr-label:not(.corr-latest){{display:none}}}}</style>
{plot(report)}<p>Dashed line: fitted linear relationship. Latest checkpoint highlighted; earliest and highest-SigMOS checkpoints labeled. All values appear below.</p>
<p>Exploratory ranking, not a validated replacement for audio evaluation. These checkpoints come from one run, and the same observations select and assess the winner.
Training steps alone correlate at r = {report['step_pearson_r']:.3f}; after removing a linear step trend from both variables,
the selected loss has partial r = {fmt(selected['partial_r_controlling_steps'])}.
On the overlapping 88-item panel: r = {fmt(report['panel88']['pearson_r'])}, n = {report['panel88']['count']} checkpoints; this is not independent validation.
Metrics remain report-only; no training or checkpoint-selection settings change.</p>
<details open><summary>Selected metric by checkpoint</summary><div style="overflow-x:auto"><table class="scores comparison"><thead><tr><th>Checkpoint</th><th>Selected loss</th><th>Train mel</th><th>Median SigMOS · 10 items</th></tr></thead><tbody>{values}</tbody></table></div></details>
<details><summary>All {report['ranked_count']} loss correlations, ranked by |Pearson r|</summary><div style="overflow-x:auto"><table class="scores comparison"><thead><tr><th>Metric</th><th>Pearson r</th><th>Spearman ρ</th><th>Partial r · steps</th></tr></thead><tbody>{ranking}</tbody></table></div></details>
<p>{' · '.join(links)}</p></section>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--events', type=Path)
    source.add_argument('--epochs', type=Path, help='Previously exported portable epoch-loss JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    epochs = collect_epochs(args.events, comparison) if args.events else json.loads(args.epochs.read_text())
    report = analyze(comparison, epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('metric_correlation.json', report), ('metric_correlation_epochs.json', epochs)]:
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    for key in ('ranking', 'points'):
        filename = 'metric_correlation_rankings.csv' if key == 'ranking' else 'metric_correlation_points.csv'
        with (args.output / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(report[key][0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(report[key])
    if (args.output / 'selection.jsonl').exists():
        from training.scripts.generate_checkpoint_listening import write_pages
        rows = [json.loads(line) for line in (args.output / 'selection.jsonl').read_text().splitlines()]
        step = json.loads((args.output / 'report.json').read_text())['checkpoint_step']
        write_pages(args.output, rows, step)
    print(json.dumps(report['ranking'][0]))


if __name__ == '__main__':
    main()
