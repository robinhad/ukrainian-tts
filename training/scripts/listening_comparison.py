"""Small-multiple SVG plots for the listening page's measured comparison table."""
import html


def comparison_plots(comparison):
    rows = comparison['rows']
    checkpoints = [r for r in rows if r['train_mel'] is not None]
    xmin = min(r['train_mel'] for r in checkpoints)
    xmax = max(r['train_mel'] for r in checkpoints)
    ymin = min(r[k] for r in rows for k in ('sigmos_10', 'sigmos_88') if r[k] is not None)
    ymax = max(r[k] for r in rows for k in ('sigmos_10', 'sigmos_88') if r[k] is not None)
    # Identical scales in both panels; range frames stop at observed extrema.
    x = lambda value: 70 + (value - xmin) / (xmax - xmin) * 490
    y = lambda value: 335 - (value - ymin) / (ymax - ymin) * 275
    panels = []
    for metric, title in [('sigmos_10', 'Same 10 listening items'), ('sigmos_88', 'Full 88-item panel')]:
        points = [r for r in checkpoints if r[metric] is not None]
        latest = points[-1]['label']
        parts = [f'<svg class="mos-plot" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" '
                 f'role="img" aria-label="{title}: training mel loss versus median overall SigMOS. Exact values in the adjacent table.">',
                 f'<text x="70" y="25" class="plot-title">{title}</text>',
                 '<path class="plot-axis" d="M70 60V335H560" fill="none"/>']
        for tick in (39, 41, 43, 45):
            if xmin <= tick <= xmax:
                parts.append(f'<text x="{x(tick):.2f}" y="359" text-anchor="middle" class="plot-tick">{tick}</text>')
        for tick in (2.0, 2.4, 2.8, 3.2):
            if ymin <= tick <= ymax:
                parts.append(f'<text x="57" y="{y(tick)+5:.2f}" text-anchor="end" class="plot-tick">{tick:.1f}</text>')
        parts.extend(['<text x="315" y="388" text-anchor="middle" class="plot-axis-label">Train mel loss (lower ←)</text>',
                      '<text transform="translate(20 200) rotate(-90)" text-anchor="middle" class="plot-axis-label">Median SigMOS (higher ↑)</text>'])
        for ref, label, dash in [(rows[0], 'Original', '5 5'), (rows[1], 'Processed', '2 4')]:
            value = ref[metric]
            parts.append(f'<line x1="70" x2="560" y1="{y(value):.2f}" y2="{y(value):.2f}" '
                         f'class="plot-reference" stroke-dasharray="{dash}"/>')
            parts.append(f'<text x="80" y="{y(value)-9:.2f}" class="plot-reference-label">{label} {value:.3f}</text>')
        for row in points:
            px, py = x(row['train_mel']), y(row[metric])
            label = html.escape(row['label'])
            highlight = ' plot-latest' if row['label'] == latest else ''
            detail = f"{label}: train mel {row['train_mel']:.3f}; SigMOS {row[metric]:.3f}"
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" class="plot-point{highlight}"><title>{detail}</title></circle>')
            dx, dy = {'25K': (-30, -12), '50K': (10, 20), '75K': (10, -12),
                      '100K': (10, 22), '120K': (20, 16), '127K': (26, -12),
                      '131K': (-24, -32), '154K': (10, 25),
                      '171K': (40, -12), '183K': (10, 12),
                      '192K': (10, 30)}.get(row['label'], (10, -12))
            parts.append(f'<text x="{px+dx:.2f}" y="{py+dy:.2f}" class="plot-point-label{highlight}">{label}</text>')
        parts.append('</svg>')
        panels.append(''.join(parts))
    return ('<section aria-label="SigMOS and training loss plots"><h2>Lower training loss does not guarantee higher SigMOS</h2>'
            '<p>Each point is a checkpoint. Panels use identical axes; dotted and dashed lines show processed and original references. '
            'The latest evaluated checkpoint in each panel is highlighted. Missing evaluations are not plotted.</p>'
            '<div class="mos-plots">' + ''.join(panels) + '</div></section>')
