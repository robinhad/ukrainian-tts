#!/usr/bin/env python3
"""Build local listening pages for the exact per-item-best, SigMOS >=3.5 pilot."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import zipfile

from plot_individual_sigmos_changes import SOURCES, read_panel
from plot_processing_sigmos import LABELS


def page(rows, *, preview, baseline='native'):
    baseline_label = 'Normalized original' if baseline == 'normalized' else 'Native original'
    baseline_variant = 'normalized_original' if baseline == 'normalized' else 'original'
    variant_labels = {baseline_variant: baseline_label, 'processed': 'Full cascade'}
    if baseline == 'brute_force':
        variant_labels = {**LABELS, 'normalized_original': 'Normalized original'}
    if baseline == 'mfa':
        variant_labels = {'mfa_trim': 'MFA boundary trim / normalize'}
    cards = []
    for row in rows:
        source = SOURCES[row['source_id']]
        comparison = (f"native original {row['original_score']:.3f} / selected {row['selected_score']:.3f}"
                      if baseline in {'brute_force', 'mfa'} else
                      f"{baseline_label.lower()} {row['original_score']:.3f} / full cascade {row['processed_score']:.3f}")
        reference_player = (f'<p>Native original</p><audio controls preload="none" aria-label="Native original {html.escape(source)} recording">'
                            f'<source src="{html.escape(row["reference_url"], quote=True)}" type="audio/wav"></audio><p>Selected version</p>'
                            if 'reference_url' in row else '')
        if 'energy_url' in row:
            reference_player += (f'<p>Current energy trim · {row["energy_score"]:.3f} SigMOS</p>'
                                 f'<audio controls preload="none" aria-label="Current energy trim"><source src="{html.escape(row["energy_url"], quote=True)}" type="audio/wav"></audio>'
                                 '<p>MFA trim</p>')
        review = ('<p class="meta">Review: ' + html.escape(', '.join(row['review_flags'])) +
                  ' · untrimmed audio preserved</p>' if row.get('review_flags') else '')
        cards.append(f'''<article data-source="{html.escape(row['source_id'])}"
data-variant="{row['selected_variant']}">
<h2>{html.escape(source)} <span>{row['selected_score']:.3f} SigMOS</span></h2>
<p class="meta">Selected: <strong>{variant_labels[row['selected_variant']]}</strong> · {row['seconds']:.2f} s
· {comparison}</p>
<p lang="uk">{html.escape(row['text'])}</p>
{review}
{reference_player}
<audio controls preload="none" aria-label="Selected {html.escape(source)} recording">
<source src="{html.escape(row['url'], quote=True)}" type="audio/wav"></audio>
<p class="id">{row['sample_id']}</p></article>''')
    options = ''.join(f'<option value="{html.escape(k)}">{html.escape(v)}</option>' for k, v in SOURCES.items())
    subtitle = (f'{len(rows)} examples · up to four per source across retained score ranges. Audio is embedded for offline playback.'
                if preview else f'All {len(rows)} retained recordings from the completed 800-recording pilot.')
    title = 'Brute-force best + SigMOS ≥3.5' if baseline == 'brute_force' else 'Per-item best + SigMOS ≥3.5'
    description = ('Compare the native original with the highest-scoring version across all 17 variants. Every selected score is at least 3.5. These are labeled listening comparisons, not a blind test.'
                   if baseline == 'brute_force' else
                   f'Each player contains the chosen audio: full-cascade processing if its overall SigMOS is higher, otherwise the {baseline_label.lower()}. Every selected score is at least 3.5.')
    if baseline == 'mfa':
        title = 'MFA boundary trimming — listening comparison'
        subtitle = f'{len(rows)} examples · four per source, emphasizing large trims and review cases. No SigMOS cutoff.'
        description = 'Compare native original, current energy trim, and MFA boundary trim with 100 ms padding. Review cases preserve untrimmed audio. Scores and labels are visible; this is not a blind test.'
    variants = (sorted({r['selected_variant'] for r in rows}) if baseline in {'brute_force', 'mfa'} else [baseline_variant, 'processed'])
    variant_options = ''.join(f'<option value="{html.escape(v)}">{html.escape(variant_labels[v])}</option>' for v in variants)
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>''' + title + ''' — listening</title>
<style>
:root{color-scheme:light dark;--bg:#fffff8;--fg:#222;--muted:#666;--rule:#ccc;--accent:#285b80}
@media(prefers-color-scheme:dark){:root{--bg:#151515;--fg:#ddd;--muted:#aaa;--rule:#555;--accent:#8dbadd}}
*{box-sizing:border-box}body{margin:0 auto;padding:24px;max-width:980px;background:var(--bg);color:var(--fg);font:17px/1.55 system-ui,sans-serif}
h1{font:600 32px/1.2 Georgia,serif;margin-bottom:12px}h2{font:600 22px/1.3 Georgia,serif;margin-bottom:8px}
h2 span{float:right;color:var(--accent)}.meta,.id{color:var(--muted);font-size:14px}.id{font-family:monospace}
article{border-top:1px solid var(--rule);padding:18px 0}article[hidden]{display:none}audio{width:100%;max-width:640px}
nav{display:flex;flex-wrap:wrap;gap:18px;margin:24px 0}label{display:grid;gap:5px}select{font:inherit;padding:6px;max-width:100%;color:var(--fg);background:var(--bg);border:1px solid var(--rule)}
@media(max-width:500px){body{padding:16px}h1{font-size:26px}h2 span{float:none;display:block}nav{display:block}label{margin:12px 0}}
</style><h1>''' + title + '''</h1><p>''' + subtitle + '''</p>
<p>''' + description + '''</p>
<p class="meta">''' + ('These are the exact scored files from the completed 800-recording pilot.' if baseline == 'mfa' else 'Full cascade: ClearVoice → Sidon → DeepFilterNet3. These are the exact scored files from the completed 800-recording pilot.') + '''</p>
''' + ('<p>Normalized original: mono, 24 kHz, boundary silence trimmed, input loudness matched with a −0.1 dBFS peak cap; no enhancement models.</p>' if baseline == 'normalized' else '') + '''
<p><a href="''' + ('index.html">All retained recordings' if preview else 'preview.html">Embedded audio preview') + '''</a> · <a href="listening_set.zip">Download listening set</a></p>
<nav aria-label="Recording filters"><label>Source<select id="source"><option value="">All sources</option>''' + options + '''</select></label>
<label>Selected version<select id="variant"><option value="">All versions</option>''' + variant_options + '''</select></label></nav>
<p id="count" aria-live="polite"></p><main>''' + '\n'.join(cards) + '''</main>
<script>
const source=document.getElementById('source'),variant=document.getElementById('variant'),cards=[...document.querySelectorAll('article')];
function filter(){let n=0;cards.forEach(c=>{c.hidden=!!((source.value&&c.dataset.source!==source.value)||(variant.value&&c.dataset.variant!==variant.value));if(!c.hidden)n++;else c.querySelectorAll('audio').forEach(a=>a.pause())});document.getElementById('count').textContent=n+' recordings shown'}
source.addEventListener('change',filter);variant.addEventListener('change',filter);
document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true);filter();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', choices=['native', 'normalized'], default='native')
    parser.add_argument('--selection', type=Path,
                        help='Require exact agreement with the normalized selector decision JSONL')
    args = parser.parse_args()
    original, processed = read_panel(args.original), read_panel(args.processed)
    if original.keys() != processed.keys():
        raise ValueError('Mismatched panels')
    decisions = None
    if args.baseline == 'normalized':
        if args.selection is None:
            parser.error('--baseline normalized requires --selection')
        records = [json.loads(line) for line in args.selection.read_text().splitlines() if line.strip()]
        decisions = {r['sample_id']: r for r in records}
        if len(records) != 800 or len(decisions) != 800:
            raise ValueError('Expected 800 unique selection decisions')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audio').mkdir(exist_ok=True)
    rows = []
    for key in sorted(original):
        before, after = original[key], processed[key]
        chosen = after if after['sigmos_overall'] > before['sigmos_overall'] else before
        identifier = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        variant = 'processed' if chosen is after else ('normalized_original' if args.baseline == 'normalized' else 'original')
        if decisions is not None:
            decision = decisions[identifier]
            if (decision['source_id'] != key[0]
                    or decision['normalized_sigmos_overall'] != before['sigmos_overall']
                    or decision['processed_sigmos_overall'] != after['sigmos_overall']
                    or decision['selected_variant'] != variant
                    or decision['selected_sigmos_overall'] != chosen['sigmos_overall']
                    or decision['selected_audio_sha256'] != chosen['audio_sha256']
                    or decision['retained'] != (chosen['sigmos_overall'] >= 3.5)):
                raise ValueError('Listening selection differs from evaluated selector')
        if chosen['sigmos_overall'] < 3.5:
            continue
        source = Path(chosen['audio_path'])
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        if checksum != chosen['audio_sha256']:
            raise ValueError('Scored audio changed')
        target = args.output / 'audio' / f'{identifier}.wav'
        shutil.copyfile(source, target)
        rows.append({'sample_id': identifier, 'source_id': key[0],
                     'selected_variant': variant,
                     'selected_score': chosen['sigmos_overall'],
                     'original_score': before['sigmos_overall'], 'processed_score': after['sigmos_overall'],
                     'seconds': chosen['duration_seconds'], 'text': before['text'],
                     'url': f'audio/{identifier}.wav', 'audio_sha256': checksum})
    rows.sort(key=lambda r: (r['source_id'], r['selected_score'], r['sample_id']))
    if not rows:
        raise ValueError('Selection retained no recordings')
    (args.output / 'index.html').write_text(page(rows, preview=False, baseline=args.baseline))
    (args.output / 'selection.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    preview = []
    for source in SOURCES:
        candidates = [r for r in rows if r['source_id'] == source]
        count = min(4, len(candidates))
        for i in range(count):
            row = dict(candidates[round(i * (len(candidates) - 1) / max(1, count - 1))])
            row['url'] = 'data:audio/wav;base64,' + base64.b64encode((args.output / row['url']).read_bytes()).decode()
            preview.append(row)
    (args.output / 'preview.html').write_text(page(preview, preview=True, baseline=args.baseline))
    with zipfile.ZipFile(args.output / 'listening_set.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in [args.output / 'index.html', args.output / 'preview.html', args.output / 'selection.jsonl',
                     *[args.output / row['url'] for row in rows]]:
            archive.write(path, path.relative_to(args.output))
    report = {'baseline': args.baseline, 'recordings': len(rows), 'preview_recordings': len(preview),
              'sources': dict(Counter(r['source_id'] for r in rows)),
              'selected_variants': dict(Counter(r['selected_variant'] for r in rows)),
              'median_selected_sigmos': statistics.median(r['selected_score'] for r in rows),
              'exact_scored_files_verified': True}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
