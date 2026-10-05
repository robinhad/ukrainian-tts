"""Training-corpus totals and portable source/speaker plots for listening pages."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from training.quality.common import file_hash, write_json


SOURCE_NAMES = {'common_voice_available_uk': 'Common Voice', 'fleurs_uk': 'FLEURS',
                'opentts_lada': 'OpenTTS Lada', 'opentts_mykyta': 'OpenTTS Mykyta',
                'opentts_tetiana': 'OpenTTS Tetiana', 'tg_voices_uk': 'Telegram voices',
                'ua_ser': 'UA SER', 'ukr_dialects': 'Ukrainian dialects'}


def summarize(frame):
    required = ['utterance_id', 'speaker_id', 'source_id', 'duration', 'split']
    if frame[required].isna().any().any() or frame.utterance_id.duplicated().any():
        raise ValueError('Missing dataset fields or duplicate recordings')
    if frame.empty or not frame['split'].str.endswith('_train').all():
        raise ValueError('Statistics must describe a nonempty training split only')
    if not np.isfinite(frame.duration).all() or (frame.duration <= 0).any():
        raise ValueError('Invalid recording duration')
    frame = frame.copy()
    frame['recording_id_placeholder'] = frame.speaker_id.eq(frame.utterance_id)
    datasets = []
    for source, group in frame.groupby('source_id'):
        datasets.append({'source_id': source, 'label': SOURCE_NAMES.get(source, source),
                         'recordings': len(group), 'speaker_ids': group.speaker_id.nunique(),
                         'hours': float(group.duration.sum() / 3600)})
    speakers = []
    for speaker, group in frame.groupby('speaker_id'):
        speakers.append({'speaker_id': speaker, 'source_ids': sorted(group.source_id.unique()),
                         'recordings': len(group), 'hours': float(group.duration.sum() / 3600),
                         'recording_id_placeholder': bool(group.recording_id_placeholder.all())})
    datasets.sort(key=lambda r: (-r['hours'], r['source_id']))
    speakers.sort(key=lambda r: (-r['hours'], r['speaker_id']))
    return {'scope': 'training split only; processed audio; development and evaluation excluded',
            'recordings': len(frame), 'total_hours': float(frame.duration.sum() / 3600),
            'total_speaker_ids': len(speakers), 'verified_unique_people': None,
            'recording_level_speaker_ids': sum(r['recording_id_placeholder'] for r in speakers),
            'datasets': datasets, 'speakers': speakers}


def write_stats(output, manifest, config):
    config = yaml.safe_load(Path(config).read_text())
    speech = [r[0] for r in config['train_data_path_and_name_and_type'] if r[1] == 'speech']
    if len(speech) != 1:
        raise ValueError('Expected one training speech index')
    frame = pd.read_parquet(manifest)
    indexed = dict(line.split(maxsplit=1) for line in Path(speech[0]).read_text().splitlines() if line.strip())
    if set(indexed) != set(frame.utterance_id):
        raise ValueError('Training manifest does not match checkpoint training IDs')
    for row in frame.itertuples():
        if Path(indexed[row.utterance_id]).resolve() != Path(row.audio_path).resolve():
            raise ValueError('Training manifest points to different audio')
    result = summarize(frame)
    result['manifest_sha256'] = file_hash(manifest)
    output = Path(output)
    write_json(output / 'training_data.json', result)
    for name, rows in [('datasets', result['datasets']), ('speakers', result['speakers'])]:
        with (output / f'training_{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    return result


def bar_plot(entries, title):
    height = 95 + 29 * len(entries)
    maximum = max(value for label, value in entries)
    parts = [f'<svg class="data-bars" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 {height}" '
             f'role="img" aria-label="{html.escape(title)}; hours are printed beside each bar.">',
             f'<text x="12" y="27" class="bar-title">{html.escape(title)}</text>']
    for i, (label, value) in enumerate(entries):
        y = 65 + i * 29
        width = value / maximum * 380
        parts.extend([f'<text x="238" y="{y+5}" text-anchor="end">{html.escape(label)}</text>',
                      f'<rect x="250" y="{y-8}" width="{width:.3f}" height="14" class="hour-bar"/>',
                      f'<text x="{260+width:.3f}" y="{y+5}">{value:.4f} h</text>'])
    parts.append('</svg>')
    mobile = '<div class="mobile-hours"><h3>' + html.escape(title) + '</h3><ol>' + ''.join(
        f'<li><span>{html.escape(label)}</span><span>{value:.4f} h</span>'
        f'<i aria-hidden="true" style="width:{value / maximum * 100:.4f}%"></i></li>'
        for label, value in entries) + '</ol></div>'
    return '<div class="hours-chart">' + ''.join(parts) + mobile + '</div>'


def stats_html(output, embedded=False):
    output = Path(output)
    if not (output / 'training_data.json').exists():
        return ''
    stats = json.loads((output / 'training_data.json').read_text())
    sources, speakers = stats['datasets'], stats['speakers']
    top = speakers[:15]
    def speaker_label(row):
        name = row['speaker_id']
        if len(name) > 18:
            name = name[:8] + '…' + name[-6:]
        return name
    top_share = sum(r['hours'] for r in top) / stats['total_hours'] * 100
    source_plot = bar_plot([(r['label'], r['hours']) for r in sources],
                          f"{sources[0]['label']} supplies {sources[0]['hours'] / stats['total_hours'] * 100:.1f}% of hours")
    speaker_plot = bar_plot([(speaker_label(r), r['hours']) for r in top],
                           f'Top {len(top)} speaker IDs cover {top_share:.1f}% of hours')
    downloads = []
    for name, label in [('training_datasets.csv', 'Dataset hours CSV'), ('training_speakers.csv', 'All speaker IDs CSV'),
                        ('training_data.json', 'Complete statistics JSON')]:
        url = name
        if embedded:
            mime = 'application/json' if name.endswith('.json') else 'text/csv'
            url = f'data:{mime};base64,' + base64.b64encode((output / name).read_bytes()).decode()
        downloads.append(f'<a download="{name}" href="{url}">{label}</a>')
    # JSON stays data, with HTML-significant characters escaped to prevent script termination.
    payload = json.dumps(speakers, ensure_ascii=True).replace('<', '\\u003c').replace('&', '\\u0026')
    return f'''<section aria-label="Training data statistics" class="training-stats">
<h2>Training data behind this checkpoint</h2>
<p><strong>{stats['total_hours']:.3f} hours</strong> · <strong>{stats['total_speaker_ids']:,} speaker IDs</strong> ·
{stats['recordings']:,} recordings · {len(sources)} datasets.</p>
<p>Processed training audio only; development and held-out evaluation are excluded. These are unique recording durations, not repeated training exposure.
{stats['recording_level_speaker_ids']:,} speaker IDs are recording-level placeholders. The number of distinct people is unknown; speaker-ID counts must not be read as verified human counts.</p>
<div class="dataset-plots">{source_plot}{speaker_plot}</div>
<p>The speaker plot shows the top {len(top)} of {len(speakers):,} IDs, covering {top_share:.1f}% of training hours. Search the full breakdown below.</p>
<p>{' · '.join(downloads)}</p>
<details><summary>Browse hours for all {len(speakers):,} speaker IDs</summary>
<label for="speaker-search">Search speaker ID or dataset</label><input id="speaker-search" type="search" placeholder="Speaker ID or dataset"/>
<p id="speaker-count" aria-live="polite"></p><div style="overflow-x:auto"><table class="scores comparison"><thead><tr>
<th scope="col">Speaker ID</th><th scope="col">Dataset</th><th scope="col">Recordings</th><th scope="col">Hours</th><th scope="col">ID type</th>
</tr></thead><tbody id="speaker-rows"></tbody></table></div>
<button type="button" id="speaker-prev">Previous 50</button> <button type="button" id="speaker-next">Next 50</button></details>
<script type="application/json" id="speaker-data">{payload}</script>
<script>(()=>{{const rows=JSON.parse(document.getElementById('speaker-data').textContent);let page=0;
const input=document.getElementById('speaker-search'),body=document.getElementById('speaker-rows');
function render(){{const q=input.value.toLowerCase().trim(),filtered=rows.filter(r=>(r.speaker_id+' '+r.source_ids.join(' ')).toLowerCase().includes(q));
const start=page*50;body.replaceChildren();filtered.slice(start,start+50).forEach(r=>{{const tr=document.createElement('tr');
[r.speaker_id,r.source_ids.join(', '),r.recordings,r.hours.toFixed(4),r.recording_id_placeholder?'Recording-level':'Source-provided'].forEach(v=>{{const td=document.createElement('td');td.textContent=v;tr.append(td)}});body.append(tr)}});
document.getElementById('speaker-count').textContent=filtered.length?`${{start+1}}–${{Math.min(start+50,filtered.length)}} of ${{filtered.length}} speaker IDs`:'No matching speaker IDs';
document.getElementById('speaker-prev').disabled=page===0;document.getElementById('speaker-next').disabled=start+50>=filtered.length}}
input.addEventListener('input',()=>{{page=0;render()}});document.getElementById('speaker-prev').onclick=()=>{{page--;render()}};
document.getElementById('speaker-next').onclick=()=>{{page++;render()}};render()}})();</script></section>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = write_stats(args.output, args.manifest, args.config)
    from training.scripts.generate_checkpoint_listening import write_pages
    rows = [json.loads(line) for line in (args.output / 'selection.jsonl').read_text().splitlines()]
    report = json.loads((args.output / 'report.json').read_text())
    write_pages(args.output, rows, report['checkpoint_step'])
    print(json.dumps({key: value for key, value in result.items() if key not in ('datasets', 'speakers')}))


if __name__ == '__main__':
    main()
