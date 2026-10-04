"""Generate a reproducible held-out checkpoint listening page with reference audio."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import shutil
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(panel, count):
    """Round-robin sources with seeded ordering, independent of synthesis quality."""
    groups = defaultdict(list)
    for row in sorted(panel, key=lambda r: r['utterance_id']):
        groups[row['source_id']].append(row)
    rng = random.Random(777)
    for source in sorted(groups):
        rng.shuffle(groups[source])
    selected = []
    while len(selected) < count:
        for source in sorted(groups):
            if groups[source] and len(selected) < count:
                selected.append(groups[source].pop())
        if not any(groups.values()) and len(selected) < count:
            raise ValueError('Not enough held-out items')
    return selected


def page(rows, step, embedded=False, output=None):
    cards = []
    for index, row in enumerate(rows, 1):
        players = []
        for key, label in [('synthesis', f'Generated · {step:,} steps'),
                           ('reference', 'Original recording'),
                           ('processed', 'Processed reference')]:
            url = row[key]
            if embedded:
                url = 'data:audio/wav;base64,' + base64.b64encode((output / url).read_bytes()).decode()
            quality = row.get('quality', {}).get(key)
            scores = ''
            if quality:
                metrics = [('sigmos_overall', 'SigMOS overall'), ('audiobox_pq', 'Audiobox PQ'),
                           ('sigmos_speech', 'Speech'), ('sigmos_noise', 'Noise'),
                           ('sigmos_coloration', 'Coloration'), ('sigmos_discontinuity', 'Discontinuity'),
                           ('sigmos_loudness', 'Loudness'), ('sigmos_reverb', 'Reverberation')]
                scores = '<table class="scores"><caption>Whole-file quality</caption><tbody>' + ''.join(
                    f'<tr><th scope="row">{name}</th><td>{quality[metric]:.3f}</td></tr>'
                    for metric, name in metrics) + '</tbody></table>'
            players.append(f'<div><label>{label}</label><audio controls preload="none" src="{html.escape(url)}"></audio>{scores}</div>')
        cards.append(f'<article><div class="meta">{index:02d} · {html.escape(row["source_id"])}'
                     f' · generated {row["duration_seconds"]:.2f}s</div>'
                     f'<p lang="uk">{html.escape(row["text"])}</p>'
                     f'<div class="players">{"".join(players)}</div></article>')
    quality_note = ('<p>Scores are model predictions for the exact audio below, not human ratings. '
                    'Higher is better within each metric; SigMOS and Audiobox PQ use different scales. '
                    'All scores are report-only. <a href="quality.csv">Download CSV</a> · '
                    '<a href="quality.jsonl">Download JSONL</a></p>'
                    if any(r.get('quality') for r in rows) else '')
    if embedded and quality_note:
        for name, mime in [('quality.csv', 'text/csv'), ('quality.jsonl', 'application/x-ndjson')]:
            encoded = base64.b64encode((output / name).read_bytes()).decode()
            quality_note = quality_note.replace(f'href="{name}"', f'download="{name}" href="data:{mime};base64,{encoded}"')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ukrainian TTS · {step:,}-step listening test</title>
<style>:root{{color-scheme:light dark;font-family:system-ui,sans-serif;background:#f8f8f5;color:#202622}}
body{{max-width:1100px;margin:40px auto;padding:0 20px}}h1{{font-size:2rem;letter-spacing:-.04em;margin-bottom:12px}}
header p{{max-width:850px;line-height:1.6;color:#545e56}}article{{padding:24px 0;border-top:1px solid #d5dbd5}}
.meta{{font-size:.85rem;color:#545e56}}article p{{font-size:1.15rem;line-height:1.6}}
.players{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}}label{{display:block;font-size:.9rem;margin-bottom:9px}}
.scores{{width:100%;font-size:.85rem;border-collapse:collapse;margin-top:12px;font-variant-numeric:tabular-nums}}.scores caption{{text-align:left;margin-bottom:6px}}.scores th{{text-align:left;font-weight:400}}.scores td{{text-align:right}}.scores th,.scores td{{padding:3px 0}}
audio{{width:100%}}a{{color:#226342}}@media(max-width:740px){{.players{{grid-template-columns:1fr}}body{{margin:24px auto}}}}
@media(prefers-color-scheme:dark){{:root{{background:#151b17;color:#ebefea}}header p,.meta{{color:#acb9ad}}article{{border-color:#354036}}a{{color:#9ad5ab}}}}
</style><header><h1>Ukrainian TTS · {step:,} steps</h1>
<p>{len(rows)} held-out texts across {len(set(r['source_id'] for r in rows))} sources. Selected before synthesis with a fixed seed.
This is a saved training checkpoint. Speaker conditioning uses the processed reference.
References belong to the unfiltered evaluation set; they were not used to train the model.
Generated audio is presented without enhancement or loudness normalization.</p>{quality_note}</header>
{''.join(cards)}<script>document.addEventListener('play',e=>{{if(e.target.tagName==='AUDIO')
document.querySelectorAll('audio').forEach(a=>{{if(a!==e.target)a.pause()}})}},true)</script></html>'''


def write_pages(output, rows, step):
    output.joinpath('index.html').write_text(page(rows, step))
    output.joinpath('preview.html').write_text(page(rows, step, True, output))
    with zipfile.ZipFile(output / 'listening_set.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.relative_to(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'manifest', 'panel', 'xvector', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--quality-config', type=Path, default=Path('training/conf/quality.yaml'))
    parser.add_argument('--skip-quality', action='store_true', help='Generate audio without GPU MOS scoring')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('--output must be empty to avoid mixing checkpoint samples')
    from kaldiio import load_scp
    import torch
    from espnet2.bin.tts_inference import Text2Speech
    from training.frontend.phonemize import UkrainianPhonemizer

    torch.manual_seed(777)
    np.random.seed(777)
    torch.set_num_threads(2)
    frame = pd.read_parquet(args.manifest).set_index('utterance_id')
    selected = select([json.loads(line) for line in args.panel.read_text().splitlines()], args.count)
    embeddings = load_scp(str(args.xvector))
    frontend = UkrainianPhonemizer()
    model = Text2Speech(train_config=args.config, model_file=args.checkpoint, device=args.device)
    if int(model.fs) != 24000:
        raise ValueError('Expected a 24 kHz model')
    audio = args.output / 'audio'
    audio.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(selected, 1):
        identifier = row['utterance_id']
        item = frame.loc[identifier]
        if not str(item['split']).endswith('_eval') or item['source_id'] != row['source_id']:
            raise ValueError('Panel and evaluation manifest disagree')
        if checksum(row['audio_path']) != row['reference_sha256'] or checksum(item['audio_path']) != item['audio_sha256']:
            raise ValueError('Reference audio hash mismatch')
        vector = np.array(embeddings[identifier], dtype=np.float32, copy=True).squeeze()
        if vector.shape != (192,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError('Invalid speaker embedding')
        text, _ = frontend.phonemize(row['text'])
        with torch.inference_mode():
            waveform = model(text, spembs=vector)['wav'].flatten().cpu().numpy()
        if not waveform.size or not np.isfinite(waveform).all() or not np.any(waveform):
            raise ValueError('Invalid generated waveform')
        if np.max(np.abs(waveform)) > 1:
            raise ValueError('Generated waveform would clip when saved as PCM')
        paths = {name: f'audio/{index:02d}_{name}.wav' for name in ('synthesis', 'reference', 'processed')}
        sf.write(args.output / paths['synthesis'], waveform, 24000, subtype='PCM_24')
        shutil.copyfile(row['audio_path'], args.output / paths['reference'])
        shutil.copyfile(item['audio_path'], args.output / paths['processed'])
        rows.append({'utterance_id': identifier, 'source_id': row['source_id'], 'text': row['text'],
                     'duration_seconds': len(waveform) / 24000, **paths,
                     'audio_sha256': {name: checksum(args.output / path) for name, path in paths.items()}})
        print(f'Generated {index}/{args.count}: {row["source_id"]}', flush=True)
    args.output.joinpath('selection.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    args.output.joinpath('report.json').write_text(json.dumps({
        'checkpoint_step': args.step, 'checkpoint_sha256': checksum(args.checkpoint),
        'config_sha256': checksum(args.config), 'sample_count': len(rows),
        'sources': sorted({r['source_id'] for r in rows}), 'seed': 777,
        'selection': 'source round-robin, fixed seeded item order',
        'speaker_conditioning': 'processed evaluation reference ECAPA',
        'postprocessing': 'none; PCM24 serialization only'}, indent=2) + '\n')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not args.skip_quality:
        from training.scripts.score_checkpoint_listening import score_listening
        score_listening(args.output, args.quality_config)
    else:
        write_pages(args.output, rows, args.step)


if __name__ == '__main__':
    main()
