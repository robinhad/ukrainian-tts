"""Build an offline before/after listening page from completed A/B evaluations."""
import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .convert import sha256


def indexed(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {(r['id'], r['condition']): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate file/condition in {path.name}')
    return result


def render(root, step):
    record = json.loads((root / 'experiment.json').read_text())
    selection = pd.read_parquet(Path(record['voice_data']) / 'dev/selection.parquet')
    texts = dict(zip(selection.utterance_id, selection.text_sanitized))
    stages = {}
    for stage in [0, step]:
        directory = root / f'evaluation_{stage}'
        stages[stage] = {name: indexed(directory / f'{name}.jsonl')
                         for name in ['manifest', 'quality', 'phones']}
        if not all(rows.keys() == stages[stage]['manifest'].keys()
                   for rows in stages[stage].values()):
            raise ValueError('Evaluation metric IDs differ from audio IDs')
    if stages[0]['manifest'].keys() != stages[step]['manifest'].keys():
        raise ValueError('Baseline and final evaluation selections differ')
    ids = sorted(uid for uid, role in stages[0]['manifest'] if role == 'synthesis')
    if set(ids) != set(texts):
        raise ValueError('Evaluation selection differs from development transcripts')
    for uid in ids:
        a, b = [stages[n]['manifest'][uid, 'reference'] for n in [0, step]]
        if a['audio_sha256'] != b['audio_sha256']:
            raise ValueError('Reference audio changed between evaluations')
        if stages[0]['manifest'][uid, 'synthesis']['expected_phones'] != stages[step]['manifest'][uid, 'synthesis']['expected_phones']:
            raise ValueError('Synthesis frontend tokens changed')
    conditions = [(step, 'reference', 'Оброблений оригінал'),
                  (0, 'synthesis', f'До · {record["base_step"]:,} кроків'),
                  (step, 'synthesis', f'Після · +{step:,} кроків')]
    final = json.loads((root / f'evaluation_{step}/summary.json').read_text())
    history = [json.loads(p.read_text()) for p in sorted(root.glob('evaluation_*/summary.json'),
               key=lambda p: int(p.parent.name.split('_')[-1]))
               if int(p.parent.name.split('_')[-1]) <= step]
    arm = html.escape(record['arm'])
    parts = ['<!doctype html><html lang="uk"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             f'<title>Перевірка р · {arm} · +{step}</title>',
             '<style>body{max-width:1200px;margin:auto;padding:24px;font:17px/1.5 system-ui;color:#222;background:#fff}h1{font-size:1.7rem}article{border-top:1px solid #aaa;margin-top:28px;padding-top:12px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:22px}audio{width:100%}small{display:block;overflow-wrap:anywhere}table{border-collapse:collapse}th,td{padding:8px 14px;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}@media(prefers-color-scheme:dark){body{background:#171717;color:#eee}}</style>',
             f'<h1>Вимова «р»: {arm}, +{step:,} кроків</h1>',
             f'<p>{len(ids)} однакових validation-фраз, голос Тетяни. Порівняння обробленого оригіналу, початкового чекпоінта та результату донавчання. Аудіо вбудоване у сторінку; додаткова нормалізація гучності не застосована.</p>',
             '<p>Ембеддинги: LR 0.0002; решта моделі: LR 0.00002. Фонетичні мітки — автоматична діагностика, не оцінка людини. SigMOS сам по собі не визначає правильність вимови.</p>',
             '<div class="scroll"><table><tr><th>Додаткові кроки</th><th>Розпізнано r</th><th>r → l</th><th>Медіана SigMOS</th><th>Медіана Audiobox PQ</th></tr>']
    for row in history + [dict(additional_steps='Оброблений оригінал',
                               phone_counts={'synthesis': final['phone_counts']['reference']},
                               metrics={k.replace('reference_', 'synthesis_'): v
                                        for k, v in final['metrics'].items() if k.startswith('reference_')})]:
        counts = row['phone_counts']['synthesis']['r_counts']
        n = sum(counts.values())
        m = row['metrics']
        parts.append(f'<tr><td>{row["additional_steps"]}</td><td>{counts.get("r", 0)}/{n}</td><td>{counts.get("l", 0)}/{n}</td><td>{m["synthesis_sigmos_overall_median"]:.3f}</td><td>{m["synthesis_audiobox_pq_median"]:.3f}</td></tr>')
    parts.append('</table></div>')
    players = 0
    for index, uid in enumerate(ids, 1):
        parts.append(f'<article><h2>{index}. {html.escape(texts[uid])}</h2><div class="players">')
        for stage, role, label in conditions:
            data = stages[stage]
            row = data['manifest'][uid, role]
            path = Path(row['audio_path'])
            if sha256(path) != row['audio_sha256'] or data['phones'][uid, role]['audio_sha256'] != row['audio_sha256']:
                raise ValueError('Audio or recognition provenance mismatch')
            audio, _ = sf.read(path)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError('Invalid listening audio')
            m = data['quality'][uid, role]
            phones = [p['recognized'] for p in data['phones'][uid, role]['rhotics'] if p['expected'] == 'r']
            encoded = base64.b64encode(path.read_bytes()).decode()
            parts.append(f'<section><h3>{html.escape(label)}</h3><audio controls preload="none" aria-label="{html.escape(label, quote=True)}" src="data:audio/wav;base64,{encoded}"></audio><small>SigMOS {m["sigmos_overall"]:.3f} · Audiobox PQ {m["audiobox_pq"]:.3f}</small><small>Очікуване r → {html.escape(" · ".join(phones) or "немає r")}</small></section>')
            players += 1
        parts.append('</div></article>')
    parts.append('<script>document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>')
    output = root / f'listening_{step}'
    output.mkdir(exist_ok=True)
    (output / 'preview.html').write_text('\n'.join(parts))
    result = dict(arm=record['arm'], additional_steps=step, base_step=record['base_step'],
                  files=len(ids), players=players, history=history,
                  final_checkpoint_sha256=final['checkpoint_sha256'])
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(page=str(output / 'preview.html'), files=len(ids), players=players)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--step', type=int, default=2000, choices=range(500, 2001, 500))
    args = parser.parse_args()
    render(args.root, args.step)


if __name__ == '__main__':
    main()
