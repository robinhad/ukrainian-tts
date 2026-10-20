"""Aggregate matched phone diagnostics and render a standalone listening comparison."""
import argparse
import base64
from collections import Counter, defaultdict
import html
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .convert import EMBEDDING, sha256
from .rhotic_diagnostic import RECOGNIZER, REVISION


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    return {condition: dict(files=len(group),
                           r_counts=dict(Counter(p['recognized'] for row in group
                                                 for p in row['rhotics'] if p['expected'] == 'r')),
                           l_counts=dict(Counter(p['recognized'] for row in group
                                                 for p in row['rhotics'] if p['expected'] == 'l')))
            for condition, group in groups.items()}


def report(root, panel, bundle, old_checkpoint, public):
    baseline = read(root / 'baseline_scores.jsonl')
    probes = read(root / 'probes/scores.jsonl')
    training = read(root / 'train_probes/scores.jsonl')
    manifest = read(root / 'probes/manifest.jsonl')
    import yaml
    tokens = yaml.safe_load((bundle / 'config.yaml').read_text())['token_list']
    ri, li = tokens.index('r'), tokens.index('l')
    embedding_stats = {}
    for label, path in [('initial_transfer', bundle / 'model.pth'),
                        ('transfer_43k', root / '43epoch.pth'), ('scratch_478k', old_checkpoint)]:
        weight = torch.load(path, map_location='cpu', weights_only=True)[EMBEDDING]
        embedding_stats[label] = dict(r_l_cosine=float(torch.nn.functional.cosine_similarity(weight[ri], weight[li], dim=0)),
                                     r_l_l2=float((weight[ri] - weight[li]).norm()),
                                     r_norm=float(weight[ri].norm()), checkpoint_sha256=sha256(path))
    baseline_audio = {r['id']: r for r in manifest if r['condition'] == 'baseline_43k'}
    controls = 0
    for row in manifest:
        if row['condition'].startswith('scratch') or 'r' in row['expected_phones']:
            continue
        wav, rate = sf.read(row['audio_path'])
        original, original_rate = sf.read(baseline_audio[row['id']]['audio_path'])
        assert rate == original_rate and np.array_equal(wav, original)
        controls += 1
    durations = {}
    for condition in sorted({r['condition'] for r in manifest}):
        values = [duration * 256 / row['sample_rate'] * 1000
                  for row in manifest if row['condition'] == condition
                  for phone, duration in zip(row['expected_phones'], row['durations']) if phone == 'r']
        durations[condition] = dict(count=len(values), median_ms=float(np.median(values)),
                                   min_ms=min(values), max_ms=max(values))
    result = dict(recognizer=RECOGNIZER, recognizer_revision=REVISION,
                  method='Greedy unconstrained phone recognition; token Levenshtein alignment; all conditions use identical expected phones',
                  limitation='Automatic phone labels are not human listening judgments; observed interventions do not replace a randomized retraining comparison of initializers',
                  checkpoint_history=summary(baseline), heldout_interventions=summary(probes),
                  training_interventions=summary(training), embeddings=embedding_stats,
                  r_durations=durations, exact_unchanged_no_r_waveforms=controls,
                  conclusion='Transfer checkpoint fails to distinguish r from l on this panel; both rows began as identical copies of donor L and remain almost collinear. Source audio and preprocessing retain r. Duration/reference-conditioning changes do not resolve the confusion.',
                  running_training_modified=False)
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    public.mkdir(parents=True, exist_ok=True)
    (public / 'phonetic_transfer_rhotic_diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for name, rows in [('history', baseline), ('interventions', probes), ('training', training)]:
        (public / f'phonetic_transfer_rhotic_{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

    panel_rows = read(panel / 'selection.jsonl')
    conditions = [('reference', 'Оригінал Тетяни'), ('processed', 'Оброблений оригінал'),
                  ('scratch_478k', 'Модель з нуля · 478K'), ('baseline_43k', 'Transfer · 43K'),
                  ('r_input_as_l', 'Transfer · вхідні r замінено на l'),
                  ('r_contrast_8', 'Transfer · розділення r/l ×8'),
                  ('r_duration_2', 'Transfer · тривалість r ×2'),
                  ('r_initial_english_R', 'Transfer · ембеддинг англійського R')]
    audios = {(r['id'], r['condition']): Path(r['audio_path']) for r in manifest}
    scores = {(r['id'], r['condition']): r for r in probes}
    for row in panel_rows:
        for role in ['reference', 'processed']:
            audios[row['utterance_id'], role] = panel / row[role]
    for row in baseline:
        if row['condition'] in ['reference', 'processed']:
            scores[row['id'], row['condition']] = row
    for embedded, filename in [(False, 'index.html'), (True, 'preview.html')]:
        parts = ['<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
                 '<title>Діагностика вимови р · Tetiana</title><style>body{max-width:1100px;margin:auto;padding:24px;font:18px/1.5 system-ui}article{border-top:1px solid #aaa;margin-top:32px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}audio{width:100%}small{display:block}td,th{padding:6px;text-align:left}@media(prefers-color-scheme:dark){body{background:#171717;color:#eee}}</style>',
                 '<h1>Перевірка розрізнення «р» і «л»</h1><p>Однакові фрази; незмінний фронтенд. Втручання виконані лише на копії моделі, без навчання. Аудіо без нормалізації гучності.</p>',
                 '<p>Нижче — автоматичні фонетичні мітки Wav2Vec2-XLSR, не оцінка людини. Числа стосуються 16 позицій r у 7 фразах. Підсилення різниці ембеддингів та заміна на англійський R — діагностичні тести, не готове виправлення.</p>',
                 '<table><tr><th>Варіант</th><th>Розпізнано r</th><th>Розпізнано l</th><th>Інше / пропуск</th></tr>']
        for condition, label in conditions:
            group = (result['checkpoint_history'] if condition in ['reference', 'processed'] else result['heldout_interventions'])[condition]
            c = group['r_counts']
            parts.append(f'<tr><td>{html.escape(label)}</td><td>{c.get("r",0)}</td><td>{c.get("l",0)}</td><td>{sum(c.values())-c.get("r",0)-c.get("l",0)}</td></tr>')
        parts.append('</table>')
        for row in panel_rows:
            if 'r' not in row['phonemes']:
                continue
            uid = row['utterance_id']
            parts.append(f'<article><h2>{html.escape(row["text"])}</h2><div class="players">')
            for condition, label in conditions:
                path = audios[uid, condition]
                src = 'data:audio/wav;base64,' + base64.b64encode(path.read_bytes()).decode() if embedded else os.path.relpath(path, root)
                outcomes = ', '.join(p['recognized'] for p in scores[uid, condition]['rhotics'] if p['expected'] == 'r')
                parts.append(f'<div><b>{html.escape(label)}</b><audio controls preload="none" src="{html.escape(src)}"></audio><small>Мітки в позиціях r: {html.escape(outcomes)}</small></div>')
            parts.append('</div></article>')
        parts.append('</html>')
        (root / filename).write_text('\n'.join(parts))
    print(json.dumps(result['heldout_interventions'], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'panel', 'bundle', 'old-checkpoint', 'public']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.panel, args.bundle, args.old_checkpoint, args.public)


if __name__ == '__main__':
    main()
