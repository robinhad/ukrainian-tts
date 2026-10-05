"""Prepare a >=4.0 processed-SigMOS training subset with unchanged dev/eval data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from training.quality.common import digest, file_hash, write_json


def select_training(frame, decisions, threshold=4.0):
    selected = []
    for row in frame.itertuples():
        decision = json.loads((decisions / f'{row.utterance_id}.json').read_text())
        if decision['utterance_id'] != row.utterance_id or decision['output_sha256'] != row.audio_sha256:
            raise ValueError('Score is not bound to the training waveform')
        score = decision['processed_scores']['MOS_OVRL']
        if not np.isfinite(score) or decision['heldout']:
            raise ValueError('Invalid training score or held-out item in training')
        if score >= threshold:
            selected.append({'utterance_id': row.utterance_id, 'source_id': row.source_id,
                             'duration_seconds': float(row.duration), 'sigmos_overall': float(score),
                             'audio_sha256': row.audio_sha256})
    if not selected:
        raise ValueError('No training recordings meet the threshold')
    return sorted(selected, key=lambda row: row['utterance_id'])


def filter_keyed(source, destination, ids):
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    keys = [line.split(maxsplit=1)[0] for line in lines]
    if len(keys) != len(set(keys)) or not ids <= set(keys):
        raise ValueError(f'Missing or duplicate data keys in {source.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(line for line in lines if line.split(maxsplit=1)[0] in ids) + '\n')


def prepare(root, preview_only=False):
    source_data = root / 'data/quality_v12'
    data = root / 'data/quality_v13'
    source_dump, dump = root / 'dump_quality_v12', root / 'dump_quality_v13'
    stats_name = 'tts_stats_raw_phn_espeak_ng_ukrainian'
    source_stats = root / 'exp_quality_v12' / stats_name
    stats = root / 'exp_quality_v13' / stats_name
    run = root / 'quality_runs/v13'
    frames = {split: pd.read_parquet(source_data / 'manifests' / f'quality_v12_{split}.parquet')
              for split in ('train', 'dev', 'eval')}
    selected = select_training(frames['train'], source_data / 'decisions')
    keep = {r['utterance_id'] for r in selected}
    counts = Counter(r['source_id'] for r in selected)
    report = {'threshold': 4.0, 'metric': 'processed_sigmos_overall', 'inclusive': True,
              'training_before': len(frames['train']), 'training_after': len(selected),
              'training_hours': sum(r['duration_seconds'] for r in selected) / 3600,
              'sources': dict(sorted(counts.items())), 'dev_items_unchanged': len(frames['dev']),
              'eval_items_unchanged': len(frames['eval']), 'selected': selected,
              'start_step': 284000, 'end_step': 334000, 'additional_updates': 50000,
              'normalization': 'preserve V12 statistics and token vocabulary',
              'selection_sha256': digest(selected)}
    if preview_only:
        return report
    frames['train'] = frames['train'][frames['train'].utterance_id.isin(keep)].copy()
    ids = {split: set(frame.utterance_id) for split, frame in frames.items()}
    if any(ids[a] & ids[b] for a, b in [('train', 'dev'), ('train', 'eval'), ('dev', 'eval')]):
        raise ValueError('Dataset splits overlap')
    source_binding = file_hash(root / 'quality_runs/v12/prepared.json')
    seal = run / 'prepared.json'
    if seal.exists():
        previous = json.loads(seal.read_text())
        if previous['selection_sha256'] != report['selection_sha256'] or previous['source_binding'] != source_binding:
            raise ValueError('Filtered dataset source or selection changed')
        if any(file_hash(root / name) != sha for name, sha in previous['files'].items()):
            raise ValueError('Filtered prepared artifacts changed')
        return report
    owned = []
    manifests = data / 'manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    # Preserve split names for compatibility with the existing evaluator; roots are isolated.
    for split, frame in frames.items():
        path = manifests / f'quality_v12_{split}.parquet'
        frame.to_parquet(path, index=False); owned.append(path)
        name = f'quality_v12_{split}'
        raw = dump / 'raw' / name
        for filename in ('wav.scp', 'text', 'utt2spk', 'utt2num_samples'):
            filter_keyed(source_dump / 'raw' / name / filename, raw / filename, ids[split])
            owned.append(raw / filename)
        shutil.copyfile(source_dump / 'raw' / name / 'feats_type', raw / 'feats_type')
        owned.append(raw / 'feats_type')
        speakers = defaultdict(list)
        for line in (raw / 'utt2spk').read_text().splitlines():
            utterance, speaker = line.split(); speakers[speaker].append(utterance)
        (raw / 'spk2utt').write_text(''.join(f'{speaker} {" ".join(sorted(utterances))}\n'
                                          for speaker, utterances in sorted(speakers.items())))
        owned.append(raw / 'spk2utt')
        path = dump / 'xvector' / name / 'xvector.scp'
        filter_keyed(source_dump / 'xvector' / name / 'xvector.scp', path, ids[split]); owned.append(path)
    path = manifests / 'all.parquet'
    pd.concat(frames.values(), ignore_index=True).to_parquet(path, index=False); owned.append(path)
    tokens = dump / 'token_list/phn_espeak_ng_ukrainian/tokens.txt'
    tokens.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_dump / 'token_list/phn_espeak_ng_ukrainian/tokens.txt', tokens); owned.append(tokens)
    for phase, split in [('train', 'train'), ('valid', 'dev')]:
        target = stats / phase; target.mkdir(parents=True, exist_ok=True)
        for path in (source_stats / phase).glob('*.npz'):
            shutil.copyfile(path, target / path.name); owned.append(target / path.name)
        for filename in ('speech_shape', 'text_shape', 'text_shape.phn', 'spembs_shape'):
            filter_keyed(source_stats / phase / filename, target / filename, ids[split]); owned.append(target / filename)
        for path in (source_stats / phase / 'collect_feats').glob('*.scp'):
            destination = target / 'collect_feats' / path.name
            filter_keyed(path, destination, ids[split]); owned.append(destination)
    write_json(run / 'selection.json', report)
    owned.append(run / 'selection.json')
    write_json(seal, {'selection_sha256': report['selection_sha256'], 'source_binding': source_binding,
                     'files': {str(path.relative_to(root)): file_hash(path) for path in sorted(owned)}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = prepare(Path('training'), args.preview_only)
    if args.report:
        write_json(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'selected'}))


if __name__ == '__main__':
    main()
