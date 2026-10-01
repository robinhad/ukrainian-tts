#!/usr/bin/env python3
"""Reconstruct canonical unenhanced PCM24 data from pinned non-VOA downloads."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import shutil
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from training.frontend.sanitize import sanitize_text
from training.quality.common import digest, file_hash, write_json
from training.quality.metrics import normalize_text
from training.scripts.prepare_audio import trim_silence


def decode(content):
    with av.open(io.BytesIO(content)) as container:
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=24000)
        frames = [part.to_ndarray().T for frame in container.decode(audio=0)
                  for part in resampler.resample(frame)]
        frames.extend(part.to_ndarray().T for part in resampler.resample(None))
    if not frames:
        raise ValueError('No audio frames')
    return np.concatenate(frames).astype(np.float32)


def convert(task):
    source_id, source, raw, output = task
    audio_object = raw.get('audio') or {}
    content = audio_object.get('bytes')
    path = str(audio_object.get('path') or 'unknown')
    if content is None:
        return None, 'missing_audio'
    text = next((str(raw[k]).strip() for k in ('transcription', 'transcribed_text', 'text', 'raw_transcription')
                 if raw.get(k) is not None and str(raw[k]).strip()), '')
    text = sanitize_text(text)
    if not normalize_text(text) or not any(c.isalpha() for c in text) or len(text) > 500:
        return None, 'invalid_text'
    original_hash = hashlib.sha256(content).hexdigest()
    identifier = f'{source_id}_{original_hash[:24]}'
    target = output / 'audio_24k' / f'{identifier}.wav'
    metadata = output / 'records' / f'{identifier}.json'
    if metadata.is_file() and target.is_file():
        row = json.loads(metadata.read_text())
        if file_hash(target) == row['audio_sha256']:
            return row, None
    try:
        audio = decode(content)
        audio, trim = trim_silence(audio, 24000)
        duration = len(audio) / 24000
        if not 2 <= duration <= 20 or not np.isfinite(audio).all() or not np.any(audio):
            return None, 'duration_or_invalid_audio'
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp.wav')
        sf.write(temporary, audio, 24000, subtype='PCM_24')
        temporary.replace(target)
        speaker = next((str(raw[k]) for k in ('speaker_id', 'client_id', 'speaker', 'speaker_name')
                        if raw.get(k) is not None), None)
        if source_id.startswith('opentts_'):
            speaker = source_id.removeprefix('opentts_')
        normalized = normalize_text(text)
        # Global text groups cannot cross splits, even across source datasets.
        bucket = int(digest(['non-voa-v10', normalized])[:8], 16) % 10000
        split = 'eval' if bucket < 200 else 'dev' if bucket < 400 else 'train'
        row = {'utterance_id': identifier, 'source_id': source_id,
               'speaker_id': speaker or identifier, 'speaker_stratum_id': speaker or identifier,
               'audio_path': str(target.resolve()), 'canonical_raw_audio_path': str(target.resolve()),
               'text_raw': text, 'text_sanitized': text,
               'source': f"hf://datasets/{source['repo_id']}@{source['revision']}",
               'source_license': source['license'], 'source_audio_path': path,
               'source_group': digest(normalized), 'source_original_split': raw.get('_split', 'unknown'),
               'label_kind': 'human', 'audio_sha256_source': original_hash,
               'audio_sha256': file_hash(target), 'text_sha256_source': digest(normalized),
               'split': f'quality_v10_{split}', 'duration': duration, 'sample_rate': 24000,
               'channels': 1, 'format': 'WAV/PCM_24', 'qc_flags': [], **trim}
        write_json(metadata, row)
        return row, None
    except Exception as error:
        return None, f'{type(error).__name__}: {error}'


def input_rows(source):
    root = Path(source['snapshot'])
    if 'dataset.csv' in source['files']:
        with (root / 'dataset.csv').open() as stream:
            for row in csv.DictReader(stream):
                filename = f"clips/{row['filename']}"
                yield {**row, 'audio': {'bytes': (root / filename).read_bytes(), 'path': filename}}
    else:
        for name in source['files']:
            for batch in pq.ParquetFile(root / name).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    row['_split'] = name
                    yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, default=Path('training/data/non_voa_downloads/downloads.json'))
    parser.add_argument('--output', type=Path, default=Path('training/data/quality_v10_raw'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    downloads = json.loads(args.downloads.read_text())
    if len(downloads) != 8 or any('voa' in name for name in downloads):
        raise ValueError('Wait for all eight non-VOA sources to finish downloading')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, rejected, seen_audio, seen_text = [], Counter(), set(), set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for source_id, source in downloads.items():
            iterator = iter(input_rows(source))
            while True:
                # Bound in-flight audio bytes. Executor.map over a corpus is not bounded.
                batch = []
                for _ in range(args.workers * 2):
                    row = next(iterator, None)
                    if row is None:
                        break
                    batch.append((source_id, source, row, args.output))
                if not batch:
                    break
                if shutil.disk_usage(args.output).free < 30 * 1024**3:
                    raise RuntimeError('Disk reserve below 30 GiB')
                for row, error in pool.map(convert, batch):
                    if error:
                        rejected[error] += 1
                        continue
                    ah, th = row['audio_sha256_source'], row['text_sha256_source']
                    if ah in seen_audio or th in seen_text:
                        rejected['duplicate_audio_or_normalized_text'] += 1
                        continue
                    seen_audio.add(ah)
                    seen_text.add(th)
                    rows.append(row)
                write_json(args.output / 'progress.json', {'source': source_id, 'records': len(rows),
                                                          'rejected': dict(rejected)})
            print(json.dumps({'source': source_id, 'total_records': len(rows)}), flush=True)
    import pandas as pd
    frame = pd.DataFrame(rows).sort_values('utterance_id')
    frame.to_parquet(args.output / 'all.parquet', index=False)
    with (args.output / 'canonical_records.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'records': len(rows), 'hours': float(frame.duration.sum() / 3600),
              'sources': {name: {'records': len(part), 'hours': float(part.duration.sum() / 3600)}
                          for name, part in frame.groupby('source_id')},
              'splits': frame['split'].value_counts().to_dict(), 'rejected': dict(rejected),
              'reconstruction': True, 'voa_records': 0, 'format': '24 kHz mono PCM24',
              'split_policy': 'global normalized-text hash, 96/2/2 train/dev/eval; duplicate text/audio removed'}
    write_json(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
