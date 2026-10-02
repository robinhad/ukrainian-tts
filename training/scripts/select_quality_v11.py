#!/usr/bin/env python3
"""Stream native audio, enhance on CUDA, select max SigMOS, and retain >= threshold."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf

from training.quality.common import digest, file_hash, read_rows, write_json, write_tables
from training.scripts.build_quality_v10_manifest import json_value


def choose(original, processed, threshold):
    if not all(np.isfinite(x) for x in (original, processed, threshold)):
        raise ValueError('Nonfinite selection scores')
    variant = 'processed' if processed > original else 'original'
    score = max(original, processed)
    return variant, score, score >= threshold


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=json_value) + '\n')
    temporary.replace(path)


def worker(args):
    from training.quality.processing import process
    from training.quality.backends import resample
    from training.quality.sigmos_only import SigMOSOnly
    from training.audio_enhancement.fair_comparison import atomic_write_pcm24
    rows = read_rows(args.worker_panel)
    temporary = args.worker_panel.parent / 'processed'
    process(args.worker_panel, args.profile, temporary, device='cuda:0', resume=True, cpu_workers=2)
    model = SigMOSOnly(args.sigmos_dir)
    for row in rows:
        identifier = row['utterance_id']
        native, native_rate = sf.read(row['audio_path'], dtype='float32', always_2d=True)
        candidate = temporary / f'{identifier}.wav'
        processing_metadata = json.loads(candidate.with_suffix('.json').read_text())
        enhanced, rate = sf.read(candidate, dtype='float32', always_2d=True)
        original_scores = model.score(native.mean(axis=1), native_rate)
        processed_scores = model.score(enhanced.mean(axis=1), rate)
        variant, score, retained = choose(original_scores['MOS_OVRL'], processed_scores['MOS_OVRL'], args.threshold)
        record = row['manifest_record']
        result = {'utterance_id': identifier, 'source_id': row['source_id'], 'run_key': args.run_key,
                  'source_sha256': record['audio_sha256_source'],
                  'canonical_sha256': file_hash(record['audio_path']),
                  'native_sha256': file_hash(row['audio_path']),
                  'backend_identity': processing_metadata['backend_identity'],
                  'original_scores': original_scores, 'processed_scores': processed_scores,
                  'selected_variant': variant, 'selected_score': score, 'retained': retained,
                  'original_seconds': len(native) / native_rate,
                  'processed_seconds': len(enhanced) / rate, 'sigmos_identity': model.identity}
        if retained:
            target = args.output / 'audio_24k' / f'{identifier}.wav'
            target.parent.mkdir(parents=True, exist_ok=True)
            if variant == 'processed':
                os.replace(candidate, target)
                training_scores = processed_scores
            else:
                # Preserve native timing; only convert channel count, rate and encoding.
                original_24k = resample(native.mean(axis=1), native_rate, 24000)
                atomic_write_pcm24(target, original_24k[:, None], 24000)
                actual, actual_rate = sf.read(target, dtype='float32')
                training_scores = model.score(actual, actual_rate)
            info = sf.info(target)
            result.update(output_sha256=file_hash(target), training_scores=training_scores,
                          training_duration=info.duration,
                          training_score_below_threshold=training_scores['MOS_OVRL'] < args.threshold)
            final = {**record, 'audio_path': str(target.resolve()), 'audio_sha256': result['output_sha256'],
                     'duration': info.duration, 'sample_rate': 24000, 'channels': 1, 'format': 'WAV/PCM_24',
                     'split': record['split'].replace('quality_v10_', 'quality_v11_'),
                     'processing_profile_sha256': file_hash(args.profile)}
            if variant == 'original':
                final.update(trim_applied=False, trim_start_seconds=0., trim_end_seconds=0., trim_removed_seconds=0.,
                             duration_before_trim=info.duration, trim_config_hash=None)
            result['record'] = final
        # Commit the decision only after a retained audio file is durable and hashed.
        write_json(args.output / 'decisions' / f'{identifier}.json', json.loads(json.dumps(result, default=json_value)))
    write_json(args.worker_panel.parent / 'complete.json', {'completed': len(rows)})


def finish(args, rows):
    decisions = [json.loads((args.output / 'decisions' / f"{r['utterance_id']}.json").read_text()) for r in rows]
    retained = [d for d in decisions if d['retained']]
    if not retained:
        raise ValueError('Selection retained no recordings')
    for decision in retained:
        if file_hash(decision['record']['audio_path']) != decision['output_sha256']:
            raise ValueError('Retained audio changed')
    write_rows(args.output / 'records.jsonl', [d['record'] for d in retained])
    public = [{'sample_id': hashlib.sha256(json.dumps((d['source_id'], d['utterance_id'])).encode()).hexdigest()[:16],
               **{k: d.get(k) for k in ['source_id', 'original_scores', 'processed_scores', 'selected_variant',
                    'selected_score', 'retained', 'original_seconds', 'processed_seconds',
                    'training_scores', 'training_duration', 'training_score_below_threshold']}}
              for d in decisions]
    write_tables(args.output, 'selection', public)
    groups = []
    for source in ['all', *sorted({r['source_id'] for r in rows})]:
        all_source = [d for d in decisions if source == 'all' or d['source_id'] == source]
        kept = [d for d in all_source if d['retained']]
        scores = [d['selected_score'] for d in kept]
        groups.append({'source_id': source, 'evaluated': len(all_source), 'retained': len(kept),
                       'hours': sum(d['training_duration'] for d in kept) / 3600,
                       'median_sigmos': float(np.median(scores)) if scores else None,
                       'mean_sigmos': float(np.mean(scores)) if scores else None,
                       'selected_variants': dict(Counter(d['selected_variant'] for d in kept)),
                       'rendered_below_threshold': sum(d['training_score_below_threshold'] for d in kept)})
    write_tables(args.output, 'summary', groups)
    alphabet = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
    counts = Counter(''.join(d['record']['text_sanitized'].lower() for d in retained))
    write_json(args.output / 'coverage.json', {'letters': {c: counts[c] for c in alphabet},
               'missing_letters': [c for c in alphabet if not counts[c]],
               'splits': dict(Counter(d['record']['split'] for d in retained)),
               'letters_by_split': {split: {c: sum(d['record']['text_sanitized'].lower().count(c)
                   for d in retained if d['record']['split'] == split) for c in alphabet}
                   for split in sorted({d['record']['split'] for d in retained})}})
    write_json(args.output / 'complete.json', {'status': 'complete', 'run_key': args.run_key,
               'evaluated': len(rows), 'retained': len(retained), 'hours': groups[0]['hours']})


def run(args):
    from training.quality.references import decode_native
    from training.scripts.materialize_non_voa import input_rows
    rows = read_rows(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r['utterance_id'] for r in rows}) != len(rows) or any('voa' in r['source_id'].lower() for r in rows):
        raise ValueError('Expected unique non-VOA inputs')
    if args.workers < 1 or args.chunk_size < 1 or args.limit < 0 or not 1 <= args.threshold <= 5:
        raise ValueError('Invalid settings')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'manifest': file_hash(args.manifest), 'profile': file_hash(args.profile),
                  'sigmos': file_hash(args.sigmos_dir / 'model-sigmos_1697718653_41d092e8-epo-200.onnx'),
                  'selection_code': file_hash(__file__), 'sigmos_code': file_hash(args.sigmos_dir / 'sigmos.py'),
                  'adapter_hashes': {name: file_hash(Path('training') / name) for name in [
                      'quality/sigmos_only.py', 'quality/references.py', 'quality/processing.py',
                      'quality/devices.py', 'scripts/run_enhancement_review_backend.py',
                      'scripts/run_fair_enhancement_comparison.py', 'audio_enhancement/fair_comparison.py']},
                  'threshold': args.threshold, 'limit': args.limit,
                  'rule': 'processed_if_strictly_higher_otherwise_native_original; selected_score>=threshold',
                  'split_policy': 'preserve_source_assignments', 'hnr_rejection': False}
    args.run_key = digest(provenance)
    run_path = args.output / 'run.json'
    if run_path.exists() and json.loads(run_path.read_text())['run_key'] != args.run_key:
        raise ValueError('Existing selection has different provenance')
    write_json(run_path, {**provenance, 'run_key': args.run_key})
    completed = set()
    for row in rows:
        decision = args.output / 'decisions' / f"{row['utterance_id']}.json"
        if decision.exists():
            prior = json.loads(decision.read_text())
            if prior['run_key'] != args.run_key or prior['canonical_sha256'] != file_hash(row['audio_path']):
                raise ValueError('Stale decision')
            if prior['retained'] and file_hash(prior['record']['audio_path']) != prior['output_sha256']:
                raise ValueError('Selected audio changed')
            completed.add(row['utterance_id'])
    initial = len(completed)
    active = []
    started = time.monotonic()
    last_progress = 0.
    def poll(wait=False):
        nonlocal last_progress
        while True:
            for child, stream, chunk, identifiers in list(active):
                code = child.poll()
                if code is not None:
                    stream.close()
                    active.remove((child, stream, chunk, identifiers))
                    if code:
                        raise RuntimeError(f'Selection worker failed ({code}); inspect {chunk}.log')
                    completed.update(identifiers)
                    shutil.rmtree(chunk)
            if time.monotonic() - last_progress >= 30:
                now = datetime.now(ZoneInfo('Europe/Kyiv'))
                done = len(completed) - initial
                eta = now + timedelta(seconds=(len(rows) - len(completed)) * (time.monotonic() - started) / done) if done else None
                report = {'checked_kyiv': now.isoformat(timespec='seconds'), 'completed': len(completed),
                          'expected': len(rows), 'workers': len(active),
                          'eta_kyiv': eta.isoformat(timespec='minutes') if eta else None}
                write_json(args.output / 'progress.json', report)
                print(json.dumps(report), flush=True)
                last_progress = time.monotonic()
            if not wait or len(active) < args.workers:
                break
            time.sleep(2)
    chunk_index = 0
    def submit(batch, chunk):
        panel = chunk / 'panel.jsonl'
        write_rows(panel, batch)
        stream = chunk.with_suffix('.log').open('a')
        command = [sys.executable, '-m', 'training.scripts.select_quality_v11',
                   '--worker-panel', str(panel), '--profile', str(args.profile), '--output', str(args.output),
                   '--sigmos-dir', str(args.sigmos_dir), '--threshold', str(args.threshold), '--run-key', args.run_key]
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        active.append((child, stream, chunk, [r['utterance_id'] for r in batch]))
        poll(wait=True)
    try:
        downloads = json.loads(args.downloads.read_text())
        for source in sorted({r['source_id'] for r in rows}):
            pending = {r['audio_sha256_source']: r for r in rows
                       if r['source_id'] == source and r['utterance_id'] not in completed}
            batch = []
            chunk = args.output / '.work' / f'chunk-{chunk_index}'
            for raw in input_rows(downloads[source]) if pending else []:
                poll()
                content = (raw.get('audio') or {}).get('bytes')
                if content is None:
                    continue
                key = hashlib.sha256(content).hexdigest()
                if key not in pending:
                    continue
                record = pending.pop(key)
                if shutil.disk_usage(args.output).free < 30 * 1024**3:
                    raise RuntimeError('Free disk below 30 GiB reserve')
                audio, rate = decode_native(content)
                chunk.mkdir(parents=True, exist_ok=True)
                native = chunk / f"{record['utterance_id']}.wav"
                sf.write(native, audio, rate, subtype='FLOAT')
                batch.append({'utterance_id': record['utterance_id'], 'source_id': source,
                              'audio_path': str(native.resolve()), 'reference_sha256': file_hash(native),
                              'processing_audio_path': record['audio_path'],
                              'processing_input_sha256': record['audio_sha256'], 'manifest_record': record})
                if len(batch) == args.chunk_size or not pending:
                    submit(batch, chunk)
                    batch = []
                    chunk_index += 1
                    chunk = args.output / '.work' / f'chunk-{chunk_index}'
                if not pending:
                    break
            if pending:
                raise ValueError(f'Native source coverage incomplete: {source}, {len(pending)} missing')
        while active:
            poll()
            if active:
                time.sleep(2)
        finish(args, rows)
    finally:
        for child, stream, _, _ in active:
            child.terminate()
        for child, stream, _, _ in active:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('training/data/quality_v10_raw/all.parquet'))
    parser.add_argument('--downloads', type=Path, default=Path('training/data/non_voa_downloads/downloads.json'))
    parser.add_argument('--profile', type=Path, default=Path('training/conf/quality_v11_processing.yaml'))
    parser.add_argument('--sigmos-dir', type=Path, default=Path('training/vendor/quality-models/SIG-Challenge/ICASSP2024/sigmos'))
    parser.add_argument('--output', type=Path, default=Path('training/data/quality_v11'))
    parser.add_argument('--threshold', type=float, default=3.5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--worker-panel', type=Path)
    parser.add_argument('--run-key')
    args = parser.parse_args()
    worker(args) if args.worker_panel else run(args)


if __name__ == '__main__':
    main()
