"""Stream the full non-VOA corpus through MFA, the fixed cascade, and SigMOS >=3.5."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf

from training.audio_enhancement.fair_comparison import atomic_write_pcm24
from training.frontend.sanitize import sanitize_text
from training.quality.common import digest, file_hash, read_rows, write_json, write_tables
from training.quality.mfa_trim import boundaries
from training.scripts.select_quality_v11 import write_rows
from training.scripts.build_quality_v10_manifest import json_value


def transcript(text):
    words = re.findall(r"[\w']+", sanitize_text(text).lower().replace('\u0301', ''), flags=re.UNICODE)
    words = [w.strip("'") for w in words if w.strip("'")]
    unsupported = [w for w in words if not re.fullmatch("[а-щьюяєіїґ']+", w)]
    return ' '.join(words), unsupported


def retention(score, split, threshold=3.5):
    if not np.isfinite(score) or not 1 <= score <= 5 or not 1 <= threshold <= 5:
        raise ValueError('Invalid SigMOS score or threshold')
    passed = score >= threshold
    return passed, passed or split.endswith('_eval')


def mfa_environment(args, work):
    env = dict(os.environ)
    mfa_env, mfa_tools = args.mfa_env.resolve(), args.mfa_tools.resolve()
    env['PATH'] = f'{mfa_env}/bin:{mfa_tools}/sysroot/usr/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = f'{mfa_tools}/sysroot/usr/lib/aarch64-linux-gnu:{mfa_env}/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['MFA_ROOT_DIR'] = str(work.resolve())
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def mfa_command(args, command, work, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([str(args.mfa_env / 'bin/mfa'), *map(str, command)],
                       env=mfa_environment(args, work), stdout=stream, stderr=subprocess.STDOUT, check=True)


def dictionary(args, rows):
    root = args.output / 'mfa'
    root.mkdir(parents=True, exist_ok=True)
    assets = json.loads(Path('training/conf/quality_mfa_assets.json').read_text())
    for name, filename in [('dictionary', 'ukrainian_mfa_dictionary.dict'),
                           ('acoustic', 'ukrainian_mfa_acoustic.zip'), ('g2p', 'ukrainian_mfa_g2p.zip')]:
        if file_hash(args.mfa_models / filename) != assets[name]['sha256']:
            raise ValueError('Pinned MFA asset changed')
    base = args.mfa_models / 'ukrainian_mfa_dictionary.dict'
    vocabulary = {w for row in rows for w in transcript(row['text_sanitized'])[0].split()
                  if re.fullmatch("[а-щьюяєіїґ']+", w)}
    known = {line.split()[0] for line in base.read_text().splitlines() if line.strip()}
    oovs = sorted(vocabulary - known)
    key = digest([file_hash(base), assets['g2p']['sha256'], oovs])
    marker, expanded = root / 'dictionary.json', root / 'expanded.dict'
    if marker.exists() and expanded.exists():
        old = json.loads(marker.read_text())
        if old['key'] == key and old['sha256'] == file_hash(expanded):
            return expanded
    words, generated = root / 'oovs.txt', root / 'generated.dict'
    words.write_text('\n'.join(oovs) + '\n')
    if oovs:
        mfa_command(args, ['g2p', words, args.mfa_models / 'ukrainian_mfa_g2p.zip', generated,
                          '--num_jobs', args.mfa_workers, '--num_pronunciations', '1', '--clean'],
                    root / 'g2p_work', root / 'g2p.log')
    expanded.write_text(base.read_text().rstrip() + '\n' + (generated.read_text() if oovs else ''))
    write_json(marker, {'key': key, 'sha256': file_hash(expanded), 'vocabulary': len(vocabulary), 'oovs': len(oovs)})
    return expanded


def source_batches(args, rows, completed):
    from training.scripts.materialize_non_voa import input_rows
    downloads = json.loads(args.downloads.read_text())
    for source in sorted({r['source_id'] for r in rows}):
        if all(r['utterance_id'] in completed for r in rows if r['source_id'] == source):
            continue
        pending = {r['audio_sha256_source']: r for r in rows if r['source_id'] == source}
        if len(pending) != sum(r['source_id'] == source for r in rows):
            raise ValueError('Duplicate source hashes')
        batches = {}
        for raw in input_rows(downloads[source]) if pending else []:
            content = (raw.get('audio') or {}).get('bytes')
            if content is None:
                continue
            key = hashlib.sha256(content).hexdigest()
            if key not in pending:
                continue
            record = pending.pop(key)
            batch = batches.setdefault(record['split'], [])
            batch.append((record, content))
            if len(batch) == args.alignment_chunk_size:
                # Freeze alignment context across resumes and keep split-specific
                # speaker adaptation separate. Only fully completed chunks skip.
                if any(r['utterance_id'] not in completed for r, _ in batch):
                    yield batch
                batches[record['split']] = []
            if not pending:
                break
        if pending:
            raise ValueError(f'Native source coverage missing: {source}, {len(pending)} files')
        for split in sorted(batches):
            batch = batches[split]
            if batch and any(r['utterance_id'] not in completed for r, _ in batch):
                yield batch


def prepare_batch(args, batch, expanded, pilot):
    from training.quality.references import decode_native
    from training.scripts.materialize_non_voa import decode
    key = digest([args.run_key, [r['utterance_id'] for r, _ in batch]])[:20]
    chunk = args.output / '.work' / key
    chunk.mkdir(parents=True, exist_ok=True)
    marker = chunk / 'ready.json'
    if marker.exists():
        rows = read_rows(chunk / 'panel.jsonl')
        if (json.loads(marker.read_text())['run_key'] == args.run_key
                and {r['utterance_id'] for r in rows} == {r['utterance_id'] for r, _ in batch}
                and all(file_hash(r['processing_audio_path']) == r['processing_input_sha256'] for r in rows)):
            return chunk
        raise ValueError('Stale MFA chunk')
    if shutil.disk_usage(args.output).free < args.minimum_disk_gib * 1024**3:
        raise RuntimeError('Disk reserve reached before MFA preparation')
    def decode_item(pair):
        record, content = pair
        identifier = record['utterance_id']
        sample = hashlib.sha256(json.dumps((record['source_id'], identifier)).encode()).hexdigest()[:16]
        audio, rate = decode_native(content)
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format='WAV', subtype='FLOAT')
        native = buffer.getvalue()
        decoded = decode(native)
        target = chunk / 'raw' / (identifier + '.wav')
        atomic_write_pcm24(target, decoded, 24000)
        input_hash = file_hash(target)
        text, unsupported = transcript(record['text_sanitized'])
        old = pilot.get(identifier)
        if old and input_hash != old['alignment_audio_sha256']:
            raise ValueError('Full-corpus decoding differs from the frozen MFA pilot')
        if not old:
            speaker = digest([record['source_id'], record['speaker_id']])[:16]
            corpus = chunk / 'corpus' / speaker / (sample + '.wav')
            corpus.parent.mkdir(parents=True, exist_ok=True)
            if not corpus.exists():
                os.link(target, corpus)
            corpus.with_suffix('.lab').write_text(text + '\n')
        return {'utterance_id': identifier, 'source_id': record['source_id'], 'sample_id': sample,
                'audio_path': str(target.resolve()), 'reference_sha256': input_hash,
                'native_reference_sha256': hashlib.sha256(native).hexdigest(),
                'alignment_text': text, 'unsupported_tokens': unsupported,
                'pilot_alignment': bool(old), 'manifest_record': record}
    with ThreadPoolExecutor(max_workers=args.mfa_workers) as pool:
        rows = list(pool.map(decode_item, batch))
    if any(not r['pilot_alignment'] for r in rows):
        mfa_command(args, ['align', chunk / 'corpus', expanded,
                          args.mfa_models / 'ukrainian_mfa_acoustic.zip', chunk / 'aligned',
                          '--config_path', 'training/conf/quality_mfa_alignment.yaml',
                          '--output_format', 'json', '--num_jobs', args.mfa_workers,
                          '--no_textgrid_cleanup', '--clean'], chunk / 'mfa_work', args.output / 'logs' / (key + '-mfa.log'))
    grids = {p.stem: p for p in (chunk / 'aligned').rglob('*.json')}
    audits = []
    for row in rows:
        source = Path(row['audio_path'])
        audio, rate = sf.read(source, dtype='float32', always_2d=True)
        old = pilot.get(row['utterance_id'])
        grid = Path(old['grid_path']) if old else grids.get(row['sample_id'])
        alignment = json.loads(grid.read_text()) if grid else {}
        duration = len(audio) / rate
        start, end, flags = boundaries(alignment, duration)
        if row['unsupported_tokens']:
            flags.append('unsupported_transcript_tokens')
        if grid and abs(alignment['end'] - duration) > .03:
            flags.append('alignment_duration_mismatch')
        if flags:
            start, end = 0., duration
        begin, finish = max(0, math.floor(start * rate)), min(len(audio), math.ceil(end * rate))
        target = chunk / 'inputs' / (row['utterance_id'] + '.wav')
        atomic_write_pcm24(target, audio[begin:finish], rate)
        status = 'review_preserved_untrimmed' if flags else 'mfa_aligned'
        audit = {'utterance_id': row['utterance_id'], 'source_id': row['source_id'],
                 'input_sha256': row['reference_sha256'], 'alignment_sha256': file_hash(grid) if grid else None,
                 'start_frame': begin, 'end_frame': finish, 'input_frames': len(audio), 'output_frames': finish - begin,
                 'padding_ms': 100, 'max_removed_fraction': .5, 'status': status, 'flags': sorted(set(flags)),
                 'pilot_alignment_reused': bool(old)}
        row.update(processing_audio_path=str(target.resolve()), processing_input_sha256=file_hash(target),
                   boundary_method='mfa', mfa_status=status, mfa_flags=audit['flags'], mfa_audit=audit)
        if old and row['processing_input_sha256'] != old['processing_input_sha256']:
            raise ValueError('MFA crop differs from the frozen pilot')
        audits.append(audit)
    write_rows(chunk / 'panel.jsonl', rows)
    write_tables(args.output / 'alignments', key, audits)
    write_json(marker, {'run_key': args.run_key, 'count': len(rows)})
    return chunk


def process_batch(args, chunk, scorer):
    from training.quality.parallel_processing import process_parallel
    rows = read_rows(chunk / 'panel.jsonl')
    rows = [r for r in rows if not (args.output / 'decisions' / (r['utterance_id'] + '.json')).exists()]
    write_rows(chunk / 'pending.jsonl', rows)
    destination = chunk / 'processed'
    if len(rows) == 1:
        from training.quality.processing import process
        metadata = process(chunk / 'pending.jsonl', args.profile, destination, resume=True, cpu_workers=2)
    else:
        metadata = process_parallel(chunk / 'pending.jsonl', args.profile, destination, resume=True,
                                    cpu_workers=2, model_workers=min(args.workers, len(rows)),
                                    chunk_size=args.chunk_size, minimum_available_gib=24)
    by_id = {r['utterance_id']: r for r in metadata}
    for row in rows:
        identifier = row['utterance_id']
        path = destination / (identifier + '.wav')
        meta = by_id[identifier]
        if file_hash(path) != meta['output_sha256'] or meta['input_sha256'] != row['processing_input_sha256']:
            raise ValueError('Processed output provenance changed')
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if rate != 24000 or audio.shape[1] != 1 or len(audio) != row['mfa_audit']['output_frames']:
            raise ValueError('Processing changed MFA frame count')
        scores = scorer.score(audio[:, 0], rate)
        record = row['manifest_record']
        passed, keep = retention(scores['MOS_OVRL'], record['split'], args.threshold)
        result = {'utterance_id': identifier, 'source_id': row['source_id'], 'sample_id': row['sample_id'],
                  'run_key': args.run_key, 'source_sha256': record['audio_sha256_source'],
                  'native_reference_sha256': row['native_reference_sha256'],
                  'mfa': row['mfa_audit'], 'processing': meta, 'processed_scores': scores,
                  'sigmos_identity': scorer.identity, 'passed_threshold': passed,
                  'retained': keep, 'heldout': record['split'].endswith('_eval'),
                  'duration_seconds': len(audio) / rate, 'output_sha256': meta['output_sha256']}
        if keep:
            target = args.output / 'audio_24k' / (identifier + '.wav')
            target.parent.mkdir(exist_ok=True)
            os.replace(path, target)
            audit = row['mfa_audit']
            final = {**record, 'audio_path': str(target.resolve()), 'canonical_raw_audio_path': str(target.resolve()),
                     'audio_sha256': meta['output_sha256'], 'duration': len(audio) / rate,
                     'sample_rate': rate, 'channels': 1, 'format': 'WAV/PCM_24',
                     'split': record['split'].replace('quality_v10_', 'quality_v12_'),
                     'boundary_method': 'mfa', 'mfa_status': audit['status'], 'mfa_flags': audit['flags'],
                     'trim_applied': audit['output_frames'] != audit['input_frames'],
                     'duration_before_trim': audit['input_frames'] / rate,
                     'trim_start_seconds': audit['start_frame'] / rate,
                     'trim_end_seconds': (audit['input_frames'] - audit['end_frame']) / rate,
                     'trim_removed_seconds': (audit['input_frames'] - audit['output_frames']) / rate,
                     'trim_config_hash': digest(['mfa', .1, .5]), 'trim_threshold_rms': None,
                     'processing_profile_sha256': file_hash(args.profile), 'sigmos_overall': scores['MOS_OVRL']}
            result['record'] = final
        write_json(args.output / 'decisions' / (identifier + '.json'), json.loads(json.dumps(result, default=json_value)))
    shutil.rmtree(chunk)


def finish(args, rows):
    decisions = [json.loads((args.output / 'decisions' / (r['utterance_id'] + '.json')).read_text()) for r in rows]
    records = []
    for d in decisions:
        if d['run_key'] != args.run_key:
            raise ValueError('Stale decision at final audit')
        passed, keep = retention(d['processed_scores']['MOS_OVRL'], 'quality_v10_eval' if d['heldout'] else 'quality_v10_train', args.threshold)
        if (passed, keep) != (d['passed_threshold'], d['retained']):
            raise ValueError('Retention policy mismatch')
        if keep:
            if file_hash(d['record']['audio_path']) != d['output_sha256']:
                raise ValueError('Retained audio changed')
            records.append(d['record'])
    if not all(any(r['split'] == 'quality_v12_' + split for r in records) for split in ['train', 'dev', 'eval']):
        raise ValueError('A required split is empty')
    write_rows(args.output / 'records.jsonl', records)
    summaries = []
    for source in ['all'] + sorted({r['source_id'] for r in rows}):
        items = [d for d in decisions if source == 'all' or d['source_id'] == source]
        kept = [d for d in items if d['passed_threshold'] and not d['heldout']]
        summaries.append({'source_id': source, 'evaluated': len(items), 'train_dev_retained': len(kept),
                          'train_dev_hours': sum(d['duration_seconds'] for d in kept) / 3600,
                          'heldout_preserved': sum(d['heldout'] for d in items),
                          'median_retained_sigmos': float(np.median([d['processed_scores']['MOS_OVRL'] for d in kept])) if kept else None,
                          'alignment_status': dict(Counter(d['mfa']['status'] for d in items))})
    write_tables(args.output, 'summary', summaries)
    write_tables(args.output, 'scores', [{'sample_id': d['sample_id'], 'source_id': d['source_id'],
                 'sigmos_overall': d['processed_scores']['MOS_OVRL'], 'processed_scores': d['processed_scores'],
                 'passed_threshold': d['passed_threshold'], 'heldout': d['heldout'],
                 'duration_seconds': d['duration_seconds'], 'mfa_status': d['mfa']['status']} for d in decisions])
    alphabet = 'абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'
    letters = Counter(''.join(r['text_sanitized'].lower() for r in records if r['split'].endswith('_train')))
    write_json(args.output / 'coverage.json', {'training_letters': {c: letters[c] for c in alphabet},
               'missing_training_letters': [c for c in alphabet if not letters[c]],
               'splits': dict(Counter(r['split'] for r in records))})
    write_json(args.output / 'complete.json', {'status': 'complete', 'run_key': args.run_key,
               'evaluated': len(rows), 'retained_with_heldout': len(records), 'train_dev_hours': summaries[0]['train_dev_hours']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('training/data/quality_v10_raw/all.parquet'))
    parser.add_argument('--downloads', type=Path, default=Path('training/data/non_voa_downloads/downloads.json'))
    parser.add_argument('--output', type=Path, default=Path('training/data/quality_v12'))
    parser.add_argument('--profile', type=Path, default=Path('training/conf/quality_v12_processing.yaml'))
    parser.add_argument('--mfa-models', type=Path, default=Path('training/quality_runs/mfa_trim/models'))
    parser.add_argument('--mfa-tools', type=Path, default=Path('training/quality_runs/mfa_trim/tools'))
    parser.add_argument('--mfa-env', type=Path, default=Path('training/quality_runs/mfa_trim/tools/env'))
    parser.add_argument('--sigmos-dir', type=Path, default=Path('training/vendor/quality-models/SIG-Challenge/ICASSP2024/sigmos'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--mfa-workers', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--alignment-chunk-size', type=int, default=2048)
    parser.add_argument('--minimum-disk-gib', type=float, default=30)
    parser.add_argument('--threshold', type=float, default=3.5)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--skip-finalize', action='store_true', help='For a bounded smoke run only')
    args = parser.parse_args()
    if min(args.workers, args.mfa_workers, args.chunk_size, args.alignment_chunk_size) < 1 or args.limit < 0:
        parser.error('Invalid chunk or worker settings')
    rows = read_rows(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r['utterance_id'] for r in rows}) != len(rows) or any('voa' in r['source_id'].lower() for r in rows):
        raise ValueError('Expected unique non-VOA manifest')
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / '.processing.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    provenance = {'manifest': file_hash(args.manifest), 'profile': file_hash(args.profile),
                  'code': file_hash(__file__), 'mfa_assets': file_hash('training/conf/quality_mfa_assets.json'),
                  'alignment_config': file_hash('training/conf/quality_mfa_alignment.yaml'),
                  'adapter_hashes': {name: file_hash(Path('training') / name) for name in [
                      'quality/processing.py', 'quality/parallel_processing.py', 'quality/mfa_trim.py',
                      'quality/references.py', 'quality/sigmos_only.py', 'scripts/materialize_non_voa.py']},
                  'threshold': args.threshold, 'limit': args.limit, 'alignment_chunk_size': args.alignment_chunk_size,
                  'policy': 'fixed_processed_audio_only; inclusive_SigMOS_threshold_train_dev; preserve_all_eval',
                  'hnr_rejection': False, 'pilot_alignment_reuse': True}
    args.run_key = digest(provenance)
    run_path = args.output / 'run.json'
    if run_path.exists() and json.loads(run_path.read_text())['run_key'] != args.run_key:
        raise ValueError('Existing run has different provenance')
    write_json(run_path, {**provenance, 'run_key': args.run_key})
    completed = set()
    for row in rows:
        path = args.output / 'decisions' / (row['utterance_id'] + '.json')
        if path.exists():
            d = json.loads(path.read_text())
            if d['run_key'] != args.run_key or d['source_sha256'] != row['audio_sha256_source']:
                raise ValueError('Stale processing decision')
            if d['retained'] and file_hash(d['record']['audio_path']) != d['output_sha256']:
                raise ValueError('Retained audio changed')
            completed.add(row['utterance_id'])
    # Only this run's regenerable scratch is removed on restart. Durable scored
    # decisions, retained audio, boundary audits, and logs remain intact.
    if (args.output / '.work').exists():
        shutil.rmtree(args.output / '.work')
    initial, started = len(completed), time.monotonic()
    def progress(stage):
        now = datetime.now(ZoneInfo('Europe/Kyiv'))
        done = len(completed) - initial
        eta = now + timedelta(seconds=(len(rows) - len(completed)) * (time.monotonic() - started) / done) if done else None
        state = {'stage': stage, 'completed': len(completed), 'expected': len(rows),
                 'checked_kyiv': now.isoformat(timespec='seconds'), 'eta_kyiv': eta.isoformat(timespec='minutes') if eta else None}
        write_json(args.output / 'progress.json', state)
        print(json.dumps(state), flush=True)
    progress('dictionary')
    expanded = dictionary(args, rows)
    grids = {p.stem: p for p in Path('training/quality_runs/mfa_trim/aligned').rglob('*.json')}
    pilot = {r['utterance_id']: {**r, 'grid_path': str(grids[r['sample_id']])}
             for r in read_rows('training/quality_runs/mfa_processing/panel.jsonl')}
    from training.quality.sigmos_only import SigMOSOnly
    scorer = SigMOSOnly(args.sigmos_dir)
    batches = iter(source_batches(args, rows, completed))
    def prepare_next():
        batch = next(batches, None)
        return prepare_batch(args, batch, expanded, pilot) if batch else None
    with ThreadPoolExecutor(max_workers=1) as producer:
        future = producer.submit(prepare_next)
        while True:
            progress('MFA alignment')
            chunk = future.result()
            if chunk is None:
                break
            identifiers = {r['utterance_id'] for r in read_rows(chunk / 'panel.jsonl')}
            future = producer.submit(prepare_next)
            progress('enhancement and scoring')
            process_batch(args, chunk, scorer)
            completed.update(identifiers)
            progress('chunk complete')
    if not args.skip_finalize:
        finish(args, rows)
    progress('complete')


if __name__ == '__main__':
    main()
