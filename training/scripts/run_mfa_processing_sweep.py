"""Bounded GPU workers for the MFA processing comparison, with resumable SigMOS."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf
import yaml

from training.quality.common import digest, file_hash, read_rows, write_json, write_tables

SIGMOS = {'MOS_OVRL': 'overall', 'MOS_SIG': 'speech', 'MOS_NOISE': 'noise',
          'MOS_COL': 'coloration', 'MOS_DISC': 'discontinuity',
          'MOS_LOUD': 'loudness', 'MOS_REVERB': 'reverb'}


def worker(args):
    import torch
    from training.quality.processing import process
    from training.quality.sigmos_only import SigMOSOnly
    task = json.loads(args.task.read_text())
    panel, output = Path(task['panel']), Path(task['output'])
    rows = read_rows(panel)
    profile = yaml.safe_load(Path(task['profile']).read_text())
    if any(r.get('boundary_method') != 'mfa' for r in rows):
        raise ValueError('All processing requires MFA inputs')
    output.mkdir(parents=True, exist_ok=True)
    if profile['backend'] == 'legacy_cascade':
        import pandas as pd
        manifest = args.task.with_suffix('.parquet')
        pd.DataFrame([{'utterance_id': r['utterance_id'], 'audio_path': r['processing_audio_path']}
                      for r in rows]).to_parquet(manifest, index=False)
        for row in rows:
            if file_hash(row['processing_audio_path']) != row['processing_input_sha256']:
                raise ValueError('Legacy MFA input changed')
        subprocess.run([sys.executable, '-m', 'training.scripts.preprocess_training_cascade_audio',
                        '--manifest', str(manifest), '--input-audio-manifest', str(manifest),
                        '--output-root', str(output / 'wav'), '--output-results', str(args.task.with_suffix('.legacy.jsonl')),
                        '--model-cache', profile['model_cache'], '--rnnoise-binary', profile['rnnoise_binary'],
                        '--profile-name', 'mfa_legacy_cascade', '--resume'], check=True)
    else:
        process(panel, task['profile'], output / 'wav', device='cuda:0', resume=True,
                cpu_workers=2, report_tag=task['tag'])
    gc.collect()
    torch.cuda.empty_cache()
    scorer = SigMOSOnly(args.sigmos_dir)
    scores = []
    score_dir = output / 'scores'
    score_dir.mkdir(exist_ok=True)
    for row in rows:
        audio_path = output / 'wav' / (row['utterance_id'] + '.wav')
        meta_path = audio_path.with_suffix('.wav.json' if profile['backend'] == 'legacy_cascade' else '.json')
        meta = json.loads(meta_path.read_text())
        audio_hash = file_hash(audio_path)
        if (meta['output_sha256'] != audio_hash
                or meta.get('input_sha256', meta.get('source_sha256')) != row['processing_input_sha256']):
            raise ValueError('Output provenance mismatch')
        audio, rate = sf.read(audio_path, dtype='float32', always_2d=True)
        if rate != 24000 or audio.shape[1] != 1 or len(audio) != row['mfa_end_frame'] - row['mfa_start_frame']:
            raise ValueError('Processing changed MFA duration')
        path = score_dir / (row['utterance_id'] + '.json')
        cache_key = digest([audio_hash, scorer.identity, file_hash(__file__)])
        item = json.loads(path.read_text()) if path.exists() else None
        if item is None or item.get('key') != cache_key:
            values = scorer.score(audio[:, 0], rate)
            item = {'utterance_id': row['utterance_id'], 'source_id': row['source_id'],
                    'sample_id': row['sample_id'], 'profile': task['name'], 'key': cache_key,
                    'reference_sha256': row['reference_sha256'], 'audio_sha256': audio_hash,
                    'processing_input_sha256': row['processing_input_sha256'],
                    'duration_seconds': len(audio) / rate, 'boundary_method': 'mfa',
                    'mfa_status': row['mfa_status'], 'mfa_flags': row['mfa_flags'],
                    **{'sigmos_' + name: float(values[key]) for key, name in SIGMOS.items()}}
            write_json(path, item)
        scores.append(item)
    write_tables(output / 'chunks', task['tag'], scores)
    write_json(args.task.with_suffix('.done.json'), {
        'count': len(scores), 'task_sha256': file_hash(args.task),
        'sigmos_identity': scorer.identity})


def sweep(args):
    rows = read_rows(args.run / 'panel.jsonl')
    if (len(rows) != 800 or sorted(Counter(r['source_id'] for r in rows).values()) != [100] * 8
            or len({r['utterance_id'] for r in rows}) != len(rows)
            or any('voa' in r['source_id'].lower() or r.get('boundary_method') != 'mfa' for r in rows)):
        raise ValueError('Expected the fixed source-balanced 800-item non-VOA MFA panel')
    config = yaml.safe_load(args.config.read_text())
    work = args.run / 'workers'
    work.mkdir(parents=True, exist_ok=True)
    tasks = []
    for name, overrides in config['profiles'].items():
        output = args.run / name
        output.mkdir(exist_ok=True)
        profile = {**config['defaults'], **overrides}
        profile_path = output / 'profile.yaml'
        if profile_path.exists() and yaml.safe_load(profile_path.read_text()) != profile:
            raise ValueError('Existing sweep profile changed')
        profile_path.write_text(yaml.safe_dump(profile))
        for index, start in enumerate(range(0, len(rows), args.chunk_size)):
            tag = f'chunk-{index}'
            panel = work / f'{name}-{tag}.panel.jsonl'
            panel.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[start:start + args.chunk_size]))
            task_path = work / f'{name}-{tag}.task.json'
            write_json(task_path, {'name': name, 'tag': tag, 'panel': str(panel),
                                   'profile': str(profile_path), 'output': str(output)})
            # Let processor/scorer caches verify hashes on every resumed task.
            task_path.with_suffix('.done.json').unlink(missing_ok=True)
            tasks.append(task_path)
    active, next_task, complete, last_message = [], 0, 0, 0.
    started = time.monotonic()
    try:
        while next_task < len(tasks) or active:
            available = next(int(line.split()[1]) / 1024**2 for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))
            if available < args.minimum_available_gib or shutil.disk_usage(args.run).free < 30 * 1024**3:
                raise RuntimeError('Memory or disk reserve reached')
            for child, stream, task in list(active):
                code = child.poll()
                if code is not None:
                    stream.close()
                    active.remove((child, stream, task))
                    if code:
                        raise RuntimeError(f'Worker failed ({code}): {task.name}; inspect its log')
                    complete += 1
            while next_task < len(tasks) and len(active) < args.workers:
                task = tasks[next_task]
                stream = task.with_suffix('.log').open('a')
                child = subprocess.Popen([sys.executable, '-m', 'training.scripts.run_mfa_processing_sweep',
                                          '--task', str(task), '--sigmos-dir', str(args.sigmos_dir)],
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((child, stream, task))
                next_task += 1
            elapsed = time.monotonic() - started
            now = datetime.now(ZoneInfo('Europe/Kyiv'))
            eta = (now + timedelta(seconds=elapsed / complete * (len(tasks) - complete))).isoformat(timespec='minutes') if complete else None
            state = {'completed_chunks': complete, 'expected_chunks': len(tasks), 'active_workers': len(active),
                     'elapsed_seconds': elapsed, 'available_gib': available,
                     'checked_kyiv': now.isoformat(timespec='seconds'), 'eta_kyiv': eta}
            write_json(args.run / 'progress.json', state)
            if time.monotonic() - last_message > 60:
                telemetry = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,utilization.gpu',
                                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
                state['gpu_power_watts_utilization_percent'] = telemetry.stdout.strip()
                with (args.run / 'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(state) + '\n')
                print(json.dumps(state), flush=True)
                last_message = time.monotonic()
            if active:
                time.sleep(5)
        for name in config['profiles']:
            results = [json.loads(p.read_text()) for p in (args.run / name / 'scores').glob('*.json')]
            if len(results) != len(rows) or {r['utterance_id'] for r in results} != {r['utterance_id'] for r in rows}:
                raise ValueError('Incomplete or duplicate score coverage')
            write_tables(args.run / name, 'per_file', sorted(results, key=lambda r: r['utterance_id']))
        write_json(args.run / 'complete.json', {'status': 'complete', 'profiles': list(config['profiles']),
                                               'items_per_profile': len(rows), 'elapsed_seconds': time.monotonic() - started})
    finally:
        import os
        import signal
        for child, _, _ in active:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child, stream, _ in active:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('training/quality_runs/mfa_processing'))
    parser.add_argument('--config', type=Path, default=Path('training/conf/quality_mfa_sweep.yaml'))
    parser.add_argument('--sigmos-dir', type=Path, default=Path('training/vendor/quality-models/SIG-Challenge/ICASSP2024/sigmos'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--minimum-available-gib', type=float, default=16)
    parser.add_argument('--task', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1 or args.minimum_available_gib < 0:
        parser.error('Invalid worker, chunk, or memory settings')
    worker(args) if args.task else sweep(args)


if __name__ == '__main__':
    main()
