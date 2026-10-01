"""Bound model-process lifetimes and audit merged preprocessing coverage."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .common import read_rows, write_json, write_tables


def process_parallel(panel, profile, output, device='cuda', resume=False,
                     cpu_workers=2, model_workers=2, chunk_size=64,
                     minimum_available_gib=4):
    rows = read_rows(panel)
    expected = {r['utterance_id'] for r in rows}
    if (len(expected) != len(rows) or not 2 <= model_workers <= len(rows)
            or chunk_size < 1 or minimum_available_gib < 0):
        raise ValueError('Invalid worker/chunk settings or duplicate panel IDs')
    if any('voa' in str(row['source_id']).lower() for row in rows):
        raise ValueError('VOA is forbidden')
    output = Path(output)
    logs = output / '.workers'
    logs.mkdir(parents=True, exist_ok=True)
    tasks = []
    fields = ('utterance_id', 'source_id', 'audio_path', 'reference_sha256',
              'processing_audio_path', 'processing_input_sha256')
    for index, start in enumerate(range(0, len(rows), chunk_size)):
        tag = f'chunk-{index}'
        subset = [{key: row[key] for key in fields if key in row}
                  for row in rows[start:start + chunk_size]]
        manifest = logs / f'{tag}.jsonl'
        manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in subset))
        (output / f'progress-{tag}.json').unlink(missing_ok=True)
        tasks.append((tag, manifest))
    active, next_task, last_message = [], 0, 0.
    try:
        while next_task < len(tasks) or active:
            meminfo = Path('/proc/meminfo')
            if meminfo.exists():
                available = next(int(line.split()[1]) / 1024**2
                                 for line in meminfo.read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                if available < minimum_available_gib:
                    raise RuntimeError(f'Available memory below {minimum_available_gib:g} GiB reserve')
            for child, stream in list(active):
                code = child.poll()
                if code is not None:
                    stream.close()
                    active.remove((child, stream))
                    if code:
                        raise RuntimeError(f'Processing worker failed ({code}); inspect {logs}')
            while next_task < len(tasks) and len(active) < model_workers:
                tag, manifest = tasks[next_task]
                stream = (logs / f'{tag}.log').open('a')
                command = [sys.executable, '-m', 'training.quality', 'process',
                           '--panel', str(manifest), '--profile', str(profile), '--output', str(output),
                           '--device', device, '--cpu-workers', str(cpu_workers), '--report-tag', tag]
                if resume:
                    command.append('--resume')
                try:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                active.append((child, stream))
                next_task += 1
            complete = sum(json.loads(path.read_text())['completed'] for tag, _ in tasks
                           if (path := output / f'progress-{tag}.json').exists())
            write_json(output / 'progress.json', {'completed': complete, 'expected': len(rows),
                                                  'model_workers': model_workers, 'chunk_size': chunk_size})
            if time.monotonic() - last_message >= 60:
                print(f'Processing: {complete}/{len(rows)}; {model_workers} workers, {chunk_size}-file lifetimes', flush=True)
                last_message = time.monotonic()
            if active:
                time.sleep(2)
        results = [row for tag, _ in tasks for row in read_rows(output / f'processing-{tag}.jsonl')]
        if len(results) != len(rows) or {r['utterance_id'] for r in results} != expected:
            raise RuntimeError('Worker results do not cover the input panel exactly')
        results.sort(key=lambda row: row['utterance_id'])
        write_tables(output, 'processing', results)
        write_json(output / 'progress.json', {'completed': len(results), 'expected': len(rows),
                                              'model_workers': model_workers, 'chunk_size': chunk_size})
        return results
    finally:
        for child, _ in active:
            if child.poll() is None:
                child.terminate()
        for child, stream in active:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            stream.close()
