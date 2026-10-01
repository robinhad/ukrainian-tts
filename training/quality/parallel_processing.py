"""Independent model processes with disjoint files and an audited merged report."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .common import read_rows, write_json, write_tables


def process_parallel(panel, profile, output, device='cuda', resume=False,
                     cpu_workers=2, model_workers=2):
    rows = read_rows(panel)
    expected = {r['utterance_id'] for r in rows}
    if len(expected) != len(rows) or not 2 <= model_workers <= len(rows):
        raise ValueError('Invalid worker count or duplicate panel IDs')
    output = Path(output)
    logs = output / '.workers'
    logs.mkdir(parents=True, exist_ok=True)
    children, streams = [], []
    try:
        for index in range(model_workers):
            # Remove stale counters; verified per-file caches remain reusable.
            (output / f'progress-shard-{index}.json').unlink(missing_ok=True)
            stream = (logs / f'{index}.log').open('a')
            streams.append(stream)
            command = [sys.executable, '-m', 'training.quality', 'process',
                       '--panel', str(panel), '--profile', str(profile), '--output', str(output),
                       '--device', device, '--cpu-workers', str(cpu_workers),
                       '--num-shards', str(model_workers), '--shard-index', str(index)]
            if resume:
                command.append('--resume')
            children.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        last_message = 0.
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'Processing worker failed: {codes}; inspect {logs}')
            complete = 0
            for index in range(model_workers):
                path = output / f'progress-shard-{index}.json'
                if path.exists():
                    complete += json.loads(path.read_text())['completed']
            write_json(output / 'progress.json', {'completed': complete, 'expected': len(rows),
                                                  'model_workers': model_workers})
            if time.monotonic() - last_message >= 60:
                print(f'Processing: {complete}/{len(rows)}; {model_workers} model workers', flush=True)
                last_message = time.monotonic()
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
        results = [row for index in range(model_workers)
                   for row in read_rows(output / f'processing-shard-{index}.jsonl')]
        if len(results) != len(rows) or {r['utterance_id'] for r in results} != expected:
            raise RuntimeError('Worker results do not cover the input panel exactly')
        results.sort(key=lambda row: row['utterance_id'])
        write_tables(output, 'processing', results)
        write_json(output / 'progress.json', {'completed': len(results), 'expected': len(rows),
                                              'model_workers': model_workers})
        return results
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams:
            stream.close()
