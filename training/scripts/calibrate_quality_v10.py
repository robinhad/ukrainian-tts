#!/usr/bin/env python3
"""Measure real JETS batch throughput and power before the 100K run."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from itertools import product

from training.quality.common import write_json, write_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-bins', type=int, nargs='+', default=[2000000, 4000000, 8000000])
    parser.add_argument('--workers', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--tf32', choices=['false', 'true'], nargs='+', default=['false', 'true'])
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/v10/calibration'))
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run calibration in a SLURM GPU allocation')
    rows = []
    for bins in args.batch_bins:
        for workers, tf32 in product(args.workers, args.tf32):
            name = f'bins{bins}_workers{workers}_tf32{tf32}'
            output = args.output / name
            experiment = Path('training/exp_quality_v10') / args.output.name / name
            env = {**os.environ, 'BATCH_BINS': str(bins), 'WORKERS': str(workers),
                   'USE_TF32': tf32,
                   'STEPS': str(args.steps), 'ITERS_PER_EPOCH': str(args.steps),
                   'TTS_EXP': str(experiment.resolve())}
            if experiment.exists():
                raise ValueError(f'Use a new calibration directory; existing trial: {name}')
            result = subprocess.run([sys.executable, '-m', 'training.quality.supervise',
                                     '--output', str(output), '--interval', '5', '--',
                                     'bash', 'training/scripts/run_quality_v10_training.sh'], env=env)
            summary = json.loads((output / 'summary.json').read_text())
            log = (output / 'command.log').read_text(errors='replace')
            if (experiment / 'train.log').exists():
                log += (experiment / 'train.log').read_text(errors='replace')
            failed = bool(re.search(r'CUDA out of memory|\bNaN\b|Traceback', log, re.I))
            elapsed = summary['elapsed_seconds']
            training_seconds = None
            checkpoint = experiment / 'checkpoint.pth'
            if result.returncode == 0 and checkpoint.exists():
                import torch
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                reporter = payload['reporter']
                training = reporter['stats'][reporter['epoch']]['train']
                training_seconds = training['time'].total_seconds()
                failed |= training['total_count'] != args.steps
                del payload
            # Larger batches do more work; report bins/s as a separate throughput proxy.
            rows.append({'batch_bins': bins, 'workers': workers, 'steps': args.steps,
                         'use_tf32': tf32 == 'true',
                         'success': result.returncode == 0 and not failed and bool(training_seconds),
                         'training_seconds': training_seconds,
                         'training_steps_per_second': args.steps / training_seconds if training_seconds else None,
                         'batch_bins_per_training_second': bins * args.steps / training_seconds if training_seconds else None,
                         'steps_per_second': args.steps / elapsed,
                         'batch_bins_per_second': bins * args.steps / elapsed, **summary})
            write_tables(args.output, 'trials', rows)
    successful = [r for r in rows if r['success'] and r['minimum_available_unified_memory_gib'] >= 8]
    if not successful:
        raise RuntimeError('No successful calibration; inspect trial logs')
    fastest = max(successful, key=lambda r: r['batch_bins_per_training_second'])
    # Similar throughput: prefer the configuration that sustains more useful GPU work.
    near = [r for r in successful if r['batch_bins_per_training_second'] >= .95 * fastest['batch_bins_per_training_second']]
    chosen = max(near, key=lambda r: r['power_mean_watts'] or 0)
    write_json(args.output / 'recommended.json', {
        'batch_bins': chosen['batch_bins'], 'workers': chosen['workers'],
        'use_tf32': chosen['use_tf32'],
        'power_reference_watts': 100, 'criterion': 'batch-bin throughput during training; power breaks ties within 5 percent',
        'note': 'No power cap change. Batch bins/s is a work proxy, not measured audio samples/s.',
    })


if __name__ == '__main__':
    main()
