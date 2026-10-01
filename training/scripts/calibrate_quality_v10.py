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

from training.quality.common import read_rows, write_json, write_tables


def checkpoint_measurements(path, expected_steps):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    reporter = payload['reporter']
    training = reporter['stats'][reporter['epoch']]['train']
    seconds = training['time'].total_seconds()
    optimizers = payload.get('optimizers', [])
    per_optimizer = [{int(state['step']) for state in optimizer.get('state', {}).values()
                      if isinstance(state, dict) and 'step' in state} for optimizer in optimizers]
    steps = set().union(*per_optimizer)
    model = payload.get('model', {})
    nonfinite = [name for name, tensor in model.items() if torch.is_tensor(tensor)
                 and (tensor.is_floating_point() or tensor.is_complex())
                 and not torch.isfinite(tensor).all()]
    valid = (training['total_count'] == expected_steps and len(optimizers) == 2
             and all(s == {expected_steps} for s in per_optimizer)
             and bool(model) and not nonfinite and seconds > 0)
    return {'training_seconds': seconds, 'optimizer_steps': sorted(steps),
            'optimizer_count': len(optimizers), 'nonfinite_model_tensors': nonfinite,
            'optimizer_steps_by_optimizer': [sorted(s) for s in per_optimizer],
            'checkpoint_valid': valid}


def choose_trial(rows, minimum_available_gib=24, steps=100):
    successful = [r for r in rows if r['success'] and r['checkpoint_valid']
                  and r['optimizer_count'] == 2
                  and r['optimizer_steps_by_optimizer'] == [[steps], [steps]]
                  and r['minimum_available_unified_memory_gib'] >= minimum_available_gib]
    if not successful:
        raise RuntimeError('No successful calibration; inspect trial logs')
    fastest = max(successful, key=lambda r: r['batch_bins_per_training_second'])
    near = [r for r in successful if r['batch_bins_per_training_second'] >= .95 * fastest['batch_bins_per_training_second']]
    return max(near, key=lambda r: r['power_mean_watts'] or 0)


def validate_prior_trials(rows, steps, cuda_cache_interval):
    if any(row['steps'] != steps for row in rows):
        raise ValueError('Prior trials must use the same number of steps')
    if any(row.get('cuda_cache_interval', 0) != cuda_cache_interval for row in rows):
        raise ValueError('Prior trials must use the same CUDA cache interval')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-bins', type=int, nargs='+', default=[2000000, 4000000, 8000000])
    parser.add_argument('--workers', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--cuda-cache-interval', type=int,
                        default=int(os.getenv('UKTTS_CUDA_CACHE_INTERVAL', '10')))
    parser.add_argument('--tf32', choices=['false', 'true'], nargs='+', default=['false', 'true'])
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/v10/calibration'))
    parser.add_argument('--prior-trials', type=Path, action='append', default=[])
    parser.add_argument('--minimum-available-gib', type=float, default=24)
    args = parser.parse_args()
    if args.steps < 1 or args.cuda_cache_interval < 0:
        parser.error('Steps must be positive and CUDA cache interval nonnegative')
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run calibration in a SLURM GPU allocation')
    prior = [row for path in args.prior_trials for row in read_rows(path)]
    validate_prior_trials(prior, args.steps, args.cuda_cache_interval)
    rows = []
    for bins in args.batch_bins:
        for workers, tf32 in product(args.workers, args.tf32):
            name = f'bins{bins}_workers{workers}_tf32{tf32}'
            output = args.output / name
            experiment = Path('training/exp_quality_v10') / args.output.name / name
            env = {**os.environ, 'BATCH_BINS': str(bins), 'WORKERS': str(workers),
                   'USE_TF32': tf32,
                   'STEPS': str(args.steps), 'ITERS_PER_EPOCH': str(args.steps),
                   'UKTTS_CUDA_CACHE_INTERVAL': str(args.cuda_cache_interval),
                   'TTS_EXP': str(experiment.resolve())}
            # An earlier recommendation must not override the trial under test.
            env.pop('CALIBRATION_FILE', None)
            if experiment.exists():
                raise ValueError(f'Use a new calibration directory; existing trial: {name}')
            result = subprocess.run([sys.executable, '-m', 'training.quality.supervise',
                                     '--output', str(output), '--interval', '5',
                                     '--minimum-available-gib', str(args.minimum_available_gib),
                                     '--activity-log', str(experiment / 'train.log'), '--',
                                     'bash', 'training/scripts/run_quality_v10_training.sh'], env=env)
            summary = json.loads((output / 'summary.json').read_text())
            log = (output / 'command.log').read_text(errors='replace')
            if (experiment / 'train.log').exists():
                log += (experiment / 'train.log').read_text(errors='replace')
            failed = bool(re.search(r'CUDA out of memory|\bNaN\b|Traceback', log, re.I))
            elapsed = summary['elapsed_seconds']
            measured = {'training_seconds': None, 'optimizer_steps': [], 'optimizer_count': 0,
                        'optimizer_steps_by_optimizer': [],
                        'nonfinite_model_tensors': [], 'checkpoint_valid': False}
            checkpoint = experiment / 'checkpoint.pth'
            if result.returncode == 0 and checkpoint.exists():
                measured = checkpoint_measurements(checkpoint, args.steps)
            training_seconds = measured['training_seconds']
            # Larger batches do more work; report bins/s as a separate throughput proxy.
            rows.append({'batch_bins': bins, 'workers': workers, 'steps': args.steps,
                         'use_tf32': tf32 == 'true',
                         'cuda_cache_interval': int(env.get('UKTTS_CUDA_CACHE_INTERVAL', '0')),
                         'success': result.returncode == 0 and not failed and measured['checkpoint_valid'],
                         **measured,
                         'training_steps_per_second': args.steps / training_seconds if training_seconds else None,
                         'batch_bins_per_training_second': bins * args.steps / training_seconds if training_seconds else None,
                         'steps_per_second': args.steps / elapsed,
                         'batch_bins_per_second': bins * args.steps / elapsed, **summary})
            write_tables(args.output, 'trials', rows)
    combined = prior + rows
    write_tables(args.output, 'combined_trials', combined)
    chosen = choose_trial(combined, args.minimum_available_gib, args.steps)
    write_json(args.output / 'recommended.json', {
        'batch_bins': chosen['batch_bins'], 'workers': chosen['workers'],
        'use_tf32': chosen['use_tf32'],
        'cuda_cache_interval': chosen.get('cuda_cache_interval', 0),
        'minimum_available_gib': args.minimum_available_gib,
        'power_reference_watts': 100, 'criterion': 'batch-bin throughput during training; power breaks ties within 5 percent',
        'note': 'No power cap change. Batch bins/s is a work proxy, not measured audio samples/s.',
    })


if __name__ == '__main__':
    main()
