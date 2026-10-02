"""Upload only numeric TensorBoard scalars while supervising a training command."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def collect_scalars(events, cursors):
    """Keep source steps; ignore all images, audio, graphs, and tensor payloads."""
    rows = {}
    updated = dict(cursors)
    for phase in ('train', 'valid'):
        directory = events / phase
        if not directory.exists():
            continue
        accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get('scalars', []):
            key = f'{phase}/{tag}'
            cursor = cursors.get(key, [-1, -1])
            for event in accumulator.Scalars(tag):
                position = [event.wall_time, event.step]
                if position <= cursor:
                    continue
                updated[key] = max(updated.get(key, [-1, -1]), position)
                if math.isfinite(event.value):
                    rows.setdefault(event.step, {'training_step': event.step})[key] = event.value
    return [rows[step] for step in sorted(rows)], updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--project', default='ukrainian-tts')
    parser.add_argument('--name', default='quality-v11-50k')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--initialize-only', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.initialize_only and not command:
        parser.error('A training command is required')

    import wandb

    state = json.loads(args.state.read_text()) if args.state.exists() else {
        'run_id': wandb.util.generate_id(), 'project': args.project, 'cursors': {},
    }
    if state['project'] != args.project:
        raise ValueError('Existing W&B state belongs to another project')
    save_state(args.state, state)
    # The uploader never reads model files and never calls save/log_artifact/watch.
    # Disable automatic code, console, hardware metadata, and system collection.
    run = wandb.init(
        project=args.project, entity=state.get('entity') or os.getenv('WANDB_ENTITY'),
        id=state['run_id'], name=args.name, resume='allow',
        dir=str(args.state.parent), mode='online', save_code=False,
        settings=wandb.Settings(console='off', disable_code=True, disable_git=True,
                                x_disable_meta=True, x_disable_stats=True,
                                x_save_requirements=False),
    )
    run.define_metric('training_step')
    run.define_metric('train/*', step_metric='training_step')
    run.define_metric('valid/*', step_metric='training_step')
    state.update(url=run.url, entity=run.entity)
    save_state(args.state, state)
    print(f'W&B numeric metrics only: {run.url}', flush=True)
    if args.initialize_only:
        run.finish()
        return

    child = None
    received_signal = None

    def forward(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward)

    def upload():
        rows, cursors = collect_scalars(args.events, state['cursors'])
        for row in rows:
            run.log(row)
        state['cursors'] = cursors
        save_state(args.state, state)

    status = 1
    try:
        if received_signal:
            raise SystemExit(128 + received_signal)
        # Stay in the supervisor's process group so its memory guard can stop
        # every GPU descendant even if this uploader or a recipe shell exits.
        child = subprocess.Popen(command)
        while child.poll() is None:
            upload()
            try:
                child.wait(timeout=args.interval)
            except subprocess.TimeoutExpired:
                pass
        status = child.returncode
        upload()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        run.finish(exit_code=status)
    raise SystemExit(status if status >= 0 else 128 - status)


if __name__ == '__main__':
    main()
