"""Persist progress and Kyiv ETA while the training supervisor is alive."""
import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from training.scripts.training_status import parse_log


def snapshot(log, steps, iterations_per_epoch=1000):
    now = dt.datetime.now(ZoneInfo('Europe/Kyiv'))
    text = log.read_text(errors='replace') if log.exists() else ''
    progress = parse_log(text, iterations_per_epoch)
    # A restart appends to the same log. Do not extrapolate its installation/
    # scheduler downtime, or the old attention backend's speed, into the ETA.
    marker = 'The training was resumed using'
    if marker in text:
        current = parse_log(text.rsplit(marker, 1)[1], iterations_per_epoch)
        progress['observed_seconds_per_iteration'] = current['observed_seconds_per_iteration']
        progress['error_matches'] = current['error_matches']
    completed = progress['total_iterations'] or 0
    seconds_per_step = progress['observed_seconds_per_iteration']
    eta = (now + dt.timedelta(seconds=max(0, steps - completed) * seconds_per_step)
           if seconds_per_step else None)
    return dict(timestamp_kyiv=now.isoformat(), target_steps=steps,
                completed_steps=completed, eta_kyiv=eta.isoformat() if eta else None,
                seconds_per_step=seconds_per_step, errors=progress['error_matches'],
                metrics=progress['latest_train_metrics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--iterations-per-epoch', type=int, default=1000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            os.kill(args.pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        status = snapshot(args.log, args.steps, args.iterations_per_epoch)
        status['supervisor_alive'] = alive
        with args.output.open('a') as stream:
            stream.write(json.dumps(status) + '\n')
        print(json.dumps(status), flush=True)
        if not alive:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
