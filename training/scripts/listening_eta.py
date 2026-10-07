"""Timestamped, portable training-rate snapshots for listening-page ETAs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from training.scripts.training_status import parse_log, TIMED_PROGRESS_RE


def capture(log, scheduled_end_step=None, now=None):
    now = now or datetime.now(timezone.utc)
    log = Path(log)
    if scheduled_end_step is None:
        target = log.parent / 'target_iterations.txt'
        scheduled_end_step = int(target.read_text()) if target.exists() else None
    result = {'captured_at': now.isoformat(), 'timezone': 'Europe/Kyiv',
              'scheduled_end_step': scheduled_end_step, 'status': 'unavailable',
              'assumption': 'Continuous training at the recent measured rate; excludes pauses, preparation and evaluation overhead'}
    if not log.exists():
        return result
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 1500000))
        text = stream.read().decode(errors='replace')
    parsed = parse_log(text, 1000)
    matches = list(TIMED_PROGRESS_RE.finditer(text))
    speed = parsed['observed_seconds_per_iteration']
    if not matches or not speed or not math.isfinite(speed) or speed <= 0:
        return result
    stamp = datetime.strptime(matches[-1]['timestamp'], '%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc)
    age = (now - stamp).total_seconds()
    result.update(observed_at=stamp.isoformat(), current_step=parsed['total_iterations'],
                  seconds_per_step=speed, steps_per_hour=3600 / speed,
                  measurement_age_seconds=age,
                  status='measured' if 0 <= age <= 300 else 'stale')
    return result


def target_eta(target, snapshot):
    if snapshot.get('status') != 'measured':
        return {'status': 'unavailable'}
    if target <= snapshot['current_step']:
        return {'status': 'step_reached', 'remaining_hours': 0}
    stamp = datetime.fromisoformat(snapshot['observed_at']).astimezone(timezone.utc)
    captured = datetime.fromisoformat(snapshot['captured_at']).astimezone(timezone.utc)
    try:
        end = stamp + timedelta(seconds=(target - snapshot['current_step']) * snapshot['seconds_per_step'])
    except (OverflowError, ValueError):
        return {'status': 'unavailable'}
    limit = snapshot.get('scheduled_end_step')
    return {'status': 'projected', 'eta_kyiv': end.astimezone(ZoneInfo('Europe/Kyiv')).isoformat(),
            'remaining_hours': max(0, (end - captured).total_seconds() / 3600),
            'beyond_schedule': limit is not None and target > limit}


def attach(result, snapshot):
    result = copy.deepcopy(result)
    result['timing'] = snapshot
    for group in [result, result.get('lower_bound')]:
        if not group:
            continue
        for scenario in group['scenarios']:
            for value in scenario['targets'].values():
                value['eta'] = (target_eta(value['total_steps'], snapshot)
                                if value['status'] == 'extrapolated' else {'status': value['status']})
    if snapshot.get('scheduled_end_step'):
        result['scheduled_finish'] = target_eta(snapshot['scheduled_end_step'], snapshot)
    return result


def forecast_for_output(comparison, output):
    from training.scripts.forecast_listening_quality import forecast
    result = forecast(comparison)
    path = Path(output) / 'training_eta.json'
    return attach(result, json.loads(path.read_text())) if path.exists() else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--scheduled-end-step', type=int)
    args = parser.parse_args()
    snapshot = capture(args.log, args.scheduled_end_step)
    (args.output / 'training_eta.json').write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
    from training.scripts.generate_checkpoint_listening import write_pages
    rows = [json.loads(line) for line in (args.output / 'selection.jsonl').read_text().splitlines()]
    step = json.loads((args.output / 'report.json').read_text())['checkpoint_step']
    write_pages(args.output, rows, step)
    print(json.dumps(snapshot))


if __name__ == '__main__':
    main()
