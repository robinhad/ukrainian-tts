"""Export portable, paired training/validation scalars at completed epochs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_epochs(events: Path, through_step: int) -> list[dict]:
    phases = {}
    for phase in ('train', 'valid'):
        accumulator = EventAccumulator(str(events / phase), size_guidance={'scalars': 0})
        accumulator.Reload()
        rows = {}
        for tag in sorted(accumulator.Tags().get('scalars', [])):
            # Epoch summaries follow batch summaries at the same training step.
            # Keep the last event, as in summarize_tensorboard.py.
            for event in sorted(accumulator.Scalars(tag), key=lambda e: (e.step, e.wall_time)):
                if event.step > through_step:
                    continue
                if not math.isfinite(event.value):
                    raise ValueError(f'Nonfinite {phase}/{tag} at step {event.step}')
                rows.setdefault(event.step, {})[f'{phase}_{tag}'] = event.value
        phases[phase] = rows
    result = []
    for step, validation in sorted(phases['valid'].items()):
        if step not in phases['train']:
            raise ValueError(f'Validation step {step} has no training metrics')
        result.append({'training_step': step, **phases['train'][step], **validation})
    if not result:
        raise ValueError('No completed validation epochs are available')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--through-step', type=int, required=True)
    args = parser.parse_args()
    rows = export_epochs(args.events, args.through_step)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
    fields = ['training_step'] + sorted(set().union(*(row.keys() for row in rows)) - {'training_step'})
    with args.output_prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'completed_epochs': len(rows), 'last_step': rows[-1]['training_step']}))


if __name__ == '__main__':
    main()
