"""Run a command with durable status, power, memory, and disk telemetry."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .common import write_json


def telemetry(workspace):
    record = {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'free_disk_gib': shutil.disk_usage(workspace).free / 1024**3}
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
           if line.split()[1].isdigit()}
    record['available_unified_memory_gib'] = mem['MemAvailable'] / 1024**2
    record['total_unified_memory_gib'] = mem['MemTotal'] / 1024**2
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,utilization.gpu,temperature.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    values = []
    for line in result.stdout.splitlines():
        if len(line.split(',')) != 3:
            continue
        row = {}
        for name, value in zip(['power_watts', 'utilization_percent', 'temperature_c'], line.split(',')):
            try:
                row[name] = float(value.strip())
            except ValueError:
                row[name] = None
        values.append(row)
    record['gpus'] = values
    return record


def supervise(command, output, interval=60):
    if not 1 <= interval <= 1800:
        raise ValueError('Monitoring interval must be at most 30 minutes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    samples = []
    with (output / 'command.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                sample = telemetry(output)
                sample.update(elapsed_seconds=time.monotonic() - started, returncode=child.poll())
                sample['log_idle_seconds'] = time.time() - (output / 'command.log').stat().st_mtime
                sample['alerts'] = []
                if sample['free_disk_gib'] < 30:
                    sample['alerts'].append('disk_below_30_gib')
                if sample['available_unified_memory_gib'] < 4:
                    sample['alerts'].append('unified_memory_below_4_gib')
                if sample['log_idle_seconds'] > 1800:
                    sample['alerts'].append('no_log_update_for_30_minutes')
                for gpu in sample['gpus']:
                    if (gpu.get('power_watts') or 0) > 100:
                        sample['alerts'].append('power_above_100_watts')
                samples.append(sample)
                with (output / 'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(sample) + '\n')
                write_json(output / 'status.json', sample)
                print(json.dumps(sample), flush=True)
                if child.poll() is not None:
                    break
                # Status flags do not silently alter training hyperparameters or power caps.
                time.sleep(interval)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            raise
    power = [gpu['power_watts'] for sample in samples[1:] for gpu in sample['gpus']
             if gpu.get('power_watts') is not None]
    summary = {'returncode': child.returncode, 'elapsed_seconds': time.monotonic() - started,
               'samples': len(samples), 'power_mean_watts': sum(power) / len(power) if power else None,
               'power_max_watts': max(power) if power else None,
               'power_reference_watts': 100, 'maximum_check_interval_seconds': interval,
               'minimum_free_disk_gib': min(s['free_disk_gib'] for s in samples),
               'minimum_available_unified_memory_gib': min(s['available_unified_memory_gib'] for s in samples)}
    write_json(output / 'summary.json', summary)
    return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply a command after --')
    raise SystemExit(supervise(command, args.output, args.interval))


if __name__ == '__main__':
    main()
