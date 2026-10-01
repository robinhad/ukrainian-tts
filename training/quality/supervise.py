"""Run a command with durable status, power, memory, and disk telemetry."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from .common import write_json


def available_memory_gib():
    return next(int(line.split()[1]) / 1024**2
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


class MemoryReserveGuard:
    """Check host memory independently of potentially slow GPU telemetry."""

    def __init__(self, child, reserve):
        self.child, self.reserve = child, reserve
        self.minimum = None
        self.termination = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        while not self.stop.is_set():
            available = available_memory_gib()
            self.minimum = available if self.minimum is None else min(self.minimum, available)
            if available < self.reserve:
                self.termination = {'reason': 'available_memory_below_reserve',
                                    'reserve_gib': self.reserve, 'available_gib': available}
                signal_group(self.child.pid, signal.SIGTERM)
                # A shell may exit before its GPU descendants. Kill any remaining
                # members of the session we created, even if the leader has exited.
                time.sleep(2)
                signal_group(self.child.pid, signal.SIGKILL)
                return
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.thread.join()


def telemetry(workspace):
    record = {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'free_disk_gib': shutil.disk_usage(workspace).free / 1024**3}
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
           if line.split()[1].isdigit()}
    record['available_unified_memory_gib'] = mem['MemAvailable'] / 1024**2
    record['total_unified_memory_gib'] = mem['MemTotal'] / 1024**2
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw,utilization.gpu,temperature.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        gpu_output = result.stdout
        if result.returncode:
            record['gpu_query_error'] = 'nvidia_smi_failed'
    except subprocess.TimeoutExpired:
        gpu_output = ''
        record['gpu_query_error'] = 'nvidia_smi_timeout'
    values = []
    for line in gpu_output.splitlines():
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


def supervise(command, output, interval=60, minimum_available_gib=0, activity_logs=()):
    if not 1 <= interval <= 1800:
        raise ValueError('Monitoring interval must be at most 30 minutes')
    if minimum_available_gib < 0:
        raise ValueError('The memory reserve cannot be negative')
    if minimum_available_gib and available_memory_gib() < minimum_available_gib:
        raise RuntimeError('Insufficient available memory before launching command')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    samples = []
    with (output / 'command.log').open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        guard = MemoryReserveGuard(child, minimum_available_gib) if minimum_available_gib else None
        if guard:
            guard.thread.start()
        try:
            while True:
                sample = telemetry(output)
                sample.update(elapsed_seconds=time.monotonic() - started, returncode=child.poll())
                logs = [output / 'command.log', *(Path(p) for p in activity_logs)]
                sample['log_idle_seconds'] = time.time() - max(p.stat().st_mtime for p in logs if p.exists())
                sample['alerts'] = []
                if sample.get('gpu_query_error'):
                    sample['alerts'].append(sample['gpu_query_error'])
                if guard and guard.termination:
                    sample['alerts'].append('available_memory_below_reserve')
                    sample['termination'] = guard.termination
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
            signal_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                signal_group(child.pid, signal.SIGKILL)
            raise
        finally:
            if guard:
                guard.close()
    power = [gpu['power_watts'] for sample in samples[1:] for gpu in sample['gpus']
             if gpu.get('power_watts') is not None]
    returncode = child.returncode or (1 if guard and guard.termination else 0)
    minimum_memory = min(s['available_unified_memory_gib'] for s in samples)
    if guard and guard.minimum is not None:
        minimum_memory = min(minimum_memory, guard.minimum)
    summary = {'returncode': returncode, 'elapsed_seconds': time.monotonic() - started,
               'samples': len(samples), 'power_mean_watts': sum(power) / len(power) if power else None,
               'power_max_watts': max(power) if power else None,
               'power_reference_watts': 100, 'maximum_check_interval_seconds': interval,
               'minimum_free_disk_gib': min(s['free_disk_gib'] for s in samples),
               'minimum_available_unified_memory_gib': minimum_memory,
               'memory_reserve_gib': minimum_available_gib,
               'termination': guard.termination if guard else None}
    write_json(output / 'summary.json', summary)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--minimum-available-gib', type=float, default=0)
    parser.add_argument('--activity-log', type=Path, action='append', default=[])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply a command after --')
    raise SystemExit(supervise(command, args.output, args.interval,
                               args.minimum_available_gib, args.activity_log))


if __name__ == '__main__':
    main()
