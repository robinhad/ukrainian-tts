import datetime as dt
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from training.quality import supervise as module
from training.scripts.calibrate_quality_v10 import choose_trial


def fake_telemetry(workspace):
    return {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'free_disk_gib': 100, 'available_unified_memory_gib': 32,
            'total_unified_memory_gib': 128, 'gpus': []}


def test_memory_guard_kills_descendant_that_ignores_term(tmp_path, monkeypatch):
    marker = tmp_path / 'descendant.pid'
    monkeypatch.setattr(module, 'telemetry', fake_telemetry)
    monkeypatch.setattr(module, 'available_memory_gib', lambda: 16 if marker.exists() else 64)
    child_code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'
    command = [sys.executable, '-c',
               f'import subprocess,time; from pathlib import Path; '
               f'p=subprocess.Popen([{sys.executable!r}, "-c", {child_code!r}]); '
               f'Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(30)']
    try:
        code = module.supervise(command, tmp_path / 'monitor', 1, 24)
        assert code != 0
        summary = json.loads((tmp_path / 'monitor/summary.json').read_text())
        assert summary['termination']['reason'] == 'available_memory_below_reserve'
        assert summary['minimum_available_unified_memory_gib'] == 16
        state = Path('/proc') / marker.read_text() / 'stat'
        deadline = time.monotonic() + 2
        while state.exists() and state.read_text().split()[2] != 'Z' and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not state.exists() or state.read_text().split()[2] == 'Z'
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_training_activity_prevents_false_stall_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'telemetry', fake_telemetry)
    monkeypatch.setattr(module, 'available_memory_gib', lambda: 64)
    activity = tmp_path / 'train.log'
    command = [sys.executable, '-c',
               f'import os,time; from pathlib import Path; '
               f'Path({str(activity)!r}).write_text("training advances"); '
               f'os.utime({str(tmp_path / "command.log")!r},(1,1)); time.sleep(1.2)']
    assert module.supervise(command, tmp_path, 1, 24, [activity, tmp_path / 'missing.log']) == 0
    rows = [json.loads(line) for line in (tmp_path / 'telemetry.jsonl').open()]
    assert all('no_log_update_for_30_minutes' not in row['alerts'] for row in rows)
    assert json.loads((tmp_path / 'summary.json').read_text())['termination'] is None


def test_insufficient_memory_refuses_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'available_memory_gib', lambda: 16)
    with pytest.raises(RuntimeError, match='Insufficient available memory'):
        module.supervise([sys.executable, '-c', 'raise RuntimeError("must not launch")'],
                         tmp_path / 'monitor', 1, 24)
    assert not (tmp_path / 'monitor').exists()


def good_trial(**changes):
    return {'success': True, 'checkpoint_valid': True, 'optimizer_count': 2,
            'optimizer_steps_by_optimizer': [[100], [100]],
            'minimum_available_unified_memory_gib': 64,
            'batch_bins_per_training_second': 100, 'power_mean_watts': 50, **changes}


def test_prior_trials_cannot_promote_memory_failure_or_missing_optimizer():
    good = good_trial()
    unsafe = good_trial(minimum_available_unified_memory_gib=10,
                        batch_bins_per_training_second=200)
    incomplete = good_trial(optimizer_steps_by_optimizer=[[100], []],
                            batch_bins_per_training_second=300)
    assert choose_trial([unsafe, incomplete, good], minimum_available_gib=24) is good


def test_power_tie_break_stays_within_throughput_window():
    fastest = good_trial()
    near = good_trial(batch_bins_per_training_second=96, power_mean_watts=60)
    slow = good_trial(batch_bins_per_training_second=80, power_mean_watts=90)
    assert choose_trial([fastest, near, slow], minimum_available_gib=24) is near
