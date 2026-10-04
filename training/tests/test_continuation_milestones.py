import os
from pathlib import Path
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_preserves_nonstandard_final_milestone(tmp_path):
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    payload = b'completed checkpoint bytes'
    (experiment / '120epoch.pth').write_bytes(payload)
    # The trainer has exited; exercise the final preservation pass directly.
    subprocess.run(
        ['bash', str(ROOT / 'training/scripts/preserve_expanded_v9_milestones.sh'),
         '2147483647', '120000', str(experiment)], check=True, timeout=15,
    )
    saved = experiment / 'milestones/120k.pth'
    assert saved.read_bytes() == payload
    (experiment / '120epoch.pth').unlink()
    assert saved.read_bytes() == payload


@pytest.mark.parametrize('start,end', [('100000', '100000'), ('100000', '50000'),
                                      ('100000', '120001'), ('invalid', '120000')])
def test_invalid_continuation_stops_before_preparation(start, end):
    result = subprocess.run(
        ['bash', str(ROOT / 'training/scripts/continue_quality_v12_training.sh')],
        env={**os.environ, 'SLURM_JOB_ID': 'test', 'CONTINUE_FROM_STEPS': start,
             'CONTINUE_TO_STEPS': end}, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert 'Continuation bounds must increase' in result.stderr


def test_watcher_survives_source_replacement_while_waiting(tmp_path):
    script = tmp_path / 'watcher.sh'
    script.write_bytes((ROOT / 'training/scripts/preserve_expanded_v9_milestones.sh').read_bytes())
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    (experiment / '120epoch.pth').write_bytes(b'checkpoint')
    commands = tmp_path / 'bin'
    commands.mkdir()
    ready, release = tmp_path / 'ready', tmp_path / 'release'
    sleep = commands / 'sleep'
    sleep.write_text('#!/bin/bash\ntouch "$TEST_READY"\n'
                     'while [[ ! -e "$TEST_RELEASE" ]]; do /bin/sleep 0.02; done\n')
    sleep.chmod(0o755)
    process = subprocess.Popen(
        ['bash', str(script), '2147483647', '120000', str(experiment)],
        env={**os.environ, 'PATH': f'{commands}:{os.environ["PATH"]}',
             'TEST_READY': str(ready), 'TEST_RELEASE': str(release)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        script.write_text('#!/bin/bash\nexit 77\n' + 'exit 78\n' * 1000)
        release.touch()
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr
        assert (experiment / 'milestones/120k.pth').read_bytes() == b'checkpoint'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
