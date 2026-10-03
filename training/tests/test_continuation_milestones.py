import os
from pathlib import Path
import subprocess

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
