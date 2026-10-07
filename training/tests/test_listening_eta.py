from datetime import datetime, timezone

import pytest

from training.scripts.listening_eta import attach, capture, target_eta
from training.scripts.forecast_listening_quality import forecast, forecast_html


def snapshot():
    return {'status': 'measured', 'current_step': 284000, 'seconds_per_step': 1.5,
            'steps_per_hour': 2400., 'observed_at': '2026-10-07T14:00:00+00:00',
            'captured_at': '2026-10-07T14:00:15+00:00', 'scheduled_end_step': 478000}


def test_eta_uses_live_step_and_observation_time_not_scored_checkpoint():
    eta = target_eta(339000, snapshot())
    assert eta['eta_kyiv'] == '2026-10-08T15:55:00+03:00'
    assert eta['remaining_hours'] == pytest.approx((55000*1.5-15)/3600)
    assert not eta['beyond_schedule']
    assert target_eta(500000, snapshot())['beyond_schedule']
    assert target_eta(280000, snapshot())['status'] == 'step_reached'


def test_eta_crosses_kyiv_dst_using_elapsed_time():
    s = snapshot()
    s.update(observed_at='2026-10-24T21:00:00+00:00', captured_at='2026-10-24T21:00:00+00:00')
    assert target_eta(308000, s)['eta_kyiv'] == '2026-10-25T09:00:00+02:00'


def test_capture_rejects_stale_progress_and_omits_local_paths(tmp_path):
    log = tmp_path / 'train.log'
    log.write_text('2026-10-07 14:00:00,000 285epoch:train:1-10batch\n'
                   '2026-10-07 14:00:15,000 285epoch:train:11-20batch\n')
    (tmp_path / 'target_iterations.txt').write_text('478000')
    s = capture(log, now=datetime(2026, 10, 7, 14, 0, 20, tzinfo=timezone.utc))
    assert s['status'] == 'measured' and s['current_step'] == 284020
    assert s['seconds_per_step'] == 1.5 and s['scheduled_end_step'] == 478000
    assert str(tmp_path) not in str(s)
    stale = capture(log, now=datetime(2026, 10, 7, 15, tzinfo=timezone.utc))
    assert stale['status'] == 'stale' and target_eta(339000, stale)['status'] == 'unavailable'


def test_eta_export_and_html_label_hypothetical_forecasts():
    c = {'listening_ids': ['item'], 'rows': [
        {'label': 'Original source audio', 'train_mel': None, 'sigmos_10': 3.},
        {'label': 'Processed source audio', 'train_mel': None, 'sigmos_10': 3.5},
        *[{'label': f'{s}K', 'train_mel': 40., 'sigmos_10': 2+s/1000}
          for s in [100, 200, 300, 400]] ]}
    s = snapshot()
    s['current_step'] = 401000
    original = forecast(c)
    timed = attach(original, s)
    assert 'timing' not in original
    assert timed['scheduled_finish']['status'] == 'projected'
    text = forecast_html(timed)
    assert 'ETA snapshot:' in text and 'Kyiv' in text and '478,000 steps' in text
    assert 'hypothetical, beyond scheduled training' in text
    assert 'not a live countdown' in text
