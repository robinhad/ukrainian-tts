import pytest

from training.scripts.forecast_listening_quality import forecast


def comparison(scores):
    return {'listening_ids': ['sample'], 'rows': [
        {'label': 'Original source audio', 'train_mel': None, 'sigmos_10': 3.0},
        {'label': 'Processed source audio', 'train_mel': None, 'sigmos_10': 3.5},
        *[{'label': f'{step}K', 'train_mel': 99 - step / 100, 'sigmos_10': score}
          for step, score in zip([100, 200, 300, 400], scores)]]}


def test_linear_crossing_uses_steps_and_reports_additional_from_last_scored():
    result = forecast(comparison([2.1, 2.2, 2.3, 2.4]))
    targets = result['scenarios'][0]['targets']
    assert targets['original']['total_steps'] == pytest.approx(1000000)
    assert targets['original']['additional_steps'] == pytest.approx(600000)
    assert targets['processed']['total_steps'] == pytest.approx(1500000)


@pytest.mark.parametrize('scores', [[2.4, 2.4, 2.4, 2.4], [2.4, 2.3, 2.2, 2.1]])
def test_no_crossing_claim_for_flat_or_declining_quality(scores):
    result = forecast(comparison(scores))
    assert all(s['targets']['original']['total_steps'] is None for s in result['scenarios'])


def test_observed_target_is_not_presented_as_future_prediction():
    result = forecast(comparison([2.8, 2.9, 3.1, 3.2]))
    assert result['scenarios'][0]['targets']['original'] == {
        'status': 'observed', 'total_steps': 300000, 'additional_steps': 0}
