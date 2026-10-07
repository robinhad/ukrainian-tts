import pytest

from training.scripts.forecast_listening_quality import forecast, forecast_html


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


def test_lower_bound_is_fitted_separately_and_rendered_with_median():
    c = comparison([2.1, 2.2, 2.3, 2.4])
    c['listening_ids'] = ['a', 'b', 'c']
    c['rows'][0]['sigmos_10_values'] = [2., 3., 4.]
    c['rows'][1]['sigmos_10_values'] = [3., 3.5, 4.]
    for row in c['rows'][2:]:
        value = row['sigmos_10']
        row['sigmos_10_values'] = [value-.1, value, value+.9]
    result = forecast(c)
    lower = result['lower_bound']
    assert lower['statistic'] == 'p05'
    assert lower['current_score'] == pytest.approx(2.31)
    assert lower['targets']['processed'] == pytest.approx(3.05)
    assert lower['scenarios'][0]['targets']['processed']['total_steps'] == pytest.approx(1140000)
    assert result['scenarios'][0]['targets']['processed']['total_steps'] == pytest.approx(1500000)
    text = forecast_html(result)
    assert '1,140K total' in text and '1,500K total' in text
    assert 'P05 lower bound' in text and 'Current' in text


def test_missing_per_item_scores_do_not_fabricate_lower_forecast():
    c = comparison([2.1, 2.2, 2.3, 2.4])
    result = forecast(c)
    assert result['lower_bound'] is None
    assert 'P05 prediction unavailable' in forecast_html(result)
    with pytest.raises(ValueError, match='requires per-item scores'):
        forecast(c, 'p05')
