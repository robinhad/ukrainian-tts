import json

import numpy as np
import pytest

from training.scripts.listening_metric_correlation import analyze, correlation, correlation_html, plot


def fixture():
    steps = [25, 50, 75, 100, 125]
    scores = [2.0, 2.8, 2.3, 2.6, 3.0]
    comparison = {'through_step': 125000, 'listening_ids': ['fixed'], 'rows': [
        {'label': 'Original source audio', 'train_mel': None, 'sigmos_10': 4.0},
        *[{'label': f'{s}K', 'train_mel': 10-i, 'sigmos_10': y, 'sigmos_88': None}
          for i, (s, y) in enumerate(zip(steps, scores))]]}
    epochs = [{'training_step': s*1000, 'valid_winner_loss': 5-y,
               'train_generator_g_mel_loss': 10-i, 'train_constant_loss': 1,
               'train_time': y, 'valid_incomplete_loss': y if i else None}
              for i, (s, y) in enumerate(zip(steps, scores))]
    return comparison, epochs


def test_absolute_ranking_uses_same_checkpoints_and_excludes_incomplete_metrics():
    c, epochs = fixture()
    r = analyze(c, epochs[::-1])
    assert r['selected_metric'] == 'valid_winner_loss'
    assert r['ranking'][0]['pearson_r'] == pytest.approx(-1)
    assert r['ranking'][0]['spearman_rho'] == pytest.approx(-1)
    assert r['ranking'][0]['partial_r_controlling_steps'] == pytest.approx(-1)
    assert r['candidate_count'] == 4 and r['ranked_count'] == 2
    assert r['excluded_metrics'] == ['train_constant_loss', 'valid_incomplete_loss']
    assert len(r['points']) == 5
    assert r['panel88'] == {'count': 0, 'pearson_r': None}


def test_missing_epoch_or_duplicate_step_is_rejected():
    c, epochs = fixture()
    with pytest.raises(ValueError, match='Missing epoch'):
        analyze(c, epochs[:-1])
    with pytest.raises(ValueError, match='unique'):
        analyze(c, epochs + [epochs[0]])


def test_correlations_are_undefined_for_degenerate_data():
    assert correlation([1, 1, 1], [2, 3, 4]) is None
    assert correlation([1, 2, np.nan], [2, 3, 4]) is None
    assert correlation([1, 2], [2, 3]) is None


def test_html_rejects_changed_comparison_scores(tmp_path):
    c, epochs = fixture()
    (tmp_path/'metric_correlation.json').write_text(json.dumps(analyze(c, epochs)))
    c['rows'][-1]['sigmos_10'] += .1
    with pytest.raises(ValueError, match='does not match'):
        correlation_html(tmp_path, c)


def test_score_dispersion_uses_matched_items_and_preserves_correlations():
    c, epochs = fixture()
    baseline = analyze(c, epochs)
    c['listening_ids'] = ['first', 'second', 'third']
    for row in c['rows'][1:]:
        row['sigmos_10_values'] = [row['sigmos_10'] - .5, row['sigmos_10'], row['sigmos_10'] + .5]
    result = analyze(c, epochs)
    assert result['ranking'] == baseline['ranking']
    assert all(p['sigmos_stddev'] == pytest.approx(np.sqrt(1/6)) for p in result['points'])
    svg = plot(result)
    assert svg.count('<path class="corr-error') == 5
    assert svg.count('tabindex="0"') == 5
    assert 'SD 0.40825' in svg
    # Missing measurements stay missing instead of implying zero dispersion.
    del c['rows'][1]['sigmos_10_values']
    assert analyze(c, epochs)['points'][0]['sigmos_stddev'] is None


@pytest.mark.parametrize('values', [[2.0], [1., 2., float('nan')], [1., 1., 1.]])
def test_score_dispersion_rejects_wrong_panel_nonfinite_or_mismatched_median(values):
    c, epochs = fixture()
    c['listening_ids'] = ['first', 'second', 'third']
    c['rows'][1]['sigmos_10_values'] = values
    with pytest.raises(ValueError, match='Per-item SigMOS'):
        analyze(c, epochs)
