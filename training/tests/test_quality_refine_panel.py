import json

import pytest

from training.quality.common import read_rows, write_tables
from training.quality.evaluate import evaluate
from training.quality.refine_panel import derive_report, select
from training.tests.test_quality import CONFIG, FakeModels, panel


def test_derived_reports_preserve_measurements_and_reject_changed_audio(tmp_path):
    source_panel, audio, waveform = panel(tmp_path)
    rows = read_rows(source_panel)
    rows[0]['split'] = 'v10_train'
    write_tables(tmp_path, 'parent', rows)
    parent = tmp_path / 'parent.jsonl'
    full = tmp_path / 'full'
    evaluate(parent, full, 'original', CONFIG, FakeModels())
    subset = select(parent, tmp_path / 'subset', per_source=1)
    derived = tmp_path / 'derived'
    derive_report(full, subset, derived)
    assert read_rows(derived / 'per_file.jsonl') == read_rows(full / 'per_file.jsonl')
    assert read_rows(derived / 'aggregate.jsonl') == read_rows(full / 'aggregate.jsonl')
    assert json.loads((derived / 'run.json').read_text())['derived_from']['method'].endswith('without_rescoring')
    assert select(parent, tmp_path / 'subset', per_source=1) == subset
    with pytest.raises(ValueError, match='frozen'):
        select(parent, tmp_path / 'subset', per_source=1, seed=42)
    import soundfile as sf
    sf.write(audio, waveform * .5, 24000)
    with pytest.raises(ValueError, match='audio changed'):
        derive_report(full, subset, tmp_path / 'stale')


def test_refinement_cannot_include_heldout_items(tmp_path):
    parent, _, _ = panel(tmp_path)
    with pytest.raises(ValueError, match='training items only'):
        select(parent, tmp_path / 'subset', per_source=1)
