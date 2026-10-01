import json
import threading
from pathlib import Path

import pytest

from training.quality import search
from training.quality.common import read_rows, write_tables


def test_concurrent_trials_keep_reports_separate(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    config = tmp_path / 'config.yaml'
    config.write_text('models: {}\n')
    metrics = search.QUALITY_METRICS + [
        'ecapa_similarity', 'cer', 'wer', 'clipping_fraction', 'hf_burst_count']

    def run(command, log=None):
        destination = Path(command[command.index('--output') + 1])
        if command[0] == 'process':
            barrier.wait(timeout=5)
        if command[0] == 'evaluate':
            value = 1. if 'identity' in str(destination) else 2.
            write_tables(destination, 'aggregate', [
                {'source_id': 'source_macro', 'metric': metric, 'mean': value}
                for metric in metrics])

    monkeypatch.setattr(search, 'run', run)
    monkeypatch.setattr('sys.argv', [
        'search', '--panel', str(tmp_path / 'panel.jsonl'), '--output', str(tmp_path / 'out'),
        '--config', str(config), '--profile-workers', '2',
        '--backends', 'identity', 'deepfilternet3', '--wet', '1'])
    search.main()
    rows = read_rows(tmp_path / 'out/combinations.jsonl')
    assert {r['profile']: r['sigmos_overall'] for r in rows} == {
        'identity_wet1': 1., 'deepfilternet3_wet1': 2.}
    report = json.loads((tmp_path / 'out/search_report.json').read_text())
    assert report['profiles'] == 2 and report['selected_profile'] is None


def test_rounding_collisions_rejected_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', [
        'search', '--panel', 'unused', '--output', str(tmp_path),
        '--backends', 'deepfilternet3', '--wet', '.5', '.50000001'])
    with pytest.raises(SystemExit):
        search.main()
    assert not list(tmp_path.iterdir())
