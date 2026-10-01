from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from training.quality.common import file_hash, read_rows, write_tables
from training.quality.compare import compare
from training.quality.evaluate import evaluate, QUALITY_METRICS
from training.quality.metrics import content_scores, signal_scores, windows
from training.quality.selection import select


CONFIG = yaml.safe_load((Path(__file__).parents[1] / 'conf/quality.yaml').read_text())


class FakeModels:
    """Only exercises orchestration; never used to report real quality results."""
    identity = {'test_backend': True}

    def quality(self, audio, rate):
        return dict.fromkeys(QUALITY_METRICS, float(3 + np.std(audio)))

    def embedding(self, audio, rate):
        return np.array([1., float(np.std(audio))])

    def transcribe(self, audio, rate):
        return 'Привіт світе'


def panel(tmp_path):
    audio = tmp_path / 'original.wav'
    rate = 24000
    waveform = .1 * np.sin(2 * np.pi * 220 * np.arange(rate * 4) / rate)
    sf.write(audio, waveform, rate, subtype='PCM_24')
    path = tmp_path / 'panel.jsonl'
    write_tables(tmp_path, 'panel', [{'utterance_id': 'one', 'source_id': 'lada',
                                    'audio_path': str(audio), 'reference_sha256': file_hash(audio),
                                    'text': 'Привіт, світе!', 'split': 'v10_eval'}])
    return path, audio, waveform


def test_ukrainian_content_normalization_and_deletions():
    assert content_scores('П’ять, слі́в!', "п'ять слів")['wer'] == 0
    result = content_scores('Привіт світе', 'Привіт')
    assert result['wer'] == .5
    assert result['character_errors'] == 5
    assert content_scores('', 'текст')['wer'] is None


def test_diagnostics_detect_clipping_and_local_hf_burst():
    rate = 24000
    waveform = .05 * np.sin(2 * np.pi * 220 * np.arange(rate * 3) / rate)
    waveform[rate:rate + 1200] += .8 * np.sin(2 * np.pi * 8000 * np.arange(1200) / rate)
    waveform[0] = 1.
    result = signal_scores(waveform, rate, CONFIG['thresholds'])
    assert result['clipping_samples'] == 1
    assert result['hf_burst_count'] >= 1
    assert any(.95 < burst['start_seconds'] < 1.1 for burst in result['hf_bursts'])
    assert signal_scores(waveform[:100], 8000, CONFIG['thresholds'])['hf_available'] is False


def test_windows_cover_tail_and_short_clips():
    assert windows(50, 10, 3, 1.5) == [(0, 30), (15, 45), (20, 50)]
    assert windows(10, 10, 3, 1.5) == [(0, 10)]
    with pytest.raises(ValueError):
        windows(50, 10, 1, 2)


def test_evaluation_resume_comparison_and_report_only(tmp_path):
    path, original, waveform = panel(tmp_path)
    reference = tmp_path / 'reference'
    current = tmp_path / 'current'
    wavs = tmp_path / 'wavs'
    wavs.mkdir()
    sf.write(wavs / 'one.wav', waveform * 3, 24000)
    evaluate(path, reference, 'original', CONFIG, FakeModels())
    evaluate(path, current, '25k', CONFIG, FakeModels(), wavs)
    assert read_rows(current / 'per_file.jsonl')[0]['wer'] == 0
    assert read_rows(current / 'segments.jsonl')[0]['content_score_status'].startswith('not_scored')
    assert (current / 'aggregate.csv').exists()
    summary = compare(current, {'original': reference, 'previous': reference, 'best': reference}, current)
    assert any(r['mean_delta'] > 0 for r in summary if r['metric'] == 'sigmos_overall')
    assert json.loads((current / 'comparison.json').read_text())['automatic_promotion'] is False
    evaluate(path, current, '25k', CONFIG, FakeModels(), wavs, resume=True)
    with pytest.raises(ValueError, match='provenance'):
        evaluate(path, current, '50k', CONFIG, FakeModels(), wavs, resume=True)
    sf.write(original, waveform * .5, 24000)
    with pytest.raises(RuntimeError):
        evaluate(path, current, '25k', CONFIG, FakeModels(), wavs, resume=True)
    assert json.loads((current / 'run.json').read_text())['status'] == 'incomplete'


def test_coverage_failure_and_incompatible_baseline(tmp_path):
    path, _, waveform = panel(tmp_path)
    wavs = tmp_path / 'wavs'
    wavs.mkdir()
    with pytest.raises(ValueError, match='coverage'):
        evaluate(path, tmp_path / 'missing', 'bad', CONFIG, FakeModels(), wavs)
    ref, candidate = tmp_path / 'ref', tmp_path / 'candidate'
    evaluate(path, ref, 'ref', CONFIG, FakeModels())
    changed = {**CONFIG, 'segment_seconds': 4}
    evaluate(path, candidate, 'candidate', changed, FakeModels())
    with pytest.raises(ValueError, match='incompatible'):
        compare(candidate, {'original': ref}, tmp_path / 'compare')


def test_balanced_frozen_non_voa_selection_and_leakage(tmp_path):
    rows = []
    for source in ['lada', 'cv', 'voa']:
        for split in ['train', 'dev', 'eval']:
            for i in range(3):
                identifier = f'{source}_{split}_{i}'
                audio = tmp_path / f'{identifier}.wav'
                # Distinct recording hashes, independent of file names.
                sf.write(audio, np.full(240, (len(rows) + 1) / 100), 24000)
                rows.append({'utterance_id': identifier, 'source_id': source,
                             'split': split, 'audio_path': str(audio), 'speaker_id': source,
                             'text_raw': identifier.replace('_', ' ')})
    # One held-out item leaks a training transcript. It must not enter the panel.
    rows[6]['text_raw'] = rows[0]['text_raw']
    write_tables(tmp_path, 'input', rows)
    output = tmp_path / 'selection'
    report = select(tmp_path / 'input.jsonl', output, per_source=2)
    assert report['heldout_overlap_excluded'] == 1
    assert len(read_rows(output / 'processing.jsonl')) == 4
    assert len(read_rows(output / 'heldout.jsonl')) == 4
    assert all(r['source_id'] != 'voa' for r in read_rows(output / 'heldout.jsonl'))
    with pytest.raises(ValueError, match='overwrite'):
        select(tmp_path / 'input.jsonl', output)
    with pytest.raises(ValueError):
        select(tmp_path / 'input.jsonl', tmp_path / 'other', per_source=101)
