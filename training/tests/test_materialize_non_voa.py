from __future__ import annotations

import io
import json

import numpy as np
import soundfile as sf

from training.scripts.materialize_non_voa import convert


def test_native_audio_conversion_keeps_pcm24_and_global_text_split(tmp_path):
    stream = io.BytesIO()
    samples = .1 * np.sin(2 * np.pi * 300 * np.arange(48000 * 3) / 48000)
    sf.write(stream, samples, 48000, subtype='PCM_24', format='WAV')
    source = {'repo_id': 'test/source', 'revision': 'fixed', 'license': 'CC0-1.0'}
    raw = {'audio': {'bytes': stream.getvalue(), 'path': 'a.wav'}, 'text': 'Привіт, світе!'}
    row, error = convert(('opentts_lada', source, raw, tmp_path))
    assert error is None
    assert sf.info(row['audio_path']).subtype == 'PCM_24'
    assert row['duration'] == 3
    assert row['speaker_id'] == 'lada'
    other, error = convert(('cv', source, {**raw, 'text': 'Привіт світе'}, tmp_path))
    assert error is None
    assert row['split'] == other['split']
    assert row['text_sha256_source'] == other['text_sha256_source']
    resumed, _ = convert(('opentts_lada', source, raw, tmp_path))
    assert resumed == row


def test_unusable_audio_is_reported(tmp_path):
    row, error = convert(('source', {}, {'audio': {'bytes': b'bad'}, 'text': 'Привіт'}, tmp_path))
    assert row is None
    assert error
