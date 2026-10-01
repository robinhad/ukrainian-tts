"""Transparent diagnostics. All thresholds are flags, never rejection rules."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft


def normalize_text(text):
    text = unicodedata.normalize('NFC', text).lower()
    text = text.translate(str.maketrans({'’': "'", 'ʼ': "'", '`': "'"}))
    text = ''.join(c for c in text if c not in '+\u0301')
    text = ''.join(c if c.isalnum() or c in "' " else ' ' for c in text)
    return re.sub(r'\s+', ' ', text).strip()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def content_scores(reference, hypothesis):
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    chars = ref.replace(' ', '')
    words = ref.split()
    ce = edit_distance(chars, hyp.replace(' ', ''))
    we = edit_distance(words, hyp.split())
    return {'cer': ce / len(chars) if chars else None, 'wer': we / len(words) if words else None,
            'character_errors': ce, 'reference_characters': len(chars),
            'word_errors': we, 'reference_words': len(words)}


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('Invalid ECAPA embedding')
    return float(np.clip(np.dot(a, b) / norm, -1, 1))


def windows(length, rate, seconds, stride):
    if seconds <= 0 or stride <= 0 or stride > seconds:
        raise ValueError('Require 0 < stride <= segment_seconds')
    size, hop = max(1, round(seconds * rate)), max(1, round(stride * rate))
    starts = list(range(0, max(1, length - size + 1), hop))
    if length > size and starts[-1] + size < length:
        starts.append(length - size)
    return [(start, min(length, start + size)) for start in starts]


def signal_scores(audio, rate, thresholds):
    audio = np.asarray(audio)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError('Empty or non-finite waveform')
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    frame = max(16, round(rate * .02))
    padded = np.pad(mono, (0, max(0, frame - len(mono))))
    freq, times, spectrum = stft(padded, fs=rate, nperseg=frame, noverlap=frame // 2,
                                 boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    cutoff = thresholds['hf_cutoff_hz']
    if rate / 2 <= cutoff:
        bursts, max_excess, max_ratio = [], None, None
    else:
        hf = power[freq >= cutoff].sum(axis=0)
        total = power.sum(axis=0)
        db = 10 * np.log10(np.maximum(hf, 1e-15))
        local = median_filter(db, size=101, mode='nearest')
        excess = db - local
        ratio = hf / np.maximum(total, 1e-15)
        active = (ratio > thresholds['hf_ratio']) & (excess > thresholds['hf_excess_db'])
        active &= 10 * np.log10(np.maximum(total, 1e-15)) > thresholds['hf_floor_dbfs']
        bursts = []
        for index in np.flatnonzero(active):
            start, end = max(0., float(times[index] - .01)), min(len(mono) / rate, float(times[index] + .01))
            if bursts and start <= bursts[-1]['end_seconds'] + .011:
                bursts[-1]['end_seconds'] = end
            else:
                bursts.append({'start_seconds': start, 'end_seconds': end})
        max_excess, max_ratio = float(excess.max()), float(ratio.max())
    clipped = np.abs(audio) >= thresholds['clip_amplitude']
    return {'duration_seconds': len(audio) / rate, 'peak_absolute': float(np.abs(audio).max()),
            'clipping_fraction': float(clipped.mean()), 'clipping_samples': int(clipped.sum()),
            'hf_burst_count': len(bursts), 'hf_bursts': bursts,
            'hf_max_excess_db': max_excess, 'hf_max_ratio': max_ratio,
            'hf_available': rate / 2 > cutoff}


def flags(metrics, thresholds):
    result = []
    if metrics.get('clipping_samples', 0):
        result.append('clipping')
    if abs(metrics.get('duration_ratio', 1.) - 1.) > thresholds['duration_relative_change']:
        result.append('duration_change')
    if metrics.get('hf_burst_count', 0):
        result.append('local_high_frequency_burst')
    for name, bound in thresholds.get('minimum', {}).items():
        if metrics.get(name) is not None and metrics[name] < bound:
            result.append(f'{name}_below_threshold')
    for name, bound in thresholds.get('maximum', {}).items():
        if metrics.get(name) is not None and metrics[name] > bound:
            result.append(f'{name}_above_threshold')
    return result
