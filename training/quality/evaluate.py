from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import digest, file_hash, read_rows, write_json, write_tables
from .metrics import content_scores, cosine, flags, signal_scores, windows

QUALITY_METRICS = ['sigmos_overall', 'sigmos_speech', 'sigmos_noise', 'sigmos_coloration',
                   'sigmos_discontinuity', 'audiobox_pq']
COMPARE_METRICS = QUALITY_METRICS + ['cer', 'wer', 'ecapa_similarity', 'clipping_fraction',
                                   'duration_ratio', 'hf_burst_count']


def aggregate(rows):
    groups = {'all': rows}
    for source in sorted({r['source_id'] for r in rows}):
        groups[source] = [r for r in rows if r['source_id'] == source]
    result = []
    for group, items in groups.items():
        for metric in COMPARE_METRICS:
            values = [r[metric] for r in items if r.get(metric) is not None]
            result.append({'source_id': group, 'metric': metric, 'count': len(values),
                           'expected_count': len(items),
                           'mean': float(np.mean(values)) if values else None,
                           'median': float(np.median(values)) if values else None,
                           'p05': float(np.percentile(values, 5)) if values else None,
                           'p95': float(np.percentile(values, 95)) if values else None})
        for metric, numerator, denominator in [('cer', 'character_errors', 'reference_characters'),
                                                ('wer', 'word_errors', 'reference_words')]:
            total = sum(r.get(denominator, 0) for r in items)
            result.append({'source_id': group, 'metric': f'corpus_{metric}', 'count': total,
                           'mean': sum(r.get(numerator, 0) for r in items) / total if total else None})
    for metric in COMPARE_METRICS:
        values = [r['mean'] for r in result if r['source_id'] != 'all'
                  and r['metric'] == metric and r['mean'] is not None]
        result.append({'source_id': 'source_macro', 'metric': metric, 'count': len(values),
                       'mean': float(np.mean(values)) if values else None})
    return result


def evaluate(panel, output, label, config, models, wav_dir=None, checkpoint=None,
             training_config=None, resume=False, allow_extra_wavs=False):
    panel, output = Path(panel), Path(output)
    if config.get('selection_mode') != 'report_only':
        raise ValueError('Only report_only selection is supported until thresholds are calibrated')
    records = read_rows(panel)
    if not records or len({r['utterance_id'] for r in records}) != len(records):
        raise ValueError('Panel is empty or contains duplicate IDs')
    if any('voa' in str(r['source_id']).lower() for r in records):
        raise ValueError('VOA is forbidden in this iteration')
    expected = {r['utterance_id'] for r in records}
    if wav_dir is not None:
        wav_dir = Path(wav_dir)
        actual = {p.stem for p in wav_dir.glob('*.wav')}
        if expected - actual or (actual - expected and not allow_extra_wavs):
            raise ValueError(f'Output coverage mismatch: missing={sorted(expected - actual)[:10]}, '
                             f'extra={sorted(actual - expected)[:10]}')
    provenance = {'schema_version': 1, 'label': label, 'panel_sha256': file_hash(panel),
                  'config': config, 'models': models.identity,
                  'checkpoint_sha256': file_hash(checkpoint) if checkpoint else None,
                  'training_config_sha256': file_hash(training_config) if training_config else None,
                  'selection_mode': 'report_only', 'hnr_rejection': False}
    run_key = digest(provenance)
    metadata_path = output / 'run.json'
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text())
        if not resume or prior['run_key'] != run_key:
            raise ValueError('Existing report has different provenance or --resume was not set')
    output.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, {**provenance, 'run_key': run_key, 'status': 'running'})
    rows, segment_rows, errors = [], [], []
    thresholds = config['thresholds']
    for record in records:
        identifier = record['utterance_id']
        reference_path = Path(record['audio_path'])
        candidate_path = wav_dir / f'{identifier}.wav' if wav_dir else reference_path
        cache_path = output / 'cache' / f'{digest(identifier)}.json'
        try:
            reference_hash = file_hash(reference_path)
            if reference_hash != record['reference_sha256']:
                raise ValueError('Frozen reference audio changed')
            candidate_hash = file_hash(candidate_path)
            cache_key = digest([run_key, reference_hash, candidate_hash, record])
            if resume and cache_path.exists():
                cached = json.loads(cache_path.read_text())
                if cached['key'] == cache_key:
                    rows.append(cached['file'])
                    segment_rows.extend(cached['segments'])
                    continue
            original, original_rate = sf.read(reference_path, dtype='float32', always_2d=True)
            audio, rate = sf.read(candidate_path, dtype='float32', always_2d=True)
            signal_scores(original, original_rate, thresholds)
            signal = signal_scores(audio, rate, thresholds)
            mono, reference_mono = audio.mean(axis=1), original.mean(axis=1)
            reference_embedding = models.embedding(reference_mono, original_rate)
            quality = models.quality(mono, rate)
            hypothesis = models.transcribe(mono, rate)
            metrics = {**signal, **quality, **content_scores(record['text'], hypothesis),
                       'duration_ratio': (len(audio) / rate) / (len(original) / original_rate),
                       'ecapa_similarity': cosine(models.embedding(mono, rate), reference_embedding)}
            if any(not np.isfinite(metrics[k]) for k in QUALITY_METRICS + ['ecapa_similarity']):
                raise ValueError('Model returned non-finite scores')
            segments = []
            for index, (start, end) in enumerate(windows(len(audio), rate, config['segment_seconds'],
                                                        config['segment_stride_seconds'])):
                crop = mono[start:end]
                values = {**models.quality(crop, rate),
                          **signal_scores(audio[start:end], rate, thresholds),
                          'ecapa_similarity': cosine(models.embedding(crop, rate), reference_embedding)}
                if any(not np.isfinite(values[k]) for k in QUALITY_METRICS + ['ecapa_similarity']):
                    raise ValueError('Model returned non-finite segment scores')
                segments.append({'utterance_id': identifier, 'source_id': record['source_id'],
                                 'label': label, 'segment': index, 'start_seconds': start / rate,
                                 'end_seconds': end / rate, **values,
                                 'flags': flags(values, thresholds),
                                 'content_score_status': 'not_scored_no_time_aligned_reference',
                                 'speaker_reference': 'whole_original_recording',
                                 'short_segment': len(crop) / rate < 3.0})
            worst = {metric: [{'start_seconds': s['start_seconds'], 'end_seconds': s['end_seconds'],
                               'score': s[metric]} for s in sorted(segments, key=lambda x: x[metric])
                              [:config['worst_segments']]] for metric in QUALITY_METRICS + ['ecapa_similarity']}
            row = {'utterance_id': identifier, 'source_id': record['source_id'], 'label': label,
                   'text': record['text'], 'whisper_text': hypothesis,
                   'audio_path': str(candidate_path.resolve()), 'audio_sha256': candidate_hash,
                   'reference_sha256': reference_hash, 'sample_rate': rate, 'channels': audio.shape[1],
                   **metrics, 'flags': flags(metrics, thresholds), 'worst_segments': worst}
            write_json(cache_path, {'key': cache_key, 'file': row, 'segments': segments})
            rows.append(row)
            segment_rows.extend(segments)
        except Exception as error:
            errors.append({'utterance_id': identifier, 'error': f'{type(error).__name__}: {error}'})
        # Persist partial progress after every file; interrupted evaluations can resume.
        write_json(output / 'progress.json', {'completed': len(rows), 'failed': len(errors),
                                              'expected': len(records), 'last_id': identifier})
    write_tables(output, 'per_file', rows)
    write_tables(output, 'segments', segment_rows)
    write_tables(output, 'aggregate', aggregate(rows))
    write_tables(output, 'errors', errors)
    write_json(metadata_path, {**provenance, 'run_key': run_key,
                              'status': 'complete' if not errors else 'incomplete',
                              'expected': len(records), 'scored': len(rows), 'errors': len(errors)})
    if errors:
        raise RuntimeError(f'{len(errors)} files failed; partial scores saved in {output}')
    return rows
