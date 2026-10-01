"""Apply an explicit processing combination with the existing model adapters."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .common import digest, file_hash, read_rows, write_json, write_tables


def match_loudness_for_reporting(audio, reference, rate):
    from training.audio_enhancement.fair_comparison import (
        match_loudness_and_prevent_clipping, measure_lufs, PEAK_CEILING_DBFS)
    try:
        return match_loudness_and_prevent_clipping(audio, rate, measure_lufs(reference, rate))
    except RuntimeError as error:
        if str(error) != 'invalid integrated loudness: -inf':
            raise
        # Suppressed speech is a result to measure, not a reason to drop a file
        # or invent an unbounded loudness gain below the EBU measurement gate.
        peak = float(np.max(np.abs(audio)))
        gain = min(1., 10 ** (PEAK_CEILING_DBFS / 20) / max(peak, 1e-12))
        return audio * np.float32(gain), {
            'method': 'peak_cap_only', 'loudness_match_status': 'below_measurement_gate',
            'applied_linear_gain': gain, 'ceiling_dbfs': PEAK_CEILING_DBFS,
            'report_only': True}


def process(panel, profile_path, output, device='cuda', resume=False, cpu_workers=4):
    from training.audio_enhancement.fair_comparison import (
        atomic_write_pcm24, fit_sample_count)
    from training.scripts.run_fair_enhancement_comparison import make_backend, backend_channel
    from training.scripts.run_enhancement_review_backend import SidonBackend

    profile = yaml.safe_load(Path(profile_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(panel)
    if cpu_workers < 1:
        raise ValueError('cpu_workers must be positive')
    if len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Processing panel contains duplicate IDs')
    if not rows or any('voa' in str(r['source_id']).lower() for r in rows):
        raise ValueError('Processing requires a non-empty non-VOA panel')
    wet = float(profile.get('wet', 1.0))
    if not 0 <= wet <= 1:
        raise ValueError('wet must be in [0, 1]')
    backend_name = profile['backend']
    if 'rnnoise' in backend_name:
        raise ValueError('The bundled RNNoise executable runs its neural model on CPU; '
                         'select a GPU enhancement profile for this iteration')
    if backend_name != 'identity':
        import torch
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Enhancement models require a SLURM GPU allocation')
        torch.cuda.set_device(torch.device(device).index or 0)
    identity, backend = {'name': backend_name}, None
    if backend_name == 'sidon':
        backend = SidonBackend(device, Path(profile['model_cache']) / 'sidon')
        identity = backend.identity
    elif backend_name != 'identity':
        backend, identity = make_backend(argparse.Namespace(
            backend=backend_name, device=device, model_cache=Path(profile['model_cache']),
            rnnoise_binary=Path(profile.get('rnnoise_binary', 'training/vendor/rnnoise/examples/rnnoise_demo'))))
    root = Path(__file__).resolve().parents[1]
    if backend is not None:
        from .devices import require_gpu_models
        identity = {**identity, 'model_devices': require_gpu_models(backend)}
        print(f'GPU enhancement models verified: {identity["model_devices"]}', flush=True)
    identity = {**identity, 'adapter_hashes': {
        name: file_hash(root / name) for name in ['quality/processing.py',
        'quality/devices.py', 'scripts/run_enhancement_review_backend.py',
        'scripts/run_fair_enhancement_comparison.py']}}
    def finish(row, audio, enhanced, rate, target, metadata, input_hash, key, started, model_seconds):
        post_started = time.monotonic()
        raw_output_frames = len(enhanced)
        enhanced = fit_sample_count(enhanced, len(audio))
        if not np.isfinite(enhanced).all():
            raise RuntimeError(f"Invalid processor output: {row['utterance_id']}")
        mixed = wet * enhanced + (1 - wet) * audio
        # Preserve the no-processing control exactly, apart from PCM24 encoding.
        loudness = None
        if backend_name != 'identity':
            mixed, loudness = match_loudness_for_reporting(mixed, audio, rate)
        atomic_write_pcm24(target, mixed, rate)
        item = {'utterance_id': row['utterance_id'], 'source_id': row['source_id'],
                'key': key, 'profile': profile, 'backend_identity': identity,
                'input_sha256': input_hash, 'output_sha256': file_hash(target),
                'output_path': str(target.resolve()), 'input_frames': len(audio),
                'processor_output_frames': raw_output_frames, 'output_frames': len(mixed),
                'processor_duration_changed': raw_output_frames != len(audio),
                'model_seconds': model_seconds,
                'cpu_postprocess_seconds': time.monotonic() - post_started,
                'cpu_postprocess_workers': cpu_workers,
                'loudness_match': loudness, 'elapsed_seconds': time.monotonic() - started}
        write_json(metadata, item)
        return item

    results, pending = [], deque()

    def collect():
        results.append(pending.popleft().result())
        write_json(output / 'progress.json', {'completed': len(results), 'expected': len(rows)})

    # Keep GPU model calls on one thread: some enhancement models have mutable
    # recurrent state. Overlap their execution with bounded CPU loudness/I/O work.
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        for row in rows:
            while pending and pending[0].done():
                collect()
            if shutil.disk_usage(output).free < 30 * 1024**3:
                raise RuntimeError('Disk reserve below 30 GiB')
            source = Path(row.get('processing_audio_path') or row['audio_path'])
            input_hash = file_hash(source)
            if (row.get('processing_input_sha256') or row.get('reference_sha256', input_hash)) != input_hash:
                raise ValueError('Frozen input changed')
            target = output / f"{row['utterance_id']}.wav"
            metadata = output / f"{row['utterance_id']}.json"
            key = digest([input_hash, profile, identity])
            if resume and target.is_file() and metadata.is_file():
                old = json.loads(metadata.read_text())
                if old['key'] == key and old['output_sha256'] == file_hash(target):
                    results.append(old)
                    continue
            started = time.monotonic()
            audio, rate = sf.read(source, dtype='float32', always_2d=True)
            if rate != 24000 or audio.shape[1] != 1:
                raise ValueError('Processing inputs must be canonical mono 24 kHz audio')
            if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
                raise ValueError(f"Invalid processing input: {row['utterance_id']}")
            model_started = time.monotonic()
            if backend_name == 'identity':
                enhanced = audio.copy()
            elif backend_name == 'sidon':
                result, result_rate = backend.process(audio[:, 0], rate)['sidon_no_compression']
                from .backends import resample
                enhanced = resample(np.asarray(result).reshape(-1), result_rate, rate)[:, None]
            else:
                enhanced = backend_channel(backend_name, backend, audio[:, 0], rate)[:, None]
            model_seconds = time.monotonic() - model_started
            pending.append(pool.submit(finish, row, audio, enhanced, rate, target, metadata,
                                       input_hash, key, started, model_seconds))
            if len(pending) >= cpu_workers * 2:
                collect()
        while pending:
            collect()
    results.sort(key=lambda item: item['utterance_id'])
    write_json(output / 'progress.json', {'completed': len(results), 'expected': len(rows)})
    write_tables(output, 'processing', results)
    return results
