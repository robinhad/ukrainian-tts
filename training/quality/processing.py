"""Apply an explicit processing combination with the existing model adapters."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .common import digest, file_hash, read_rows, write_json, write_tables


def match_loudness_for_reporting(audio, reference, rate, target_lufs=None):
    from training.audio_enhancement.fair_comparison import (
        match_loudness_and_prevent_clipping, measure_lufs, PEAK_CEILING_DBFS)
    try:
        target = measure_lufs(reference, rate) if target_lufs is None else float(target_lufs)
        return match_loudness_and_prevent_clipping(audio, rate, target)
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


def process(panel, profile_path, output, device='cuda', resume=False, cpu_workers=4,
            num_shards=1, shard_index=0, report_tag=None):
    from training.audio_enhancement.fair_comparison import (
        atomic_write_pcm24, fit_sample_count)
    from training.scripts.run_fair_enhancement_comparison import make_backend, backend_channel
    from training.scripts.run_enhancement_review_backend import SidonBackend

    profile = yaml.safe_load(Path(profile_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(panel)
    if not 1 <= num_shards <= len(rows) or not 0 <= shard_index < num_shards:
        raise ValueError('Invalid processing shard count or index')
    if cpu_workers < 1:
        raise ValueError('cpu_workers must be positive')
    if len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Processing panel contains duplicate IDs')
    if not rows or any('voa' in str(r['source_id']).lower() for r in rows):
        raise ValueError('Processing requires a non-empty non-VOA panel')
    rows = rows[shard_index::num_shards]
    if report_tag is not None:
        import re
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', report_tag):
            raise ValueError('Invalid processing report tag')
    suffix = f'-{report_tag}' if report_tag else (f'-shard-{shard_index}' if num_shards > 1 else '')
    progress_path = output / f'progress{suffix}.json'
    wet = float(profile.get('wet', 1.0))
    random_seed = profile.get('random_seed')
    if random_seed is not None and (type(random_seed) is not int or random_seed < 0):
        raise ValueError('random_seed must be a nonnegative integer')
    torchscript_optimize = profile.get('torchscript_optimize')
    if torchscript_optimize is not None:
        if type(torchscript_optimize) is not bool:
            raise ValueError('torchscript_optimize must be a boolean')
        import torch
    if not 0 <= wet <= 1:
        raise ValueError('wet must be in [0, 1]')
    backend_name = profile['backend']
    output_lufs = profile.get('output_lufs')
    if output_lufs is not None and not -40 <= float(output_lufs) <= -10:
        raise ValueError('output_lufs must be in [-40, -10]')
    input_peak_dbfs = profile.get('input_peak_dbfs')
    if input_peak_dbfs is not None:
        input_peak_dbfs = float(input_peak_dbfs)
        if backend_name == 'identity' or not -30 <= input_peak_dbfs <= 0:
            raise ValueError('input_peak_dbfs requires an enhancement model and a value in [-30, 0]')
    if backend_name not in {'identity', 'rnnoise85'}:
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
        devices = {} if backend_name == 'rnnoise85' else require_gpu_models(backend)
        if 'rnnoise' in backend_name:
            devices['rnnoise'] = ['cpu']
        identity = {**identity, 'model_devices': devices}
        print(f'Enhancement model devices verified: {devices}', flush=True)
    identity = {**identity, 'adapter_hashes': {
        name: file_hash(root / name) for name in ['quality/processing.py',
        'quality/devices.py', 'scripts/run_enhancement_review_backend.py',
        'scripts/run_fair_enhancement_comparison.py']}}
    def finish(row, audio, enhanced, rate, target, metadata, input_hash, key, started, model_seconds, input_gain):
        post_started = time.monotonic()
        raw_output_frames = len(enhanced)
        enhanced = fit_sample_count(enhanced, len(audio))
        if not np.isfinite(enhanced).all():
            raise RuntimeError(f"Invalid processor output: {row['utterance_id']}")
        mixed = wet * enhanced + (1 - wet) * audio
        # Preserve the no-processing control exactly, apart from PCM24 encoding.
        loudness = None
        if backend_name != 'identity' or output_lufs is not None:
            mixed, loudness = match_loudness_for_reporting(mixed, audio, rate, output_lufs)
        atomic_write_pcm24(target, mixed, rate)
        item = {'utterance_id': row['utterance_id'], 'source_id': row['source_id'],
                'key': key, 'profile': profile, 'backend_identity': identity,
                'input_sha256': input_hash, 'output_sha256': file_hash(target),
                'output_path': str(target.resolve()), 'input_frames': len(audio),
                'processor_output_frames': raw_output_frames, 'output_frames': len(mixed),
                'processor_duration_changed': raw_output_frames != len(audio),
                'model_seconds': model_seconds,
                'model_input_gain_db': float(20 * np.log10(input_gain)),
                'cpu_postprocess_seconds': time.monotonic() - post_started,
                'cpu_postprocess_workers': cpu_workers,
                'loudness_match': loudness, 'elapsed_seconds': time.monotonic() - started}
        write_json(metadata, item)
        return item

    results, pending = [], deque()

    def collect():
        results.append(pending.popleft().result())
        write_json(progress_path, {'completed': len(results), 'expected': len(rows)})
        if len(results) % 8 == 0:
            # Native CPU allocators can retain large resampling workspaces.
            import ctypes
            import gc
            gc.collect()
            trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)

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
            input_gain = (10 ** (input_peak_dbfs / 20) / float(np.max(np.abs(audio)))
                          if input_peak_dbfs is not None else 1.)
            model_audio = audio * np.float32(input_gain) if input_peak_dbfs is not None else audio
            if random_seed is not None:
                # ClearVoice's Kaldi frontend adds random dither. Seed each file
                # independently so chunk sizes and worker order cannot change it.
                import random
                import torch
                seed = int(digest([random_seed, row['utterance_id']])[:8], 16)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            model_started = time.monotonic()
            # The profiled TorchScript executor can change Sidon numerics after
            # its first call. A profile can disable that optimization explicitly.
            with (torch.jit.optimized_execution(torchscript_optimize)
                  if torchscript_optimize is not None else nullcontext()):
                if backend_name == 'identity':
                    enhanced = audio.copy()
                elif backend_name == 'sidon':
                    result, result_rate = backend.process(model_audio[:, 0], rate)['sidon_no_compression']
                    from .backends import resample
                    enhanced = resample(np.asarray(result).reshape(-1), result_rate, rate)[:, None]
                else:
                    enhanced = backend_channel(backend_name, backend, model_audio[:, 0], rate)[:, None]
            if input_peak_dbfs is not None:
                # Blend in the original amplitude domain, not the normalized one.
                enhanced = enhanced / np.float32(input_gain)
            model_seconds = time.monotonic() - model_started
            pending.append(pool.submit(finish, row, audio, enhanced, rate, target, metadata,
                                       input_hash, key, started, model_seconds, input_gain))
            if len(pending) >= cpu_workers * 2:
                collect()
        while pending:
            collect()
    results.sort(key=lambda item: item['utterance_id'])
    write_json(progress_path, {'completed': len(results), 'expected': len(rows)})
    write_tables(output, f'processing{suffix}', results)
    return results
