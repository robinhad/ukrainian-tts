"""Recover native, untrimmed recordings for the frozen evaluation panels."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import av
import numpy as np
import soundfile as sf

from .common import file_hash


def decode_native(content):
    with av.open(io.BytesIO(content)) as container:
        container.streams.audio[0].codec_context.thread_count = 1
        resampler = av.AudioResampler(format='fltp')
        frames, rate = [], None
        for frame in container.decode(audio=0):
            if rate is not None and frame.sample_rate != rate:
                raise ValueError('Native recording changes sample rate')
            rate = frame.sample_rate
            frames.extend(part.to_ndarray().T for part in resampler.resample(frame))
        frames.extend(part.to_ndarray().T for part in resampler.resample(None))
    if not frames:
        raise ValueError('Empty native recording')
    audio = np.concatenate(frames).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError('Non-finite native recording')
    return audio, rate


def recover(panels, downloads, output):
    from training.scripts.materialize_non_voa import input_rows

    sources = json.loads(Path(downloads).read_text())
    selected = [row for panel in panels.values() for row in panel]
    destination = Path(output) / 'native_references'
    destination.mkdir(parents=True, exist_ok=True)
    for source_id in sorted({row['source_id'] for row in selected}):
        pending = {row['audio_sha256_source']: row for row in selected if row['source_id'] == source_id}
        for raw in input_rows(sources[source_id]):
            content = (raw.get('audio') or {}).get('bytes')
            if content is None:
                continue
            source_hash = hashlib.sha256(content).hexdigest()
            if source_hash not in pending:
                continue
            row = pending.pop(source_hash)
            audio, rate = decode_native(content)
            path = destination / f'{row["utterance_id"]}.wav'
            sf.write(path, audio, rate, subtype='FLOAT')
            row['processing_audio_path'] = row['audio_path']
            row['processing_input_sha256'] = row['reference_sha256']
            row['audio_path'] = str(path.resolve())
            row['reference_sha256'] = file_hash(path)
            row['reference_kind'] = 'native_untrimmed_decoded_recording'
            if not pending:
                break
        if pending:
            raise ValueError(f'Native reference coverage incomplete for {source_id}: {len(pending)} missing')
        print(f'Native references recovered: {source_id}', flush=True)
