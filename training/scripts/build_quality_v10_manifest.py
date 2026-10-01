#!/usr/bin/env python3
"""Bind the reviewed full-corpus processing outputs to the preserved raw splits."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from training.quality.common import file_hash, read_rows, write_json


def json_value(value):
    # Parquet list columns become NumPy arrays when pandas returns records.
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported manifest value: {type(value).__name__}')


def build_manifest(manifest, processed, profile, destination):
    rows = read_rows(manifest)
    profile_hash = file_hash(profile)
    expected_profile = yaml.safe_load(Path(profile).read_text())
    if not rows or any('voa' in str(r['source_id']).lower() for r in rows):
        raise ValueError('Expected only non-VOA records')
    output = []
    for row in rows:
        path = Path(processed) / f"{row['utterance_id']}.wav"
        metadata = json.loads(path.with_suffix('.json').read_text())
        info = sf.info(path)
        if (info.samplerate, info.channels, info.subtype) != (24000, 1, 'PCM_24'):
            raise ValueError(f"Invalid PCM24 output: {row['utterance_id']}")
        if metadata['input_sha256'] != file_hash(row['audio_path']) or metadata['output_sha256'] != file_hash(path):
            raise ValueError('Stale processing output')
        if metadata['profile'] != expected_profile:
            raise ValueError('Processing profile mismatch')
        output.append({**row, 'audio_path': str(path.resolve()), 'audio_sha256': metadata['output_sha256'],
                       'duration': info.duration, 'channels': 1, 'sample_rate': 24000,
                       'format': 'WAV/PCM_24', 'processing_profile_sha256': profile_hash})
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp',
                                            dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            for row in output:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                        default=json_value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    write_json(destination.with_suffix('.summary.json'), {'records': len(output), 'voa_records': 0,
               'hours': sum(r['duration'] for r in output) / 3600, 'processing_profile_sha256': profile_hash})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_manifest(args.manifest, args.processed, args.profile, args.output)


if __name__ == '__main__':
    main()
