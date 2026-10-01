#!/usr/bin/env python3
"""Bind the reviewed full-corpus processing outputs to the preserved raw splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from training.quality.common import file_hash, read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    profile_hash = file_hash(args.profile)
    if not rows or any('voa' in str(r['source_id']).lower() for r in rows):
        raise ValueError('Expected only non-VOA records')
    output = []
    for row in rows:
        path = args.processed / f"{row['utterance_id']}.wav"
        metadata = json.loads(path.with_suffix('.json').read_text())
        info = sf.info(path)
        if (info.samplerate, info.channels, info.subtype) != (24000, 1, 'PCM_24'):
            raise ValueError(f"Invalid PCM24 output: {row['utterance_id']}")
        if metadata['input_sha256'] != file_hash(row['audio_path']) or metadata['output_sha256'] != file_hash(path):
            raise ValueError('Stale processing output')
        import yaml
        if metadata['profile'] != yaml.safe_load(args.profile.read_text()):
            raise ValueError('Processing profile mismatch')
        output.append({**row, 'audio_path': str(path.resolve()), 'audio_sha256': metadata['output_sha256'],
                       'duration': info.duration, 'channels': 1, 'sample_rate': 24000,
                       'format': 'WAV/PCM_24', 'processing_profile_sha256': profile_hash})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(args.output.with_suffix('.summary.json'), {'records': len(output), 'voa_records': 0,
               'hours': sum(r['duration'] for r in output) / 3600, 'processing_profile_sha256': profile_hash})


if __name__ == '__main__':
    main()
