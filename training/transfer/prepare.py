"""Prepare JSONL pairs or existing Ukrainian parquet manifests; no training."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

from .frontend import TransferFrontend


def read_pairs(path):
    if path.suffix == '.parquet':
        import pandas as pd
        frame = pd.read_parquet(path, columns=['utterance_id', 'text_sanitized', 'audio_path'])
        for row in frame.itertuples(index=False):
            yield dict(id=row.utterance_id, text=row.text_sanitized, audio_path=row.audio_path)
    else:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def read_audio(path, sample_rate):
    audio, original_rate = sf.read(path, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original_rate // divisor)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Empty or nonfinite input audio')
    return audio


def prepare(bundle, pairs, output):
    from .convert import sha256
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    frontend = TransferFrontend(config['token_list'])
    frontend.verify(bundle / 'frontend.json')
    sample_rate = config['feats_extract_conf']['fs']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audio').mkdir()
    seen = set()
    with (output / 'text').open('w') as text_file, (output / 'wav.scp').open('w') as wav_file:
        for row in read_pairs(pairs):
            uid = str(row['id'])
            if not uid or any(c.isspace() for c in uid) or uid in seen:
                raise ValueError('Pair IDs must be unique nonempty strings without whitespace')
            seen.add(uid)
            phones, _ = frontend.encode(row['text'])
            audio = read_audio(row['audio_path'], sample_rate)
            destination = output / 'audio' / (hashlib.sha256(uid.encode()).hexdigest() + '.wav')
            sf.write(destination, audio, sample_rate, subtype='FLOAT')
            text_file.write(uid + ' ' + ' '.join(phones) + '\n')
            wav_file.write(uid + ' ' + str(destination.resolve()) + '\n')
    if not seen:
        raise ValueError('Empty dataset')
    metadata = dict(frontend_sha256=sha256(bundle / 'frontend.json'), sample_rate=sample_rate,
                    items=len(seen), text_sha256=sha256(output / 'text'), wav_scp_sha256=sha256(output / 'wav.scp'))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.bundle, args.pairs, args.output)


if __name__ == '__main__':
    main()
