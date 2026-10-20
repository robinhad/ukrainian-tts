"""Select one source-qualified speaker from already prepared audio."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .convert import sha256


def select_voice(manifests, prepared, output, speaker, source):
    frames = {}
    for split in ('train', 'dev'):
        frame = pd.read_parquet(manifests / f'quality_v12_{split}.parquet')
        frame = frame[(frame.speaker_id == speaker) & (frame.source_id == source)]
        if frame.empty or not frame.utterance_id.is_unique:
            raise ValueError(f'{split}: expected a nonempty, unique single-speaker selection')
        frames[split] = frame
    for field in ('utterance_id', 'audio_sha256'):
        if set(frames['train'][field].dropna()) & set(frames['dev'][field].dropna()):
            raise ValueError(f'Train/dev overlap in {field}')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(speaker_id=speaker, source_id=source, splits={})
    frontend_hash = None
    for split, frame in frames.items():
        original = prepared / split
        metadata = json.loads((original / 'metadata.json').read_text())
        if frontend_hash is not None and metadata['frontend_sha256'] != frontend_hash:
            raise ValueError('Train/dev frontend mismatch')
        frontend_hash = metadata['frontend_sha256']
        destination = output / split
        destination.mkdir()
        for filename, hash_key in [('text', 'text_sha256'), ('wav.scp', 'wav_scp_sha256')]:
            if sha256(original / filename) != metadata[hash_key]:
                raise ValueError('Prepared inputs changed since preparation')
            entries = dict(line.split(' ', 1) for line in (original / filename).read_text().splitlines())
            ids = frame.utterance_id.astype(str).tolist()
            if set(ids) - entries.keys():
                raise ValueError(f'Selected IDs missing from prepared {filename}')
            (destination / filename).write_text(''.join(f'{uid} {entries[uid]}\n' for uid in ids))
        # Reuse the exact resampled audio through wav.scp; do not process it again.
        metadata.update(items=len(frame), speaker_id=speaker, source_id=source,
                        text_sha256=sha256(destination / 'text'),
                        wav_scp_sha256=sha256(destination / 'wav.scp'))
        (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        frame.to_parquet(destination / 'selection.parquet', index=False)
        report['splits'][split] = dict(items=len(frame), hours=float(frame.duration.sum() / 3600))
    (output / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifests', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--speaker', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    select_voice(args.manifests, args.prepared, args.output, args.speaker, args.source)


if __name__ == '__main__':
    main()
