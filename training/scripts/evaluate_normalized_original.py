"""Render and score the normalized-original category on the fixed processing panel."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import soundfile as sf

from training.quality.common import file_hash, read_rows, write_json, write_tables
from training.quality.processing import process
from training.quality.sigmos_only import SigMOSOnly

METRICS = {'MOS_OVRL': 'overall', 'MOS_SIG': 'speech', 'MOS_NOISE': 'noise',
           'MOS_COL': 'coloration', 'MOS_DISC': 'discontinuity',
           'MOS_LOUD': 'loudness', 'MOS_REVERB': 'reverb'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('training/quality_runs/v10/panels/processing.jsonl'))
    parser.add_argument('--profile', type=Path, default=Path('training/conf/quality_normalized_original.yaml'))
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/normalized_original'))
    parser.add_argument('--sigmos-dir', type=Path, default=Path('training/vendor/quality-models/SIG-Challenge/ICASSP2024/sigmos'))
    args = parser.parse_args()
    panel = read_rows(args.panel)
    if len(panel) != 800 or sorted(Counter(r['source_id'] for r in panel).values()) != [100] * 8:
        raise ValueError('Expected the fixed 800-recording, eight-source panel')
    outputs = process(args.panel, args.profile, args.output / 'wav', device='cpu', cpu_workers=8)
    paths = {r['utterance_id']: r for r in outputs}
    model = SigMOSOnly(args.sigmos_dir)
    rows = []
    started = time.monotonic()
    for i, row in enumerate(panel, 1):
        path = paths[row['utterance_id']]['output_path']
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        scores = model.score(audio.mean(axis=1), rate)
        record = {**row, 'audio_path': path, 'audio_sha256': file_hash(path),
                  'label': 'normalized_original', 'sample_rate': rate, 'channels': audio.shape[1],
                  'duration_seconds': len(audio) / rate,
                  **{'sigmos_' + label: float(scores[key]) for key, label in METRICS.items()}}
        rows.append(record)
        if i % 25 == 0 or i == len(panel):
            now = datetime.now(ZoneInfo('Europe/Kyiv'))
            eta = now + timedelta(seconds=(len(panel) - i) * (time.monotonic() - started) / i)
            progress = {'phase': 'sigmos', 'completed': i, 'expected': len(panel),
                        'checked_kyiv': now.isoformat(timespec='seconds'), 'eta_kyiv': eta.isoformat(timespec='minutes')}
            write_json(args.output / 'progress.json', progress)
            print(progress, flush=True)
    write_tables(args.output, 'per_file', rows)
    write_json(args.output / 'provenance.json', {
        'profile_sha256': file_hash(args.profile), 'panel_sha256': file_hash(args.panel),
        'processing_code_sha256': file_hash('training/quality/processing.py'),
        'sigmos': model.identity, 'count': len(rows),
    })


if __name__ == '__main__':
    main()
