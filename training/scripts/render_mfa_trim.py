"""Render MFA boundary trims, preserving untrimmed audio when alignment needs review."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import soundfile as sf

from training.audio_enhancement.fair_comparison import atomic_write_pcm24
from training.quality.common import file_hash, read_rows, write_json, write_tables
from training.quality.mfa_trim import boundaries
from training.quality.processing import match_loudness_for_reporting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('training/quality_runs/mfa_trim'))
    parser.add_argument('--padding-ms', type=float, default=100)
    parser.add_argument('--max-removed-fraction', type=float, default=.5)
    args = parser.parse_args()
    panel = read_rows(args.run / 'panel.jsonl')
    alignment_paths = {p.stem: p for p in (args.run / 'aligned').rglob('*.json')}
    output = args.run / 'wav'
    output.mkdir(parents=True, exist_ok=True)
    def render(row):
        source = Path(row['alignment_audio_path'])
        if file_hash(source) != row['alignment_audio_sha256']:
            raise ValueError('Untrimmed alignment input changed')
        audio, rate = sf.read(source, dtype='float32', always_2d=True)
        if rate != 24000 or audio.shape[1] != 1:
            raise ValueError('Expected mono 24 kHz alignment input')
        duration = len(audio) / rate
        grid = alignment_paths.get(row['sample_id'])
        alignment = json.loads(grid.read_text()) if grid else {}
        start, end, flags = boundaries(alignment, duration, args.padding_ms / 1000, args.max_removed_fraction)
        if row['unsupported_tokens']:
            flags.append('unsupported_transcript_tokens')
        if grid and abs(alignment['end'] - duration) > .03:
            flags.append('alignment_duration_mismatch')
        requested_start, requested_end = start, end
        if flags:
            start, end = 0., duration
        begin_frame = max(0, math.floor(start * rate))
        end_frame = min(len(audio), math.ceil(end * rate))
        trimmed = audio[begin_frame:end_frame]
        finalized, loudness = match_loudness_for_reporting(trimmed, trimmed, rate)
        target = output / (row['utterance_id'] + '.wav')
        atomic_write_pcm24(target, finalized, rate)
        record = {'sample_id': row['sample_id'], 'utterance_id': row['utterance_id'], 'source_id': row['source_id'],
                  'reference_sha256': row['reference_sha256'], 'input_sha256': row['alignment_audio_sha256'],
                  'output_sha256': file_hash(target), 'output_path': str(target.resolve()),
                  'alignment_sha256': file_hash(grid) if grid else None,
                  'padding_ms': args.padding_ms, 'max_removed_fraction': args.max_removed_fraction,
                  'requested_start_seconds': requested_start, 'requested_end_seconds': requested_end,
                  'start_frame': begin_frame, 'end_frame': end_frame, 'input_frames': len(audio),
                  'output_frames': len(finalized), 'duration_seconds': len(finalized) / rate,
                  'removed_seconds': (len(audio) - len(finalized)) / rate,
                  'status': 'review_preserved_untrimmed' if flags else 'mfa_aligned',
                  'flags': sorted(set(flags)), 'loudness_match': loudness}
        write_json(target.with_suffix('.json'), record)
        return record
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = []
        for record in pool.map(render, panel):
            results.append(record)
            if len(results) % 25 == 0:
                write_json(args.run / 'render_progress.json', {'completed': len(results), 'expected': len(panel)})
    write_tables(args.run, 'processing', results)
    summary = {'count': len(results), 'status_counts': dict(Counter(r['status'] for r in results)),
               'flags': dict(Counter(f for r in results for f in r['flags'])),
               'removed_seconds': sum(r['removed_seconds'] for r in results),
               'padding_ms': args.padding_ms, 'max_removed_fraction': args.max_removed_fraction,
               'fallback': 'preserve_untrimmed_mono_24k_then_peak_safe_finalization'}
    write_json(args.run / 'render_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
