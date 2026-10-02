"""Create shared, unnormalized processing inputs from audited MFA frame boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path

import soundfile as sf

from training.audio_enhancement.fair_comparison import atomic_write_pcm24
from training.quality.common import file_hash, read_rows, write_json, write_tables


def prepare(mfa_run, output):
    mfa_run, output = Path(mfa_run), Path(output)
    rows = read_rows(mfa_run / 'panel.jsonl')
    cuts = read_rows(mfa_run / 'processing.jsonl')
    by_id = {r['utterance_id']: r for r in cuts}
    if (len(by_id) != len(cuts) or len({r['utterance_id'] for r in rows}) != len(rows)
            or set(by_id) != {r['utterance_id'] for r in rows}):
        raise ValueError('MFA boundary records must cover the panel exactly')
    grids = {p.stem: p for p in (mfa_run / 'aligned').rglob('*.json')}
    result = []
    for row in rows:
        cut = by_id[row['utterance_id']]
        source = Path(row['alignment_audio_path'])
        if (file_hash(source) != row['alignment_audio_sha256']
                or cut['input_sha256'] != row['alignment_audio_sha256']
                or cut['reference_sha256'] != row['reference_sha256']
                or file_hash(row['audio_path']) != row['reference_sha256']
                or cut['source_id'] != row['source_id']):
            raise ValueError('MFA input provenance changed')
        grid = grids.get(row['sample_id'])
        if cut['alignment_sha256'] != (file_hash(grid) if grid else None):
            raise ValueError('MFA alignment changed')
        audio, rate = sf.read(source, dtype='float32', always_2d=True)
        begin, end = cut['start_frame'], cut['end_frame']
        if (rate != 24000 or audio.shape[1] != 1 or len(audio) != cut['input_frames']
                or not 0 <= begin < end <= len(audio) or end - begin != cut['output_frames']):
            raise ValueError('Invalid MFA frame boundaries')
        if cut['status'] not in {'mfa_aligned', 'review_preserved_untrimmed'}:
            raise ValueError('Unknown MFA boundary status')
        if bool(cut['flags']) != (cut['status'] == 'review_preserved_untrimmed'):
            raise ValueError('MFA flags and status disagree')
        if cut['flags'] and (begin != 0 or end != len(audio)):
            raise ValueError('Flagged alignments must preserve the complete input')
        target = output / 'inputs' / (row['utterance_id'] + '.wav')
        atomic_write_pcm24(target, audio[begin:end], rate)
        result.append({**row, 'processing_audio_path': str(target.resolve()),
                       'processing_input_sha256': file_hash(target), 'boundary_method': 'mfa',
                       'mfa_status': cut['status'], 'mfa_flags': cut['flags'],
                       'mfa_start_frame': begin, 'mfa_end_frame': end,
                       'mfa_alignment_sha256': cut['alignment_sha256']})
    write_tables(output, 'panel', result)
    write_json(output / 'input_provenance.json', {
        'boundary_method': 'mfa', 'count': len(result),
        'status_counts': dict(Counter(r['mfa_status'] for r in result)),
        'panel_sha256': file_hash(output / 'panel.jsonl'),
        'alignment_panel_sha256': file_hash(mfa_run / 'panel.jsonl'),
        'boundary_records_sha256': file_hash(mfa_run / 'processing.jsonl'),
        'fallback': 'untrimmed; never energy-based', 'input_loudness_processing': 'none'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mfa-run', type=Path, default=Path('training/quality_runs/mfa_trim'))
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/mfa_processing'))
    args = parser.parse_args()
    print(json.dumps({'prepared': len(prepare(args.mfa_run, args.output))}))


if __name__ == '__main__':
    main()
