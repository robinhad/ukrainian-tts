"""Select the higher-scoring normalized original/cascade file, then apply a cutoff."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import soundfile as sf

from training.quality.common import file_hash, read_rows, write_json, write_tables


def choose(normalized, processed, threshold=3.5):
    if not all(math.isfinite(x) for x in (normalized, processed, threshold)):
        raise ValueError('Nonfinite score or threshold')
    if not all(1 <= x <= 5 for x in (normalized, processed, threshold)):
        raise ValueError('SigMOS scores and threshold must be in [1, 5]')
    variant = 'processed' if processed > normalized else 'normalized_original'
    score = max(normalized, processed)
    return variant, score, score >= threshold


def select(normalized_path, processed_path, output, threshold=3.5, copy_audio=False):
    def keyed(path):
        rows = read_rows(path)
        result = {(r['source_id'], r['utterance_id']): r for r in rows}
        if not rows or len(result) != len(rows):
            raise ValueError('Empty or duplicate score records')
        return result
    normalized, processed = keyed(normalized_path), keyed(processed_path)
    if normalized.keys() != processed.keys():
        raise ValueError('Normalized and processed recordings must match exactly')
    output = Path(output)
    decisions, records = [], []
    for key in sorted(normalized):
        baseline, candidate = normalized[key], processed[key]
        if baseline['reference_sha256'] != candidate['reference_sha256']:
            raise ValueError('Reference recording mismatch')
        for row in (baseline, candidate):
            metadata = json.loads(Path(row['audio_path']).with_suffix('.json').read_text())
            if metadata['input_sha256'] != baseline['processing_input_sha256']:
                raise ValueError('Canonical processing input mismatch')
            if metadata['output_sha256'] != row['audio_sha256']:
                raise ValueError('Processing metadata mismatch')
        infos = []
        for row in (baseline, candidate):
            if file_hash(row['audio_path']) != row['audio_sha256']:
                raise ValueError('Scored audio changed')
            info = sf.info(row['audio_path'])
            if (info.samplerate, info.channels, info.subtype) != (24000, 1, 'PCM_24'):
                raise ValueError('Both candidates must already be mono 24 kHz PCM24')
            infos.append(info)
        if infos[0].frames != infos[1].frames:
            raise ValueError('Candidates must have the same trimmed sample count')
        variant, score, retained = choose(baseline['sigmos_overall'], candidate['sigmos_overall'], threshold)
        chosen = candidate if variant == 'processed' else baseline
        sample_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        decision = {'sample_id': sample_id, 'source_id': key[0],
                    'normalized_sigmos_overall': baseline['sigmos_overall'],
                    'processed_sigmos_overall': candidate['sigmos_overall'],
                    'selected_sigmos_overall': score, 'selected_variant': variant,
                    'retained': retained, 'duration_seconds': infos[0].duration,
                    'selected_audio_sha256': chosen['audio_sha256']}
        decisions.append(decision)
        if retained:
            path = Path(chosen['audio_path'])
            if copy_audio:
                target = output / 'audio_24k' / f'{sample_id}.wav'
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if file_hash(target) != chosen['audio_sha256']:
                    raise ValueError('Selected copy hash mismatch')
                path = target
            records.append({**chosen, **decision, 'audio_path': str(path.resolve())})
    write_tables(output, 'selection', decisions)
    write_tables(output, 'retained', records)
    write_json(output / 'policy.json', {
        'baseline': 'normalized_original', 'candidate': 'full_cascade',
        'comparison': 'strictly_higher_else_normalized_original', 'threshold': threshold,
        'threshold_applied_after_rendering': True, 'hnr_rejection': False,
        'evaluated': len(decisions), 'retained': len(records),
        'hours': sum(r['duration_seconds'] for r in records) / 3600,
        'normalized_metrics_sha256': file_hash(normalized_path),
        'processed_metrics_sha256': file_hash(processed_path),
    })
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalized', type=Path, required=True)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=3.5)
    parser.add_argument('--copy-audio', action='store_true')
    args = parser.parse_args()
    select(args.normalized, args.processed, args.output, args.threshold, args.copy_audio)


if __name__ == '__main__':
    main()
