"""Export portable checkpoint metrics, excluding audio paths and transcripts."""
import argparse
import csv
import json
from pathlib import Path


METRICS = set('''sigmos_overall sigmos_speech sigmos_noise sigmos_coloration
sigmos_discontinuity sigmos_loudness sigmos_reverb audiobox_pq cer wer
character_errors reference_characters word_errors reference_words
parakeet_cer parakeet_wer parakeet_character_errors parakeet_reference_characters
parakeet_word_errors parakeet_reference_words ecapa_similarity duration_seconds
duration_ratio peak_absolute clipping_fraction clipping_samples hf_burst_count
hf_max_excess_db hf_max_ratio hf_available sample_rate channels'''.split())
IDENTIFIERS = {'utterance_id', 'source_id', 'label', 'audio_sha256', 'reference_sha256'}
STRUCTURES = {'flags', 'hf_bursts', 'worst_segments'}
FIELDS = {
    'per_file': METRICS | IDENTIFIERS | STRUCTURES,
    'segments': METRICS | IDENTIFIERS | STRUCTURES | {
        'segment', 'start_seconds', 'end_seconds', 'short_segment',
        'content_score_status', 'speaker_reference'},
    'aggregate': set('count expected_count mean median metric p05 p95 source_id'.split()),
    'comparison_per_file': set('''baseline baseline_role baseline_score candidate
        candidate_score delta metric source_id utterance_id'''.split()),
    'comparison_aggregate': set('''baseline_role count direction mean_delta
        median_delta metric p05_delta p95_delta source_id'''.split()),
}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def portable_row(row, fields, label):
    result = {'checkpoint': label, **{k: v for k, v in row.items() if k in fields}}
    # Even allowed string fields must not accidentally contain local paths.
    def check(value):
        if isinstance(value, str) and ('/' in value or '\\' in value):
            raise ValueError('Unexpected path-like value in portable metric')
        if isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(result)
    return result


def write_table(prefix, records):
    prefix.with_suffix('.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in records))
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        columns = ['checkpoint'] + sorted(set().union(*(r.keys() for r in records)) - {'checkpoint'})
        writer = csv.DictWriter(stream, columns, lineterminator='\n')
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def export(root, output, labels):
    tables = {name: [] for name in FIELDS}
    runs = []
    reference_ids = reference_hashes = None
    for label in labels:
        if label != 'original' and not (label.endswith('k') and label[:-1].isdigit()):
            raise ValueError('Invalid checkpoint label')
        directory = root / ('original' if label == 'original' else label + '_quality')
        run = json.loads((directory / 'run.json').read_text())
        if run['status'] != 'complete' or run['errors'] or run['scored'] != run['expected']:
            raise ValueError(f'{label}: evaluation is incomplete or has errors')
        per_file = rows(directory / 'per_file.jsonl')
        ids = {row['utterance_id'] for row in per_file}
        hashes = {row['utterance_id']: row['reference_sha256'] for row in per_file}
        if len(ids) != len(per_file) or len(ids) != run['expected']:
            raise ValueError('Duplicate or missing panel items')
        if reference_ids is not None and (ids != reference_ids or hashes != reference_hashes):
            raise ValueError('Checkpoint panels do not match')
        reference_ids, reference_hashes = ids, hashes
        for name, fields in FIELDS.items():
            path = directory / (name + '.jsonl')
            if name.startswith('comparison_') and label == 'original':
                continue
            tables[name].extend(portable_row(row, fields, label) for row in rows(path))
        runs.append({'checkpoint': label, **{key: run.get(key) for key in (
            'panel_sha256', 'checkpoint_sha256', 'training_config_sha256',
            'selection_mode', 'hnr_rejection', 'expected', 'scored', 'errors')},
            'thresholds': run['config']['thresholds']})
    output.mkdir(parents=True, exist_ok=True)
    for name, records in tables.items():
        write_table(output / name, records)
    write_table(output / 'runs', runs)
    return {'checkpoints': labels, 'panel_items': len(reference_ids),
            'exported_rows': {name: len(records) for name, records in tables.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output, args.labels)))


if __name__ == '__main__':
    main()
