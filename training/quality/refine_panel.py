"""Freeze a smaller processing panel and derive matching measured references."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import digest, file_hash, read_rows, write_json, write_tables
from .evaluate import aggregate


def select(parent, output, per_source=20, seed=778):
    rows = read_rows(parent)
    if not 1 <= per_source <= 100 or not rows:
        raise ValueError('Require a nonempty panel and 1–100 items per source')
    if len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate panel IDs')
    if any('voa' in r['source_id'].lower() or not r['split'].endswith('_train') for r in rows):
        raise ValueError('Refinement must use non-VOA training items only')
    selected = []
    for source in sorted({r['source_id'] for r in rows}):
        group = [r for r in rows if r['source_id'] == source]
        if len(group) < per_source:
            raise ValueError(f'Insufficient items for a balanced subset: {source}')
        selected.extend(sorted(group, key=lambda r: digest([seed, r['utterance_id']]))[:per_source])
    output = Path(output)
    panel = output / 'processing.jsonl'
    selection = {'parent_panel_sha256': file_hash(parent), 'parent_items': len(rows),
                 'items': len(selected), 'per_source': per_source, 'seed': seed,
                 'purpose': 'refinement_screening_confirm_on_full_panel'}
    if panel.exists():
        if read_rows(panel) != selected or json.loads((output / 'selection.json').read_text()) != selection:
            raise ValueError('Refinement panel is frozen; choose another output directory')
    else:
        write_tables(output, 'processing', selected)
        write_json(output / 'selection.json', selection)
    return panel


def derive_report(source, panel, output):
    source, output = Path(source), Path(output)
    metadata = json.loads((source / 'run.json').read_text())
    if metadata['status'] != 'complete':
        raise ValueError('Cannot derive references from an incomplete evaluation')
    records = read_rows(panel)
    measured = read_rows(source / 'per_file.jsonl')
    by_id = {r['utterance_id']: r for r in measured}
    if len(by_id) != len(measured) or len(by_id) != metadata['expected']:
        raise ValueError('Parent evaluation coverage is inconsistent')
    selected = []
    for record in records:
        row = by_id.get(record['utterance_id'])
        if row is None or any(row[k] != record[k] for k in ('source_id', 'text', 'reference_sha256')):
            raise ValueError('Subset does not match measured parent records')
        if (file_hash(record['audio_path']) != record['reference_sha256']
                or file_hash(row['audio_path']) != row['audio_sha256']):
            raise ValueError('Measured audio changed')
        selected.append(row)
    identifiers = {r['utterance_id'] for r in selected}
    if not selected or len(identifiers) != len(selected):
        raise ValueError('Empty or duplicate subset IDs')
    provenance = {k: v for k, v in metadata.items()
                  if k not in {'run_key', 'status', 'expected', 'scored', 'errors'}}
    provenance.update(panel_sha256=file_hash(panel), derived_from={
        'run_key': metadata['run_key'], 'panel_sha256': metadata['panel_sha256'],
        'per_file_sha256': file_hash(source / 'per_file.jsonl'),
        'method': 'select_existing_measurements_without_rescoring'})
    result = {**provenance, 'run_key': digest(provenance), 'status': 'complete',
              'expected': len(selected), 'scored': len(selected), 'errors': 0}
    if (output / 'run.json').exists():
        if json.loads((output / 'run.json').read_text()) != result:
            raise ValueError('Derived reference already exists with different provenance')
    write_tables(output, 'per_file', selected)
    write_tables(output, 'segments', [r for r in read_rows(source / 'segments.jsonl')
                                      if r['utterance_id'] in identifiers])
    write_tables(output, 'aggregate', aggregate(selected))
    write_tables(output, 'errors', [])
    write_json(output / 'run.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=20)
    parser.add_argument('--seed', type=int, default=778)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    panel = select(args.parent, args.output, args.per_source, args.seed)
    for source in args.reports:
        label = json.loads((source / 'run.json').read_text())['label']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', label) or label in {'.', '..'}:
            raise ValueError('Report label is not a safe directory name')
        derive_report(source, panel, args.output / 'references' / label)
    print(f'Frozen refinement panel: {len(read_rows(panel))} files', flush=True)


if __name__ == '__main__':
    main()
