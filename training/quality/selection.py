"""Freeze disjoint, source-balanced processing and held-out evaluation panels."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .common import digest, file_hash, read_rows, write_json, write_tables
from .metrics import normalize_text


def partition(row):
    return str(row['split']).rsplit('_', 1)[-1]


def balanced(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_id']].append(row)
    if not groups:
        raise ValueError('No eligible sources remain')
    size = min(count, min(map(len, groups.values())))
    return [row for source in sorted(groups)
            for row in sorted(groups[source], key=lambda x: digest([seed, x['utterance_id']]))[:size]]


def select(manifest: Path, output: Path, per_source=100, seed=777):
    if not 1 <= per_source <= 100:
        raise ValueError('per_source must be between 1 and 100')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite a frozen panel; use a new directory')
    rows = read_rows(manifest)
    rows = [dict(r) for r in rows if 'voa' not in str(r['source_id']).lower()]
    if not rows or len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Empty corpus or duplicate utterance IDs')
    for row in rows:
        if not re.fullmatch(r'[\w.-]+', str(row['utterance_id'])):
            raise ValueError('Unsafe utterance ID')
        if partition(row) not in {'train', 'dev', 'eval', 'test'}:
            raise ValueError(f"Unknown split: {row['split']}")
        path = Path(row['audio_path']).resolve(strict=True)
        row['audio_path'] = str(path)
        row['text'] = str(row.get('text_raw') or row.get('text_sanitized') or '')
        if not normalize_text(row['text']):
            raise ValueError(f"Missing transcript: {row['utterance_id']}")
        # Hash real files, not potentially stale manifest hashes after relocation.
        row['reference_sha256'] = file_hash(path)
    train = [r for r in rows if partition(r) == 'train']
    development = [r for r in rows if partition(r) == 'dev']
    texts = {normalize_text(r['text']) for r in train + development}
    hashes = {r['reference_sha256'] for r in train + development}
    heldout = [r for r in rows if partition(r) in {'eval', 'test'}
               and normalize_text(r['text']) not in texts and r['reference_sha256'] not in hashes]
    panels = {'processing': balanced(train, per_source, seed),
              'heldout': balanced(heldout, per_source, seed)}
    # Write only JSON-native, portable fields; preserve the original source recordings.
    for name, panel in panels.items():
        clean = [{k: r.get(k) for k in ('utterance_id', 'source_id', 'speaker_id', 'text',
                                      'split', 'audio_path', 'reference_sha256')} for r in panel]
        write_tables(output, name, clean)
    report = {'schema_version': 1, 'seed': seed, 'per_source_cap': per_source,
              'manifest_sha256': file_hash(manifest), 'non_voa_records': len(rows),
              'heldout_overlap_excluded': sum(partition(r) in {'eval', 'test'} for r in rows) - len(heldout),
              'panels': {name: {'records': len(panel), 'sources': sorted({r['source_id'] for r in panel}),
                                'sha256': file_hash(output / f'{name}.jsonl')} for name, panel in panels.items()}}
    write_json(output / 'selection.json', report)
    return report
