#!/usr/bin/env python3
"""Download only the eight non-VOA sources, at the existing recorded revisions."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('training/conf/expanded_v3_sources.yaml'))
    parser.add_argument('--output', type=Path, default=Path('training/data/non_voa_downloads'))
    args = parser.parse_args()
    sources = yaml.safe_load(args.registry.read_text())['sources']
    names = ['opentts_lada', 'opentts_tetiana', 'opentts_mykyta', 'tg_voices_uk',
             'ukr_dialects', 'fleurs_uk', 'ua_ser', 'common_voice_available_uk']
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'downloads.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    api = HfApi()
    for name in names:
        source = sources[name]
        repo = source.get('repo_id', 'speech-uk/cv22-opus')
        revision = source['revision']
        # Restore the same CV22 copy previously used by this branch, not another version.
        entries = list(api.list_repo_tree(repo, repo_type='dataset', revision=revision, recursive=True))
        paths = [e.path for e in entries if e.path.endswith('.parquet')
                 and (name != 'fleurs_uk' or '/uk_ua/' in '/' + e.path)]
        if name == 'ua_ser':
            paths = [e.path for e in entries if e.path == 'dataset.csv' or e.path.startswith('clips/')]
        if not paths:
            raise RuntimeError(f'No pinned source files found: {name}')
        size = sum(getattr(e, 'size', 0) or 0 for e in entries if e.path in paths)
        if shutil.disk_usage(args.output).free < size + 30 * 1024**3:
            raise RuntimeError(f'Insufficient disk reserve for {name}')
        print(json.dumps({'source': name, 'files': len(paths), 'bytes': size, 'state': 'downloading'}), flush=True)
        snapshot = snapshot_download(repo, repo_type='dataset', revision=revision,
                                     allow_patterns=paths + ['README.md'],
                                     cache_dir=str(args.output / 'cache'), max_workers=4)
        report[name] = {'repo_id': repo, 'revision': revision, 'license': source['license'],
                        'snapshot': snapshot, 'files': paths, 'bytes': size, 'status': 'downloaded'}
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(report_path)
        print(json.dumps({'source': name, 'state': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
