#!/usr/bin/env python3
"""Accept a local eSpeak build only after all committed phoneme cases match."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.frontend.phonemize import (FrontendConfig, UkrainianPhonemizer,
                                        configure_pinned_espeak, detect_espeak_version, hash_tree)
from training.frontend.snapshot import load_cases
from training.quality.common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/v10/frontend'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = configure_pinned_espeak()
    config = FrontendConfig(espeak_version=detect_espeak_version(), espeak_data_hash=hash_tree(data))
    expected_version = (root / 'vendor/ESPEAK_NG_VERSION').read_text().strip()
    if config.espeak_version != expected_version:
        raise RuntimeError('The local eSpeak version differs from the pinned version')
    frontend = UkrainianPhonemizer(config=config)
    total = 0
    for suffix in ('', '_full'):
        expected = json.loads((root / f'tests/frontend/expected_phonemes{suffix}.json').read_text())
        for row in load_cases(root / f'tests/frontend/regression{suffix}.tsv'):
            sanitized, tokens = frontend.phonemize(row['text'])
            case = expected['cases'][row['id']]
            if sanitized != case['sanitized'] or tokens != case['tokens']:
                raise RuntimeError(f"Frontend regression differs: {row['id']}")
            total += 1
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'ESPEAK_NG_DATA_HASH').write_text(config.espeak_data_hash + '\n')
    write_json(args.output / 'verification.json', {'status': 'PASS', 'cases': total,
                                                  'frontend_config': asdict(config)})
    print(json.dumps({'status': 'PASS', 'cases': total}))


if __name__ == '__main__':
    main()
