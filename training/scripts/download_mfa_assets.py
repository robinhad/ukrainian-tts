"""Download commercially usable, attributed MFA assets with pinned hashes."""
import argparse
import json
from pathlib import Path
import urllib.request

from training.quality.common import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('training/conf/quality_mfa_assets.json'))
    parser.add_argument('--output', type=Path, default=Path('training/quality_runs/mfa_trim/models'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, asset in json.loads(args.config.read_text()).items():
        if asset['license'] != 'CC-BY-4.0':
            raise ValueError('Unexpected asset license')
        extension = '.dict' if name == 'dictionary' else '.zip'
        target = args.output / f'ukrainian_mfa_{name}{extension}'
        if target.exists() and file_hash(target) == asset['sha256']:
            continue
        temporary = target.with_suffix(target.suffix + '.download')
        urllib.request.urlretrieve(asset['url'], temporary)
        if file_hash(temporary) != asset['sha256']:
            raise ValueError(f'Unexpected download hash: {name}')
        temporary.replace(target)
        print(f'Downloaded and verified {name}', flush=True)


if __name__ == '__main__':
    main()
