"""Download pinned public donor and PHOIBLE inputs, checking every SHA-256."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from .convert import sha256


def download(output, lock_path=None):
    lock = json.loads((lock_path or Path(__file__).with_name('inputs.lock.json')).read_text())
    for entry in lock['files']:
        destination = output / entry['path']
        if destination.exists() and sha256(destination) == entry['sha256']:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.partial')
        with urlopen(entry['url'], timeout=120) as response, temporary.open('wb') as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if sha256(temporary) != entry['sha256']:
            temporary.unlink()
            raise ValueError(f"Download checksum mismatch: {entry['path']}")
        temporary.replace(destination)
        print(entry['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    download(args.output, args.lock)


if __name__ == '__main__':
    main()
