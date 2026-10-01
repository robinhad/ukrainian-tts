"""Prepare a pinned local ECAPA model for ESPnet's speaker embedding extractor."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from training.quality.common import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('training/vendor/speechbrain-spkrec-ecapa-voxceleb'))
    args = parser.parse_args()
    revision = '0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'
    names = ['hyperparams.yaml', 'classifier.ckpt', 'embedding_model.ckpt',
             'label_encoder.txt', 'mean_var_norm_emb.ckpt']
    snapshot = Path(snapshot_download('speechbrain/spkrec-ecapa-voxceleb',
                                      revision=revision, allow_patterns=names))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        destination = args.output / name
        if destination.is_symlink():
            raise ValueError('Refusing to overwrite a symlinked model file')
        shutil.copyfile(snapshot / name, destination)
    hyperparams = args.output / 'hyperparams.yaml'
    original = hyperparams.read_text()
    source_line = 'pretrained_path: speechbrain/spkrec-ecapa-voxceleb'
    if original.count(source_line) != 1:
        raise ValueError('Unexpected pinned ECAPA configuration')
    # Generated local paths stay only in the ignored model directory. ESPnet's
    # extractor must use these pinned files instead of fetching moving main.
    hyperparams.write_text(original.replace(source_line,
                          f'pretrained_path: {json.dumps(str(args.output.resolve()))}'))
    write_json(args.output / 'provenance.json', {
        'repo': 'speechbrain/spkrec-ecapa-voxceleb', 'revision': revision,
        'upstream_hashes': {name: file_hash(snapshot / name) for name in names},
        'local_hashes': {name: file_hash(args.output / name) for name in names},
        'local_hyperparams_change': 'Resolve pretrained_path to pinned local files',
    })
    print('Pinned ECAPA files prepared for training')


if __name__ == '__main__':
    main()
