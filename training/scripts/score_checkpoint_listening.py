"""Score the exact saved listening WAVs on GPU and rebuild the listening bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from training.quality.common import file_hash, write_json, write_tables


MAPPING = {'MOS_OVRL': 'overall', 'MOS_SIG': 'speech', 'MOS_NOISE': 'noise',
           'MOS_COL': 'coloration', 'MOS_DISC': 'discontinuity',
           'MOS_LOUD': 'loudness', 'MOS_REVERB': 'reverb'}
ROLES = ('synthesis', 'reference', 'processed')


class QualityModels:
    def __init__(self, config):
        import torch
        from audiobox_aesthetics.infer import initialize_predictor
        from training.quality.sigmos_only import SigMOSOnly
        from training.quality.devices import require_gpu_models

        self.torch = torch
        device = config['device']
        repository = Path(__file__).resolve().parents[2]
        self.sigmos = SigMOSOnly(repository / config['sigmos_dir'], device)
        checkpoint = repository / config['audiobox_checkpoint']
        self.audiobox = initialize_predictor(ckpt=str(checkpoint.resolve(strict=True)))
        self.audiobox.device = torch.device(device)
        self.audiobox.model.to(device)
        devices = require_gpu_models((self.audiobox.model,))
        self.identity = {'sigmos': self.sigmos.identity,
                         'audiobox_sha256': file_hash(checkpoint), 'model_devices': devices}

    def score(self, audio, rate):
        mos = self.sigmos.score(audio, rate)
        with self.torch.inference_mode():
            pq = self.audiobox.forward([{
                'path': self.torch.from_numpy(audio.copy()).unsqueeze(0), 'sample_rate': rate}])[0]['PQ']
        result = {f'sigmos_{name}': float(mos[key]) for key, name in MAPPING.items()}
        result['audiobox_pq'] = float(pq)
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError('Nonfinite quality score')
        return result


def score_listening(output, config_path, models=None):
    from training.scripts.generate_checkpoint_listening import write_pages

    output = Path(output)
    rows = [json.loads(line) for line in (output / 'selection.jsonl').read_text().splitlines()]
    report = json.loads((output / 'report.json').read_text())
    if not rows or len(rows) != report['sample_count']:
        raise ValueError('Listening sample count mismatch')
    # Verify all audio before loading models or altering any report.
    for row in rows:
        for role in ROLES:
            if file_hash(output / row[role]) != row['audio_sha256'][role]:
                raise ValueError(f'Listening audio hash mismatch: {row["utterance_id"]} {role}')
    if models is None:
        models = QualityModels(yaml.safe_load(Path(config_path).read_text())['models'])
    metrics = []
    for index, row in enumerate(rows, 1):
        row['quality'] = {}
        for role in ROLES:
            audio, rate = sf.read(output / row[role], dtype='float32', always_2d=True)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError('Invalid listening audio')
            scores = models.score(audio.mean(axis=1), rate)
            row['quality'][role] = scores
            metrics.append({'checkpoint_step': report['checkpoint_step'],
                            'utterance_id': row['utterance_id'], 'source_id': row['source_id'],
                            'role': role, 'audio_sha256': row['audio_sha256'][role], **scores})
        print(f'Scored {index}/{len(rows)} listening items (three waveforms each)', flush=True)
    write_tables(output, 'quality', metrics)
    report['quality'] = {'status': 'complete', 'waveforms_scored': len(metrics),
                         'scope': 'whole_file', 'selection_mode': 'report_only',
                         'audio_policy': 'exact saved WAV; mono channel mean; no normalization',
                         'models': models.identity}
    write_json(output / 'report.json', report)
    (output / 'selection.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
    write_pages(output, rows, report['checkpoint_step'])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Existing listening directory')
    parser.add_argument('--config', type=Path, default=Path('training/conf/quality.yaml'))
    args = parser.parse_args()
    score_listening(args.output, args.config)


if __name__ == '__main__':
    main()
