"""Explicit full-model fine-tuning entrypoint; weights-only initialization."""
import argparse
import json
from pathlib import Path

import yaml

from .convert import sha256
from .frontend import TransferFrontend


def training_config(bundle, train, valid, output):
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    frontend = TransferFrontend(config['token_list'])
    frontend.verify(bundle / 'frontend.json')
    for split, directory in [('train', train), ('valid', valid)]:
        metadata = json.loads((directory / 'metadata.json').read_text())
        if (metadata['frontend_sha256'] != sha256(bundle / 'frontend.json') or
                metadata['sample_rate'] != config['feats_extract_conf']['fs']):
            raise ValueError('Prepared data frontend/audio settings mismatch')
        for filename, key in [('text', 'text_sha256'), ('wav.scp', 'wav_scp_sha256')]:
            if metadata[key] != sha256(directory / filename):
                raise ValueError('Prepared data modified since tokenization')
        config[split + '_data_path_and_name_and_type'] = [
            [str((directory / 'text').resolve()), 'text', 'text'],
            [str((directory / 'wav.scp').resolve()), 'speech', 'sound']]
        # ESPnet's unsorted sampler still requires a key file. It reads only
        # the first column, so the checked text manifest is sufficient.
        config[split + '_shape_file'] = [str((directory / 'text').resolve())]
    for key in ('normalize_conf', 'pitch_normalize_conf', 'energy_normalize_conf'):
        config[key]['stats_file'] = str((bundle / config[key]['stats_file']).resolve())
    config.update(output_dir=str(output.resolve()), resume=False, freeze_param=[],
                  init_param=[str((bundle / 'model.pth').resolve())], ignore_init_mismatch=False)
    return config


def main():
    from espnet2.tasks.gan_tts import GANTTSTask
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'train', 'valid', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--write-config-only', action='store_true')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--external-metrics', action='store_true',
                        help='Use the repository numeric-only TensorBoard-to-W&B supervisor')
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 1000:
        parser.error('--steps must be a positive multiple of 1000')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    # Never silently resume or overwrite an earlier run/optimizer.
    if args.output.exists():
        raise ValueError('Use a new output directory for fresh optimizer fine-tuning')
    config = training_config(args.bundle, args.train, args.valid, args.output)
    config.update(max_epoch=args.steps // 1000, num_iters_per_epoch=1000,
                  batch_size=args.batch_size, keep_nbest_models=[3],
                  num_att_plot=0, log_interval=10, use_tensorboard=True)
    if args.external_metrics:
        config['use_wandb'] = False
    args.output.mkdir(parents=True)
    config_path = args.output / 'finetune.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    GANTTSTask.get_parser().parse_args(['--config', str(config_path)])
    if not args.write_config_only:
        GANTTSTask.main(cmd=['--config', str(config_path)])


if __name__ == '__main__':
    main()
