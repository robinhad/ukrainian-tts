"""Full-model fine-tuning with explicit fresh initialization or stateful resume."""
import argparse
import json
from pathlib import Path

import yaml

from .convert import sha256
from .frontend import TransferFrontend


def training_config(bundle, train, valid, output, speaker=None):
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    frontend = TransferFrontend(config['token_list'])
    frontend.verify(bundle / 'frontend.json')
    selected_source = None
    for split, directory in [('train', train), ('valid', valid)]:
        metadata = json.loads((directory / 'metadata.json').read_text())
        if speaker is not None:
            if metadata.get('speaker_id') != speaker or not metadata.get('source_id'):
                raise ValueError('Prepared split does not match the requested single voice')
            if selected_source is not None and metadata['source_id'] != selected_source:
                raise ValueError('Train/validation voice sources differ')
            selected_source = metadata['source_id']
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


def validate_resume(output, config):
    """Fail closed instead of ESPnet's implicit fresh start on missing state."""
    import torch
    checkpoint = output / 'checkpoint.pth'
    previous = yaml.safe_load((output / 'finetune.yaml').read_text())
    allowed = {'resume', 'init_param', 'max_epoch', 'transfer_attention_backend'}
    changed = [key for key in set(previous) | set(config)
               if key not in allowed and previous.get(key) != config.get(key)]
    if changed:
        raise ValueError(f'Resume would change training/data settings: {changed}')
    if not checkpoint.is_file():
        raise ValueError('Resume requires a complete checkpoint.pth with optimizer state')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not all(key in state for key in ('model', 'reporter', 'optimizers', 'schedulers')):
        raise ValueError('Resume requires a full trainer checkpoint')
    epoch = state['reporter']['epoch']
    if not 0 < epoch < config['max_epoch']:
        raise ValueError('Resume target must exceed the saved epoch')
    config.update(resume=True, init_param=[])
    return epoch


def main():
    from espnet2.tasks.gan_tts import GANTTSTask
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'train', 'valid', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--write-config-only', action='store_true')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--speaker', help='Require matching single-voice train and validation metadata')
    parser.add_argument('--flash-attention', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume all trainer state; --steps remains the total target')
    parser.add_argument('--external-metrics', action='store_true',
                        help='Use the repository numeric-only TensorBoard-to-W&B supervisor')
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 1000:
        parser.error('--steps must be a positive multiple of 1000')
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    # Never silently resume or overwrite an earlier run/optimizer.
    if args.output.exists() and not args.resume:
        raise ValueError('Use a new output directory for fresh optimizer fine-tuning')
    config = training_config(args.bundle, args.train, args.valid, args.output, args.speaker)
    config.update(max_epoch=args.steps // 1000, num_iters_per_epoch=1000,
                  batch_size=args.batch_size, keep_nbest_models=[3],
                  num_att_plot=0, log_interval=10, use_tensorboard=True)
    if args.external_metrics:
        config['use_wandb'] = False
    task = GANTTSTask
    if args.flash_attention:
        from .attention import flash_task
        task = flash_task()
        config['transfer_attention_backend'] = 'flash2'
    if args.resume:
        validate_resume(args.output, config)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    config_path = args.output / ('finetune-resume.yaml' if args.resume else 'finetune.yaml')
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    task.get_parser().parse_args(['--config', str(config_path)])
    if not args.write_config_only:
        task.main(cmd=['--config', str(config_path)])


if __name__ == '__main__':
    main()
