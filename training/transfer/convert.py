"""Convert weights only; never restore optimizer state or launch training."""
import argparse
import copy
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
import yaml

from .frontend import TransferFrontend
from .mapping import Features, english_ipa, map_tokens

EMBEDDING = 'tts.generator.encoder.embed.0.weight'
MODEL_KEYS = ('tts', 'tts_conf', 'model_conf', 'odim', 'feats_extract', 'feats_extract_conf',
              'normalize', 'normalize_conf', 'pitch_extract', 'pitch_extract_conf',
              'pitch_normalize', 'pitch_normalize_conf', 'energy_extract',
              'energy_extract_conf', 'energy_normalize', 'energy_normalize_conf')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize_rows(weight, mappings, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    # Independent rows, calibrated to the donor's empirical embedding scale.
    result = torch.randn(len(mappings), weight.shape[1], generator=generator,
                         dtype=weight.dtype) * weight.float().std().item()
    for row in mappings:
        if row['source_id'] is not None:
            result[row['target_id']].copy_(weight[row['source_id']])
    return result


def load_strict(bundle, device='cpu', checkpoint=None):
    from espnet2.tasks.gan_tts import GANTTSTask
    config = yaml.safe_load((Path(bundle) / 'config.yaml').read_text())
    # Resolve normalization paths relative to the package, not the current cwd.
    for key in ('normalize_conf', 'pitch_normalize_conf', 'energy_normalize_conf'):
        config[key]['stats_file'] = str(Path(bundle) / config[key]['stats_file'])
    args = argparse.Namespace(**config)
    model = GANTTSTask.build_model(args)
    state = torch.load(checkpoint or Path(bundle) / 'model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device), args


def convert(args):
    from espnet2.tasks.gan_tts import GANTTSTask
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source = yaml.safe_load(args.source_config.read_text())
    target_config = yaml.safe_load(args.target_config.read_text())
    target = target_config['token_list']
    if not isinstance(target, list) or not isinstance(source['token_list'], list):
        raise ValueError('Config token_list must contain the explicit inventory')
    if source['tts'] != 'jets' or source['g2p'] != 'g2p_en_no_space':
        raise ValueError('Expected an English ARPAbet JETS donor')
    if target[:2] != ['<blank>', '<unk>'] or target[-1] != '<sos/eos>':
        raise ValueError('Expected ESPnet blank/unk/EOS inventory layout')
    if not 0 < args.lr_factor < 1:
        raise ValueError('Fine-tuning learning rate must be reduced')
    state = torch.load(args.source_model, map_location='cpu', weights_only=True)
    if EMBEDDING not in state or state[EMBEDDING].shape[0] != len(source['token_list']):
        raise ValueError('Source token count/embedding mismatch; expected bare model weights')
    mappings = map_tokens(source['token_list'], target, Features(args.features), args.max_distance)
    converted = dict(state)
    converted[EMBEDDING] = initialize_rows(state[EMBEDDING], mappings, args.seed)
    assert all(torch.equal(value, converted[key]) for key, value in state.items() if key != EMBEDDING)
    torch.save(converted, output / 'model.pth')
    config = {key: copy.deepcopy(source[key]) for key in MODEL_KEYS}
    for key in ('normalize_conf', 'pitch_normalize_conf', 'energy_normalize_conf'):
        relative = Path('stats') / Path(config[key]['stats_file']).name
        (output / 'stats').mkdir(exist_ok=True)
        shutil.copyfile(args.source_root / config[key]['stats_file'], output / relative)
        config[key]['stats_file'] = str(relative)
    config['tts_conf']['generator_params']['idim'] = len(target)
    compatibility = []
    scale_keys = [key for key in state if key.startswith('tts.discriminator.msd.')]
    if any(key.endswith('.weight') for key in scale_keys) and not any(
            key.endswith(('.weight_g', '.weight_orig')) for key in scale_keys):
        # Older ESPnet checkpoints did not apply the normalization advertised
        # in their YAML. Match the actual weights before optimizers are built.
        discriminator = config['tts_conf']['discriminator_params']
        discriminator['follow_official_norm'] = False
        discriminator['scale_discriminator_params']['use_weight_norm'] = False
        discriminator['scale_discriminator_params']['use_spectral_norm'] = False
        compatibility.append('Scale discriminator normalization disabled to match actual donor weights (ESPnet issue #5240)')
    config.update(token_list=target, token_type='phn', g2p=None, cleaner=None,
                  use_preprocessor=True, resume=False, freeze_param=[], init_param=[],
                  ignore_init_mismatch=False, batch_type='unsorted', batch_size=4,
                  num_workers=2, ngpu=1, seed=args.seed, sharded_ddp=False,
                  num_iters_per_epoch=1000, max_epoch=50, use_amp=False,
                  grad_clip=-1, generator_first=True, use_wandb=True,
                  wandb_project='ukrainian-tts', wandb_model_log_interval=-1,
                  val_scheduler_criterion=['valid', 'generator_loss'],
                  early_stopping_criterion=['valid', 'generator_loss', 'min'],
                  best_model_criterion=[['valid', 'generator_g_mel_loss', 'min']])
    for suffix in ('', '2'):
        for name in ('optim', 'scheduler'):
            for ending in ('', '_conf'):
                key = name + suffix + ending
                config[key] = copy.deepcopy(source[key])
        config['optim' + suffix + '_conf']['lr'] *= args.lr_factor
    # Fill current ESPnet defaults instead of carrying donor distributed/job state.
    temporary = output / 'settings.yaml'
    temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    parsed = GANTTSTask.get_parser().parse_args(['--config', str(temporary), '--output_dir', 'exp'])
    config = vars(parsed)
    config.pop('config', None)
    config.pop('required', None)  # Parser bookkeeping, not an accepted YAML option.
    (output / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    GANTTSTask.get_parser().parse_args(['--config', str(output / 'config.yaml')])
    temporary.unlink()
    (output / 'tokens.txt').write_text('\n'.join(target) + '\n')
    frontend = TransferFrontend(target)
    (output / 'frontend.json').write_text(json.dumps(frontend.snapshot(), ensure_ascii=False, indent=2) + '\n')
    with (output / 'mapping.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mappings[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(mappings)
    report = dict(source_checkpoint_sha256=sha256(args.source_model),
                  source_config_sha256=sha256(args.source_config),
                  features_sha256=sha256(args.features), seed=args.seed,
                  max_feature_distance=args.max_distance, minimum_feature_coverage=.9,
                  distance='Unweighted categorical mismatch over known PHOIBLE articulatory features; same syllabicity required',
                  tie_break='Exact IPA, feature distance, matching stress, source token ID',
                  random_initialization='Independent normal rows, zero mean and donor embedding standard deviation',
                  source_rows=len(source['token_list']), target_rows=len(target),
                  embedding_dimension=state[EMBEDDING].shape[1],
                  unchanged_tensors=len(state) - 1, sample_rate=config['feats_extract_conf']['fs'],
                  counts=dict(Counter(row['method'] for row in mappings)), mappings=mappings)
    report['compatibility_adjustments'] = compatibility
    report['source_inventory'] = [
        dict(source_id=index, token=token, ipa=english_ipa(token))
        for index, token in enumerate(source['token_list'])
    ]
    (output / 'mapping.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    model, _ = load_strict(output)
    assert set(model.state_dict()) == set(converted)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in converted.items())
    assert all(p.requires_grad for p in model.parameters())
    print(json.dumps({k: v for k, v in report.items() if k not in {'mappings', 'source_inventory'}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-config', 'source-model', 'source-root', 'target-config', 'features', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--max-distance', type=float, default=.15)
    parser.add_argument('--lr-factor', type=float, default=.1)
    convert(parser.parse_args())


if __name__ == '__main__':
    main()
