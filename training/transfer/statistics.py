"""Train-only feature statistics and an explicit normalization-state migration."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

FEATURES = {'feats': 'normalize', 'pitch': 'pitch_normalize', 'energy': 'energy_normalize'}


def enable_phase_history():
    """Support archived validation phases in ESPnet's checkpoint ranking."""
    from espnet2.train.reporter import Reporter
    if getattr(Reporter, '_transfer_phase_history', False):
        return
    original = Reporter.sort_epochs_and_values

    def sort_present(self, key, key2, mode):
        view = copy.copy(self)
        view.stats = {epoch: stats for epoch, stats in self.stats.items()
                      if key in stats and key2 in stats[key]}
        if view.stats:
            view.epoch = max(view.stats)
        else:
            raise KeyError(f'No recorded values for {key}.{key2}')
        return original(view, key, key2, mode)

    Reporter.sort_epochs_and_values = sort_present
    Reporter._transfer_phase_history = True


def moments(path):
    # Match GlobalMVN arithmetic, including the source arrays' dtype.
    with np.load(path) as stats:
        count = stats['count']
        mean = stats['sum'] / count
        std = np.sqrt(np.maximum(stats['sum_square'] / count - mean * mean, 1e-20))
    if count <= 0 or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Invalid normalization statistics')
    return int(count), mean, std


def buffers(config):
    result = {}
    for name in FEATURES.values():
        if name + '_conf' not in config:
            continue
        _, mean, std = moments(config[name + '_conf']['stats_file'])
        result[name + '.mean'] = torch.from_numpy(np.asarray(mean).copy())
        result[name + '.std'] = torch.from_numpy(np.asarray(std).copy())
    return result


def validate_buffers(state, config):
    for key, expected in buffers(config).items():
        actual = state[key]
        # Published donors can average identical FP32 buffers across checkpoints,
        # introducing a few rounding bits. Allow only floating-point roundoff.
        if actual.shape != expected.shape or not torch.allclose(
                actual.cpu(), expected.to(actual.dtype),
                rtol=4 * torch.finfo(actual.dtype).eps, atol=0):
            raise ValueError(f'Checkpoint normalization differs from configured statistics: {key}')


def replace_buffers(state, config):
    replacements = buffers(config)
    if len(replacements) != 6:
        raise ValueError('Expected mel, pitch and energy normalization')
    for key, value in replacements.items():
        if key not in state or state[key].shape != value.shape:
            raise ValueError(f'Unexpected normalization buffer: {key}')
        state[key] = value.to(state[key])
    validate_buffers(state, config)
    return sorted(replacements)


def resolve_config(bundle):
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    for name in FEATURES.values():
        config[name + '_conf']['stats_file'] = str(
            (bundle / config[name + '_conf']['stats_file']).resolve())
    return config


def accumulate(totals, features):
    """Frame-weighted float64 moments; no padded frames or utterance averaging."""
    for key in FEATURES:
        for value, length in zip(features[key], features[key + '_lengths']):
            array = value[:int(length)].detach().cpu().numpy().astype(np.float64)
            if not len(array) or not np.isfinite(array).all():
                raise ValueError('Empty or nonfinite extracted features')
            if key not in totals:
                totals[key] = dict(count=0, sum=np.zeros(array.shape[1:], dtype=np.float64),
                                   sum_square=np.zeros(array.shape[1:], dtype=np.float64))
            totals[key]['count'] += len(array)
            totals[key]['sum'] += array.sum(axis=0)
            totals[key]['sum_square'] += np.square(array).sum(axis=0)


def collect(bundle, voice_data, output, device):
    from .convert import load_strict, sha256
    from .train import training_config
    from .frontend import TransferFrontend
    voice = json.loads((voice_data / 'selection.json').read_text())
    train = voice_data / 'train'
    config = training_config(bundle, train, voice_data / 'dev', output, voice['speaker_id'])
    metadata = json.loads((train / 'metadata.json').read_text())
    frame = pd.read_parquet(train / 'selection.parquet')
    paths = dict(line.split(maxsplit=1) for line in (train / 'wav.scp').read_text().splitlines())
    texts = dict(line.split(maxsplit=1) for line in (train / 'text').read_text().splitlines())
    if (set(paths) != set(frame.utterance_id) or set(paths) != set(texts)
            or len(paths) != metadata['items'] or not frame.split.str.endswith('_train').all()
            or not frame.speaker_id.eq(voice['speaker_id']).all()
            or not frame.source_id.eq(voice['source_id']).all()):
        raise ValueError('Statistics must use exactly the prepared single-voice training split')
    if output.exists():
        raise ValueError('Choose a new bundle; original normalization must remain available')
    torch.set_num_threads(2)
    model, args = load_strict(bundle, device)
    model.eval()
    frontend = TransferFrontend(args.token_list)
    token_ids = {token: index for index, token in enumerate(frontend.tokens)}
    totals, audio_records = {}, []
    seconds = 0.
    for index, uid in enumerate(sorted(paths), 1):
        audio, rate = sf.read(paths[uid], dtype='float32')
        if rate != args.feats_extract_conf['fs'] or audio.ndim != 1:
            raise ValueError('Prepared training audio rate/channel mismatch')
        speech = torch.tensor(audio, device=device)[None]
        ids = torch.tensor([token_ids[t] for t in texts[uid].split()], device=device)[None]
        with torch.inference_mode():
            features = model.collect_feats(
                text=ids, text_lengths=torch.tensor([ids.shape[1]], device=device),
                speech=speech, speech_lengths=torch.tensor([speech.shape[1]], device=device))
        accumulate(totals, features)
        audio_records.append(dict(utterance_id=uid, audio_sha256=sha256(Path(paths[uid]))))
        seconds += len(audio) / rate
        if index % 50 == 0 or index == len(paths):
            print(f'Collected {index}/{len(paths)} training recordings', flush=True)
    del model
    shutil.copytree(bundle, output)
    report = dict(scope='training only; development and evaluation excluded',
                  speaker_id=voice['speaker_id'], source_id=voice['source_id'],
                  items=len(paths), hours=seconds / 3600, sample_rate=args.feats_extract_conf['fs'],
                  frontend_sha256=sha256(bundle / 'frontend.json'),
                  text_sha256=metadata['text_sha256'], wav_scp_sha256=metadata['wav_scp_sha256'],
                  audio_inventory_sha256=hashlib.sha256(json.dumps(audio_records, sort_keys=True).encode()).hexdigest(),
                  extractor='ESPnetGANTTSModel.collect_feats; no normalization; real frames only',
                  accumulation_dtype='float64', features={})
    for key, name in FEATURES.items():
        destination = output / 'stats' / f'{key}_stats.npz'
        np.savez(destination, **totals[key])
        old_count, old_mean, old_std = moments(config[name + '_conf']['stats_file'])
        count, mean, std = moments(destination)
        report['features'][key] = dict(frames=count, old_frames=old_count,
                                      old_mean=old_mean.tolist(), old_std=old_std.tolist(),
                                      mean=mean.tolist(), std=std.tolist(), sha256=sha256(destination))
    state = torch.load(output / 'model.pth', map_location='cpu', weights_only=True)
    report['changed_buffers'] = replace_buffers(state, resolve_config(output))
    torch.save(state, output / 'model.pth')
    report['model_sha256'] = sha256(output / 'model.pth')
    (output / 'normalization.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('items', 'hours', 'sample_rate')}), flush=True)


def migrate(source, bundle, output):
    """After stopping the old trainer, preserve its entire state in a new run directory."""
    from .convert import sha256
    if output.exists():
        raise ValueError('Migration output must be new')
    report = json.loads((bundle / 'normalization.json').read_text())
    for key in FEATURES:
        if sha256(bundle / 'stats' / f'{key}_stats.npz') != report['features'][key]['sha256']:
            raise ValueError('Normalization statistics changed after collection')
    old_config = yaml.safe_load((source / 'finetune.yaml').read_text())
    current_config = yaml.safe_load((source / 'config.yaml').read_text())
    config = resolve_config(bundle)
    for key in ('token_list', 'feats_extract_conf', 'pitch_extract_conf', 'energy_extract_conf'):
        if config[key] != old_config[key]:
            raise ValueError(f'Migration cannot change frontend or extraction: {key}')
    train_text = Path(next(p for p, name, _ in old_config['train_data_path_and_name_and_type'] if name == 'text'))
    train_wav = Path(next(p for p, name, _ in old_config['train_data_path_and_name_and_type'] if name == 'speech'))
    if (sha256(train_text) != report['text_sha256'] or sha256(train_wav) != report['wav_scp_sha256']):
        raise ValueError('Statistics were collected from different training data')
    checkpoint = source / 'checkpoint.pth'
    old_hash = sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_buffers(state['model'], old_config)
    changed = replace_buffers(state['model'], config)
    # Keep step counts and scalar history, but do not rank/average checkpoints
    # from different normalization regimes. Old validation remains auditable.
    for epoch_stats in state['reporter']['stats'].values():
        if 'valid' in epoch_stats:
            epoch_stats['valid_previous_normalization'] = epoch_stats.pop('valid')
    output.mkdir(parents=True)
    torch.save(state, output / 'checkpoint.pth')
    epoch = int(state['reporter']['epoch'])
    torch.save(state['model'], output / f'{epoch}epoch.pth')
    (output / 'latest.pth').symlink_to(f'{epoch}epoch.pth')
    # Preserve scalar history and optimizer/scheduler state. Old
    # checkpoints remain exclusively in source with their original statistics.
    if (source / 'tensorboard').exists():
        shutil.copytree(source / 'tensorboard', output / 'tensorboard')
    for filename, previous in [('finetune.yaml', old_config), ('config.yaml', current_config)]:
        updated = copy.deepcopy(previous)
        updated['output_dir'] = str(output.resolve())
        for name in FEATURES.values():
            updated[name + '_conf']['stats_file'] = config[name + '_conf']['stats_file']
        updated['init_param'] = [str((bundle / 'model.pth').resolve())] if filename == 'finetune.yaml' else []
        (output / filename).write_text(yaml.safe_dump(updated, allow_unicode=True))
    migration = dict(checkpoint_step=epoch * old_config['num_iters_per_epoch'],
                     changed_buffers=changed, source_checkpoint_sha256=old_hash,
                     checkpoint_sha256=sha256(output / 'checkpoint.pth'),
                     normalization_sha256=sha256(bundle / 'normalization.json'),
                     trainable_weights_changed=False, optimizer_scheduler_preserved=True,
                     training_step_preserved=True,
                     validation_history_phase='valid_previous_normalization',
                     checkpoint_ranking='new normalization phase only')
    (output / 'normalization_migration.json').write_text(json.dumps(migration, indent=2) + '\n')
    print(json.dumps(migration, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect_parser = commands.add_parser('collect')
    for name in ('bundle', 'voice-data', 'output'):
        collect_parser.add_argument('--' + name, type=Path, required=True)
    collect_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    migration_parser = commands.add_parser('migrate')
    for name in ('source', 'bundle', 'output'):
        migration_parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        collect(args.bundle, args.voice_data, args.output, args.device)
    else:
        migrate(args.source, args.bundle, args.output)


if __name__ == '__main__':
    main()
