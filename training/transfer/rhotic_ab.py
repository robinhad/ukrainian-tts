"""Controlled embedding adaptation from donor or pinned Ukrainian weights."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

from .convert import EMBEDDING, load_strict, sha256
from .train import training_config


def schedule(steps, interval):
    if steps <= 0 or interval <= 0 or steps % interval:
        raise ValueError('Steps must be a positive multiple of the evaluation interval')
    return range(1, steps // interval + 1)


def initialize(state, donor, tokens, arm):
    """Only the r row may differ; A is an exact copy of the trained checkpoint."""
    result = dict(state)
    if arm == 'english_r':
        result[EMBEDDING] = state[EMBEDDING].clone()
        result[EMBEDDING][tokens.index('r')].copy_(donor[EMBEDDING][tokens.index('ɹ')])
    elif arm != 'kept_r':
        raise ValueError('Unknown experimental arm')
    return result


def regroup_optimizer(optimizer, generator, embedding, factor):
    if optimizer.state or factor <= 0 or len(optimizer.param_groups) != 1:
        raise ValueError('Regroup only a fresh optimizer with a positive multiplier')
    others = [p for p in generator.parameters() if p is not embedding]
    optimizer.param_groups[0]['params'] = others
    optimizer.add_param_group({'params': [embedding], 'lr': optimizer.defaults['lr'] * factor})
    return optimizer


def ab_task():
    from .attention import flash_task
    class RhoticTask(flash_task()):
        @classmethod
        def add_task_arguments(cls, parser):
            super().add_task_arguments(parser)
            parser.add_argument('--rhotic_embedding_lr_factor', type=float, default=10.)

        @classmethod
        def build_optimizers(cls, args, model):
            optimizers = super().build_optimizers(args, model)
            embedding = model.tts.generator.encoder.embed[0].weight
            optimizers[0] = regroup_optimizer(optimizers[0], model.tts.generator, embedding,
                                             args.rhotic_embedding_lr_factor)
            covered = [id(p) for o in optimizers for group in o.param_groups for p in group['params']]
            expected = [id(p) for p in model.parameters() if p.requires_grad]
            if len(covered) != len(set(covered)) or set(covered) != set(expected):
                raise ValueError('Optimizer parameter coverage mismatch')
            print('Rhotic A/B learning rates:', [g['lr'] for o in optimizers for g in o.param_groups], flush=True)
            return optimizers
    return RhoticTask


def prepare(args):
    from .statistics import validate_buffers, resolve_config
    epochs = schedule(args.steps, args.evaluation_interval)
    if args.base_step < 0:
        raise ValueError('Base step cannot be negative')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    donor = torch.load(args.donor_bundle / 'model.pth', map_location='cpu', weights_only=True)
    config = resolve_config(args.bundle)
    validate_buffers(state, config)
    tokens = config['token_list']
    if yaml.safe_load((args.donor_bundle / 'config.yaml').read_text())['token_list'] != tokens:
        raise ValueError('Donor and target converted token inventory mismatch')
    mapping = json.loads((args.donor_bundle / 'mapping.json').read_text())['mappings']
    row = next(row for row in mapping if row['target_token'] == 'ɹ')
    if row['source_token'] != 'R' or row['method'] != 'equivalent':
        raise ValueError('Expected the original English R embedding')
    if args.base_step == 0:
        # A fresh English start may use target-voice normalizers, but must retain
        # every converted donor model weight before the requested r replacement.
        normalizers = {f'{name}.{stat}' for name in ['normalize', 'pitch_normalize', 'energy_normalize']
                       for stat in ['mean', 'std']}
        if state.keys() != donor.keys() or any(not torch.equal(state[k], donor[k])
                                              for k in state if k not in normalizers):
            raise ValueError('Base step zero requires unchanged converted English donor weights')
    for arm in (['kept_r', 'english_r'] if args.arm == 'both' else [args.arm]):
        root = args.output / arm
        root.mkdir(parents=True, exist_ok=False)
        initial = initialize(state, donor, tokens, arm)
        torch.save(initial, root / 'initial.pth')
        settings = training_config(args.bundle, args.voice_data / 'train', args.voice_data / 'dev', root / 'train', 'tetiana')
        settings.update(init_param=[str((root / 'initial.pth').resolve())],
                        max_epoch=len(epochs), num_iters_per_epoch=args.evaluation_interval, batch_size=8,
                        keep_nbest_models=[4], num_att_plot=0, log_interval=10,
                        use_tensorboard=True, use_wandb=False,
                        transfer_attention_backend='flash2', rhotic_embedding_lr_factor=10.)
        settings['optim_conf']['lr'] = settings['optim2_conf']['lr'] = 2e-5
        # Preserve the donor recipe's decay per 1,000 optimizer steps.
        for key in ['scheduler_conf', 'scheduler2_conf']:
            settings[key]['gamma'] = settings[key]['gamma'] ** (args.evaluation_interval / 1000)
        (root / 'settings.yaml').write_text(yaml.safe_dump(settings, allow_unicode=True))
        record = dict(arm=arm, base_step=args.base_step, additional_steps=args.steps, seed=settings['seed'],
                      initialization_origin='converted English donor' if args.base_step == 0 else 'trained Ukrainian checkpoint',
                      batch_size=8, evaluation_interval=args.evaluation_interval, fresh_optimizers=True,
                      embedding_lr=2e-4, backbone_lr=2e-5, discriminator_lr=2e-5,
                      source_checkpoint_sha256=sha256(args.checkpoint), initial_sha256=sha256(root / 'initial.pth'),
                      donor_sha256=sha256(args.donor_bundle / 'model.pth'),
                      frontend_sha256=sha256(args.bundle / 'frontend.json'),
                      bundle=str(args.bundle.resolve()), voice_data=str(args.voice_data.resolve()),
                      only_changed_row='r' if arm == 'english_r' else None)
        (root / 'experiment.json').write_text(json.dumps(record, indent=2) + '\n')
        public = {k: v for k, v in record.items() if k not in ['bundle', 'voice_data']}
        (root / 'public_experiment.json').write_text(json.dumps(public, indent=2) + '\n')
        print(json.dumps(public), flush=True)


def train(root, epochs):
    from .statistics import validate_buffers
    config = yaml.safe_load((root / 'settings.yaml').read_text())
    if not 1 <= epochs <= config['max_epoch']:
        raise ValueError('Epoch is outside the prepared training schedule')
    config['max_epoch'] = epochs
    output = root / 'train'
    if epochs == 1:
        if output.exists():
            raise ValueError('Fresh arm refuses an existing training directory')
    else:
        checkpoint = torch.load(output / 'checkpoint.pth', map_location='cpu', weights_only=False)
        if checkpoint['reporter']['epoch'] != epochs - 1:
            raise ValueError('Expected the preceding evaluation-interval checkpoint')
        validate_buffers(checkpoint['model'], config)
        expected = [[2e-5, 2e-4], [2e-5]]
        for optimizer, rates in zip(checkpoint['optimizers'], expected):
            if len(optimizer['param_groups']) != len(rates):
                raise ValueError('Optimizer groups changed')
            for group, rate in zip(optimizer['param_groups'], rates):
                if not np.isclose(group['initial_lr'], rate):
                    raise ValueError('Optimizer learning-rate identity changed')
        config.update(resume=True, init_param=[])
    path = root / f'epoch_{epochs}.yaml'
    path.write_text(yaml.safe_dump(config, allow_unicode=True))
    ab_task().main(cmd=['--config', str(path)])


def evaluate(root, step):
    from .attention import enable_flash_attention
    from .rhotic_diagnostic import score
    from .rhotic_report import summary
    from training.scripts.score_checkpoint_listening import QualityModels
    from torch.utils.tensorboard import SummaryWriter
    record = json.loads((root / 'experiment.json').read_text())
    interval = record['evaluation_interval']
    if step < 0 or step > record['additional_steps'] or step % interval:
        raise ValueError('Evaluation step is outside the prepared schedule')
    bundle, dev = Path(record['bundle']), Path(record['voice_data']) / 'dev'
    checkpoint = root / ('initial.pth' if step == 0 else f'train/{step // interval}epoch.pth')
    output = root / f'evaluation_{step}'
    output.mkdir(exist_ok=False)
    (output / 'audio').mkdir()
    model, config = load_strict(bundle, 'cuda', checkpoint)
    model.eval().requires_grad_(False)
    enable_flash_attention(model)
    texts = dict(line.split(maxsplit=1) for line in (dev / 'text').read_text().splitlines())
    wavs = dict(line.split(maxsplit=1) for line in (dev / 'wav.scp').read_text().splitlines())
    if texts.keys() != wavs.keys():
        raise ValueError('Development manifest mismatch')
    rows = []
    torch.set_num_threads(2)
    torch.manual_seed(record['seed'])
    for index, uid in enumerate(sorted(texts), 1):
        phones = texts[uid].split()
        ids = torch.tensor([config.token_list.index(p) for p in phones], device='cuda')
        with torch.inference_mode():
            result = model.tts.inference(text=ids)
        wav = result['wav'].detach().cpu().numpy().reshape(-1)
        if not len(wav) or not np.isfinite(wav).all():
            raise ValueError('Invalid evaluation synthesis')
        destination = output / 'audio' / f'{index:02d}.wav'
        sf.write(destination, wav, config.feats_extract_conf['fs'], subtype='FLOAT')
        for condition, path in [('synthesis', destination), ('reference', Path(wavs[uid]))]:
            rows.append(dict(id=uid, condition=condition, expected_phones=phones, audio_path=str(path),
                             audio_sha256=sha256(path)))
    del model
    torch.cuda.empty_cache()
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    score(manifest, output / 'phones.jsonl', 'cuda')
    torch.cuda.empty_cache()
    parsed = [json.loads(line) for line in (output / 'phones.jsonl').read_text().splitlines()]
    counts = summary(parsed)
    metrics = {}
    for role, value in counts.items():
        for phone in ['r', 'l']:
            counter = value[phone + '_counts']
            metrics[f'{role}_{phone}_exact_rate'] = counter.get(phone, 0) / sum(counter.values())
            if phone == 'r':
                metrics[f'{role}_r_as_l_rate'] = counter.get('l', 0) / sum(counter.values())
    quality = QualityModels(yaml.safe_load(Path('training/conf/quality.yaml').read_text())['models'])
    quality_rows = []
    for row in rows:
        audio, rate = sf.read(row['audio_path'], dtype='float32')
        quality_rows.append(dict(id=row['id'], condition=row['condition'], **quality.score(audio, rate)))
    for role in ['synthesis', 'reference']:
        for metric in ['sigmos_overall', 'audiobox_pq']:
            metrics[f'{role}_{metric}_median'] = float(np.median([r[metric] for r in quality_rows if r['condition'] == role]))
    result = dict(arm=record['arm'], additional_steps=step, base_step=record['base_step'],
                  checkpoint_sha256=sha256(checkpoint), split='development; 13 items; no training items',
                  phone_counts=counts, metrics=metrics)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (output / 'quality.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in quality_rows))
    with SummaryWriter(str(root / 'train/tensorboard/valid' if step else root / 'baseline_events/valid')) as writer:
        for key, value in metrics.items():
            writer.add_scalar('rhotic/' + key, value, step)
    parts = ['<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Rhotic A/B evaluation</title><style>body{max-width:900px;margin:auto;padding:24px;font:18px/1.5 system-ui}audio{width:100%}</style>',
             f'<h1>{record["arm"]}: +{step} кроків від {record["base_step"]}</h1><p>13 validation-записів. Автоматичні фонетичні мітки та MOS — діагностика, не оцінка людини.</p>',
             '<pre>' + html.escape(json.dumps(metrics, indent=2)) + '</pre>']
    import base64
    for row in rows:
        src = base64.b64encode(Path(row['audio_path']).read_bytes()).decode()
        parts.append(f'<p>{html.escape(row["id"])} · {row["condition"]}</p><audio controls preload="none" src="data:audio/wav;base64,{src}"></audio>')
    (output / 'preview.html').write_text('\n'.join(parts) + '</html>')
    print('RHOTIC_EVALUATION', json.dumps(result, ensure_ascii=False), flush=True)


def run(root):
    # A failed stage stops the arm; checkpoint/evaluation files remain for diagnosis.
    record = json.loads((root / 'experiment.json').read_text())
    interval = record['evaluation_interval']
    for epoch in schedule(record['additional_steps'], interval):
        subprocess.run([sys.executable, '-m', __package__ + '.rhotic_ab', 'train',
                        '--root', str(root), '--epochs', str(epoch)], check=True)
        if epoch == 1 and (root / 'baseline_events').exists():
            import shutil
            shutil.copytree(root / 'baseline_events', root / 'train/tensorboard', dirs_exist_ok=True)
        subprocess.run([sys.executable, '-m', __package__ + '.rhotic_ab', 'evaluate',
                        '--root', str(root), '--step', str(epoch * interval)], check=True)


def smoke(root):
    """Exercise both real GAN updates on a disposable in-memory model."""
    from .attention import enable_flash_attention
    from espnet2.tasks.gan_tts import GANTTSTask
    from espnet2.train.distributed_utils import DistributedOption
    record = json.loads((root / 'experiment.json').read_text())
    original_hash = sha256(root / 'initial.pth')
    torch.set_num_threads(4)
    torch.manual_seed(777)
    model, _ = load_strict(Path(record['bundle']), 'cuda', root / 'initial.pth')
    enable_flash_attention(model)
    config = yaml.safe_load((root / 'settings.yaml').read_text())
    args = argparse.Namespace(**config)
    args.num_workers = 0
    args.num_iters_per_epoch = 1
    iterator = GANTTSTask.build_iter_factory(args, DistributedOption(), 'train')
    _, batch = next(iter(iterator.build_iter(1)))
    batch = {key: value.to('cuda') for key, value in batch.items()}
    optimizers = ab_task().build_optimizers(args, model)
    model.train()
    losses = []
    before = model.tts.generator.encoder.embed[0].weight.detach().clone()
    for index, generator in enumerate([True, False]):
        model.zero_grad(set_to_none=True)
        loss = model(**batch, forward_generator=generator)['loss']
        assert torch.isfinite(loss).all()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizers[index].step()
        losses.append(float(loss.detach()))
    assert not torch.equal(before, model.tts.generator.encoder.embed[0].weight)
    assert sha256(root / 'initial.pth') == original_hash
    result = dict(arm=record['arm'], losses=losses,
                  learning_rates=[g['lr'] for o in optimizers for g in o.param_groups],
                  optimizer_steps=[sorted({int(s['step']) for s in o.state.values() if 'step' in s}) for o in optimizers],
                  embedding_updated=True, initial_checkpoint_unchanged=True,
                  disposable_model=True, model_saved=False)
    (root / 'smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    for key in ['bundle', 'donor-bundle', 'checkpoint', 'voice-data', 'output']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--arm', choices=['both', 'kept_r', 'english_r'], default='both')
    p.add_argument('--base-step', type=int, default=43000)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--evaluation-interval', type=int, default=500)
    for command in ['train', 'evaluate', 'run', 'smoke']:
        p = subs.add_parser(command)
        p.add_argument('--root', type=Path, required=True)
        if command == 'train':
            p.add_argument('--epochs', type=int, required=True)
        if command == 'evaluate':
            p.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'train': train(args.root, args.epochs)
    elif args.command == 'evaluate': evaluate(args.root, args.step)
    elif args.command == 'smoke': smoke(args.root)
    else: run(args.root)


if __name__ == '__main__':
    main()
