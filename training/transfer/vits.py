"""Multi-speaker VITS phonetic transfer with donor-compatible speaker conditioning."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import kaldiio
import numpy as np
import soundfile as sf
import torch
import yaml

from .convert import MODEL_KEYS, initialize_rows, sha256
from .frontend import TransferFrontend
from .mapping import Features, map_tokens
from .rhotic_ab import regroup_optimizer

EMBEDDING = 'tts.generator.text_encoder.emb.weight'
NORMALIZERS = ('normalize', 'pitch_normalize', 'energy_normalize')


def validate_config(config):
    if config['tts'] != 'vits' or config['feats_extract'] != 'linear_spectrogram':
        raise ValueError('Expected the LibriTTS linear-spectrogram VITS architecture')
    if any(config.get(key) is not None for key in NORMALIZERS + ('pitch_extract', 'energy_extract')):
        raise ValueError('This donor has no feature normalization, pitch or energy extractors')
    params = config['tts_conf']['generator_params']
    if params['spk_embed_dim'] != 512 or config['tts_conf']['sampling_rate'] != 22050:
        raise ValueError('Unexpected donor audio or speaker-vector settings')


def task():
    from espnet2.tasks.gan_tts import GANTTSTask
    class VITSTransferTask(GANTTSTask):
        @classmethod
        def add_task_arguments(cls, parser):
            super().add_task_arguments(parser)
            parser.add_argument('--embedding_lr_factor', type=float, default=10.)

        @classmethod
        def build_optimizers(cls, args, model):
            optimizers = super().build_optimizers(args, model)
            embedding = model.tts.generator.text_encoder.emb.weight
            optimizers[0] = regroup_optimizer(optimizers[0], model.tts.generator, embedding,
                                             args.embedding_lr_factor)
            covered = [id(p) for o in optimizers for g in o.param_groups for p in g['params']]
            if len(covered) != len(set(covered)) or set(covered) != {id(p) for p in model.parameters() if p.requires_grad}:
                raise ValueError('Optimizer parameter coverage mismatch')
            print('VITS learning rates:', [g['lr'] for o in optimizers for g in o.param_groups], flush=True)
            return optimizers
    return VITSTransferTask


def load(bundle, checkpoint=None, device='cpu'):
    from espnet2.tasks.gan_tts import GANTTSTask
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    validate_config(config)
    TransferFrontend(config['token_list']).verify(bundle / 'frontend.json')
    model = GANTTSTask.build_model(argparse.Namespace(**config))
    state = torch.load(checkpoint or bundle / 'model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if any(getattr(model, key) is not None for key in NORMALIZERS):
        raise ValueError('Unexpected loaded normalization module')
    return model.to(device), config


def convert(args):
    from espnet2.tasks.gan_tts import GANTTSTask
    source = yaml.safe_load(args.source_config.read_text())
    validate_config(source)
    if source['g2p'] != 'g2p_en_no_space':
        raise ValueError('Expected ARPAbet English source inventory')
    tokens = yaml.safe_load(args.target_config.read_text())['token_list']
    state = torch.load(args.source_model, map_location='cpu', weights_only=True)
    if state[EMBEDDING].shape != (len(source['token_list']), 192):
        raise ValueError('Source embedding inventory mismatch')
    mappings = map_tokens(source['token_list'], tokens, Features(args.features), .15)
    r = next(row for row in mappings if row['target_token'] == 'r')
    r.update(source_id=source['token_list'].index('R'), source_token='R', source_ipa='ɹ',
             method='explicit_rhotic_override', distance=None, coverage=None,
             reason='English R initialization selected by the Ukrainian rhotic experiment')
    converted = dict(state)
    converted[EMBEDDING] = initialize_rows(state[EMBEDDING], mappings, 777)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.save(converted, output / 'model.pth')
    config = {key: copy.deepcopy(source[key]) for key in MODEL_KEYS}
    config['tts_conf']['generator_params']['vocabs'] = len(tokens)
    compatibility = []
    scale = [k for k in state if k.startswith('tts.discriminator.msd.')]
    if any(k.endswith('.weight') for k in scale) and not any(k.endswith(('.weight_g', '.weight_orig')) for k in scale):
        discriminator = config['tts_conf']['discriminator_params']
        discriminator['follow_official_norm'] = False
        discriminator['scale_discriminator_params'].update(use_weight_norm=False, use_spectral_norm=False)
        compatibility.append('Scale-discriminator normalization matched to actual donor weights')
    config.update(token_list=tokens, token_type='phn', g2p=None, cleaner=None,
                  use_preprocessor=True, resume=False, freeze_param=[], init_param=[],
                  ignore_init_mismatch=False, batch_type='unsorted', batch_size=8,
                  valid_batch_size=4, num_workers=4, ngpu=1, seed=777,
                  num_iters_per_epoch=1000, max_epoch=50, use_amp=False, grad_clip=-1,
                  generator_first=source['generator_first'], use_wandb=False,
                  use_tensorboard=True, num_att_plot=0, log_interval=10, keep_nbest_models=[3],
                  val_scheduler_criterion=['valid', 'generator_loss'],
                  early_stopping_criterion=['valid', 'generator_loss', 'min'],
                  best_model_criterion=[['valid', 'generator_mel_loss', 'min']])
    for suffix in ['', '2']:
        for name in ['optim', 'scheduler']:
            for ending in ['', '_conf']:
                key = name + suffix + ending
                config[key] = copy.deepcopy(source[key])
        config['optim' + suffix + '_conf']['lr'] *= .1
    temporary = output / 'settings.yaml'
    temporary.write_text(yaml.safe_dump(config, allow_unicode=True))
    config = vars(GANTTSTask.get_parser().parse_args(['--config', str(temporary), '--output_dir', 'exp']))
    config.pop('config', None)
    config.pop('required', None)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
    temporary.unlink()
    (output / 'tokens.txt').write_text('\n'.join(tokens) + '\n')
    (output / 'frontend.json').write_text(json.dumps(TransferFrontend(tokens).snapshot(), ensure_ascii=False, indent=2) + '\n')
    with (output / 'mapping.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mappings[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(mappings)
    report = dict(donor='espnet/kan-bayashi_libritts_xvector_vits', source_checkpoint_sha256=sha256(args.source_model),
                  source_config_sha256=sha256(args.source_config), features_sha256=sha256(args.features),
                  seed=777, source_rows=len(source['token_list']), target_rows=len(tokens),
                  embedding_dimension=192, unchanged_tensors=len(state)-1,
                  mappings=mappings, sample_rate=22050, speaker_vector_dimension=512,
                  attention='Donor relative-position attention retained; JETS FlashAttention adapter is incompatible',
                  normalization={'mel': 'not used as normalized input; waveform mel loss retained',
                                 'input': '513-bin unnormalized linear amplitude spectrogram',
                                 'pitch': None, 'energy': None, 'dataset_statistics_required': False},
                  compatibility_adjustments=compatibility)
    model, _ = load(output)
    assert all(torch.equal(state[k], model.state_dict()[k]) for k in state if k != EMBEDDING)
    assert torch.equal(model.state_dict()[EMBEDDING][tokens.index('r')], state[EMBEDDING][source['token_list'].index('R')])
    assert all(p.requires_grad for p in model.parameters())
    (output / 'mapping.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'mappings'}), flush=True)


def prepare(args):
    import pandas as pd
    bundle = args.bundle.resolve()
    config = yaml.safe_load((bundle / 'config.yaml').read_text())
    validate_config(config)
    frontend = TransferFrontend(config['token_list'])
    frontend.verify(bundle / 'frontend.json')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary, panels = {}, []
    all_ids = set()
    for split, phase in [('train', 'train'), ('dev', 'valid')]:
        prepared = args.prepared / split
        metadata = json.loads((prepared / 'metadata.json').read_text())
        if metadata['sample_rate'] != 22050 or metadata['frontend_sha256'] != sha256(bundle / 'frontend.json'):
            raise ValueError('Prepared Ukrainian frontend/sample rate differs from donor conversion')
        for filename, key in [('text', 'text_sha256'), ('wav.scp', 'wav_scp_sha256')]:
            if sha256(prepared / filename) != metadata[key]:
                raise ValueError('Prepared data was modified')
        text = dict(line.split(maxsplit=1) for line in (prepared / 'text').read_text().splitlines())
        wavs = dict(line.split(maxsplit=1) for line in (prepared / 'wav.scp').read_text().splitlines())
        embeddings = dict(line.split(maxsplit=1) for line in (args.xvectors / split / 'xvector.scp').read_text().splitlines())
        ids = set(embeddings)
        if text.keys() != wavs.keys() or not ids <= text.keys() or len(ids) < .99 * len(text):
            raise ValueError('Audio/text/vector coverage mismatch')
        if ids & all_ids:
            raise ValueError('Training/validation overlap')
        all_ids.update(ids)
        frame = pd.read_parquet(args.manifests / f'quality_v12_{split}.parquet')
        frame = frame[frame.utterance_id.isin(ids)].copy()
        if set(frame.utterance_id) != ids or frame.source_id.str.contains('voa', case=False).any():
            raise ValueError('Expected only the existing non-VOA corpus')
        directory = output / split
        directory.mkdir()
        for filename, rows in [('text', text), ('wav.scp', wavs), ('xvector.scp', embeddings)]:
            (directory / filename).write_text(''.join(f'{uid} {rows[uid]}\n' for uid in sorted(ids)))
        config[phase + '_data_path_and_name_and_type'] = [
            [str(directory / 'text'), 'text', 'text'],
            [str(directory / 'wav.scp'), 'speech', 'sound'],
            [str(directory / 'xvector.scp'), 'spembs', 'kaldi_ark']]
        config[phase + '_shape_file'] = [str(directory / 'text')]
        summary[split] = dict(items=len(frame), hours=float(frame.duration.sum()/3600),
                              speaker_labels=int(frame[['source_id','speaker_id']].drop_duplicates().shape[0]),
                              speaker_count_note='Many labels are per-recording pseudo-identities; not a verified count of people',
                              sources={s:dict(items=len(g),hours=float(g.duration.sum()/3600)) for s,g in frame.groupby('source_id')},
                              excluded_vector_ids=sorted(text.keys()-ids),
                              hashes={name:sha256(directory/name) for name in ['text','wav.scp','xvector.scp']})
        if split == 'dev':
            for _, group in frame.groupby('source_id'):
                for row in group.sort_values('utterance_id').head(2).itertuples():
                    panels.append(dict(id=row.utterance_id, source_id=row.source_id, speaker_id=row.speaker_id,
                                       text=row.text_sanitized, expected_phones=text[row.utterance_id].split(),
                                       audio_path=wavs[row.utterance_id], vector=embeddings[row.utterance_id]))
    config.update(output_dir=str(output/'train_model'), init_param=[str(bundle/'model.pth')],
                  embedding_lr_factor=10., max_epoch=args.steps//1000, batch_size=args.batch_size)
    if args.steps <= 0 or args.steps % 5000:
        raise ValueError('Training target must be a positive multiple of 5000')
    (output/'settings.yaml').write_text(yaml.safe_dump(config,allow_unicode=True))
    task().get_parser().parse_args(['--config',str(output/'settings.yaml')])
    record=dict(bundle=str(bundle), steps=args.steps, batch_size=args.batch_size,
                evaluation_interval=5000, splits=summary, speaker_vectors=json.loads((args.xvectors/'report.json').read_text()),
                base_step=0, initialization='English donor with Ukrainian phonetic rows and r <- English R',
                embedding_lr=2e-4, backbone_lr=2e-5, fresh_optimizers=True,
                feature_statistics='Not applicable: donor uses unnormalized linear spectra; pitch/energy extractors absent',
                speaker_conditioning='Per-utterance Kaldi SITW x-vector; same extractor for training and inference')
    (output/'experiment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (output/'panel.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in panels))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def train(root, epochs, resume=False):
    config=yaml.safe_load((root/'settings.yaml').read_text())
    if not 0 < epochs <= config['max_epoch']:
        raise ValueError('Invalid training target')
    config['max_epoch']=epochs
    output=root/'train_model'
    if resume:
        state=torch.load(output/'checkpoint.pth',map_location='cpu',weights_only=False)
        if not 0 < state['reporter']['epoch'] < epochs:
            raise ValueError('Resume must advance beyond a saved epoch')
        expected=[[2e-5,2e-4],[2e-5]]
        for optimizer,rates in zip(state['optimizers'],expected):
            if len(optimizer['param_groups'])!=len(rates) or any(not np.isclose(g['initial_lr'],r) for g,r in zip(optimizer['param_groups'],rates)):
                raise ValueError('Optimizer groups changed')
        previous=yaml.safe_load((output/'config.yaml').read_text())
        for key in ['token_list','tts_conf','feats_extract_conf','embedding_lr_factor','train_data_path_and_name_and_type','valid_data_path_and_name_and_type']:
            if previous[key]!=config[key]:raise ValueError(f'Resume changed {key}')
        config.update(resume=True,init_param=[])
    elif output.exists():
        raise ValueError('Fresh run refuses an existing training directory')
    for split in ['train','dev']:
        record=json.loads((root/'experiment.json').read_text())
        for name,expected in record['splits'][split]['hashes'].items():
            if sha256(root/split/name)!=expected:raise ValueError('Prepared training manifests changed')
    path=root/f'train_to_{epochs*1000}.yaml'
    path.write_text(yaml.safe_dump(config,allow_unicode=True))
    task().main(cmd=['--config',str(path)])


def smoke(root):
    from espnet2.tasks.gan_tts import GANTTSTask
    from espnet2.train.distributed_utils import DistributedOption
    config=yaml.safe_load((root/'settings.yaml').read_text())
    record=json.loads((root/'experiment.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(777)
    model,_=load(Path(record['bundle']),device='cuda')
    model.train()
    args=argparse.Namespace(**config);args.num_workers=0;args.num_iters_per_epoch=1
    iterator=GANTTSTask.build_iter_factory(args,DistributedOption(),'train')
    _,batch=next(iter(iterator.build_iter(1)))
    batch={k:v.to('cuda') for k,v in batch.items()}
    optimizers=task().build_optimizers(args,model)
    before=model.tts.generator.text_encoder.emb.weight.detach().clone()
    start=time.monotonic();losses=[]
    for forward_generator in ([True,False] if config['generator_first'] else [False,True]):
        model.zero_grad(set_to_none=True)
        loss=model(**batch,forward_generator=forward_generator)['loss']
        assert torch.isfinite(loss).all();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizers[0 if forward_generator else 1].step()
        losses.append(float(loss.detach()))
    assert not torch.equal(before,model.tts.generator.text_encoder.emb.weight)
    model.eval()
    with torch.inference_mode():
        torch.manual_seed(777)
        result=model.tts.inference(text=batch['text'][0,:batch['text_lengths'][0]],spembs=batch['spembs'][0],max_len=1500)
        torch.manual_seed(777)
        other=model.tts.inference(text=batch['text'][0,:batch['text_lengths'][0]],spembs=batch['spembs'][1],max_len=1500)
    audio=result['wav'].detach().cpu().numpy()
    assert audio.size and np.isfinite(audio).all()
    assert torch.isfinite(other['wav']).all() and not torch.equal(result['wav'],other['wav'])
    report=dict(losses=losses,elapsed_seconds=time.monotonic()-start,batch_size=len(batch['speech']),
                learning_rates=[g['lr'] for o in optimizers for g in o.param_groups],
                embedding_updated=True,finite_inference=True,model_saved=False,
                speaker_vector_dimension=batch['spembs'].shape[-1],normalization_modules_absent=True,
                different_speaker_vectors_change_synthesis=True)
    (root/'smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def infer(args):
    record=json.loads((args.root/'experiment.json').read_text())
    model,config=load(Path(record['bundle']),args.checkpoint,'cuda')
    model.eval().requires_grad_(False)
    _,ids=TransferFrontend(config['token_list']).encode(args.text)
    vectors={}
    for split in ['train','dev']:
        vectors.update(line.split(maxsplit=1) for line in (args.root/split/'xvector.scp').read_text().splitlines())
    vector=kaldiio.load_mat(vectors[args.reference_id])
    if vector.shape!=(512,) or not np.isfinite(vector).all():raise ValueError('Invalid speaker vector')
    torch.manual_seed(777)
    with torch.inference_mode():
        result=model.tts.inference(text=torch.tensor(ids,device='cuda'),spembs=torch.from_numpy(vector).to('cuda'),max_len=5168)
    audio=result['wav'].cpu().numpy().reshape(-1)
    if not audio.size or not np.isfinite(audio).all():raise ValueError('Invalid synthesis')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(args.output,audio,22050,subtype='FLOAT')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('convert')
    for key in ['source-config','source-model','target-config','features','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p=subs.add_parser('prepare')
    for key in ['bundle','prepared','xvectors','manifests','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=50000);p.add_argument('--batch-size',type=int,default=8)
    for name in ['train','smoke']:
        p=subs.add_parser(name);p.add_argument('--root',type=Path,required=True)
        if name=='train':p.add_argument('--epochs',type=int,required=True);p.add_argument('--resume',action='store_true')
    p=subs.add_parser('infer')
    for key in ['root','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference-id',required=True);p.add_argument('--text',required=True)
    args=parser.parse_args()
    if args.command=='convert':convert(args)
    elif args.command=='prepare':prepare(args)
    elif args.command=='train':train(args.root,args.epochs,args.resume)
    elif args.command=='infer':infer(args)
    else:smoke(args.root)


if __name__=='__main__':main()
