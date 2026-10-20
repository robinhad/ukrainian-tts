"""Read-only checkpoint interventions for diagnosing r/l confusion.

Writes separate audio, durations and a recognizer manifest; never saves model weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .attention import enable_flash_attention
from .convert import load_strict, sha256
from .prepare import read_audio


def generate(args):
    torch.set_num_threads(2)
    torch.manual_seed(777)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audio').mkdir()
    rows = [json.loads(line) for line in (args.panel / 'selection.jsonl').read_text().splitlines()]
    model, config = load_strict(args.bundle, 'cuda', args.checkpoint)
    model.eval().requires_grad_(False)
    enable_flash_attention(model)
    tokens = config.token_list
    if args.train_data:
        import pandas as pd
        texts = pd.read_parquet(args.train_data / 'selection.parquet').set_index('utterance_id')
        wavs = dict(line.split(maxsplit=1) for line in (args.train_data / 'wav.scp').read_text().splitlines())
        phones = dict(line.split(maxsplit=1) for line in (args.train_data / 'text').read_text().splitlines())
        rows = [dict(utterance_id=uid, text=texts.loc[uid, 'text_sanitized'],
                     phonemes=phones[uid].split(), audio_path=wavs[uid],
                     token_ids=[tokens.index(p) for p in phones[uid].split()])
                for uid in sorted(phones)
                if 'r' in phones[uid].split() and 15 <= len(phones[uid].split()) <= 80][:12]
    ri, li = tokens.index('r'), tokens.index('l')
    embedding = model.tts.generator.encoder.embed[0].weight
    original_embedding = embedding.detach().clone()
    original_duration = model.tts.generator.duration_predictor.inference
    emitted = []

    def save(condition, index, row, result, rate):
        path = args.output / 'audio' / f'{condition}_{index:02d}.wav'
        wav = result['wav'].detach().cpu().numpy().reshape(-1)
        if not wav.size or not np.isfinite(wav).all():
            raise ValueError('Nonfinite or empty synthesis')
        sf.write(path, wav, rate, subtype='FLOAT')
        emitted.append(dict(id=row['utterance_id'], condition=condition,
                            audio_path=str(path), expected_phones=row['phonemes'],
                            text=row['text'], token_ids=row['token_ids'],
                            durations=result['duration'].detach().cpu().tolist(),
                            sample_rate=rate, hop_length=256, audio_sha256=sha256(path)))

    conditions = ['baseline_43k', 'r_input_as_l', 'r_contrast_4', 'r_contrast_8',
                  'r_contrast_16', 'r_duration_2', 'r_initial_english_R']
    if args.train_data:
        conditions = ['baseline_43k', 'reference_conditioning', 'r_contrast_8']
        for row in rows:
            emitted.append(dict(id=row['utterance_id'], condition='training_reference',
                                audio_path=row['audio_path'], expected_phones=row['phonemes'],
                                audio_sha256=sha256(Path(row['audio_path']))))
    for condition in conditions:
        with torch.no_grad():
            embedding.copy_(original_embedding)
            if condition.startswith('r_contrast_'):
                scale = int(condition.rsplit('_', 1)[1])
                embedding[ri].copy_(original_embedding[li] + scale * (original_embedding[ri] - original_embedding[li]))
            elif condition == 'r_initial_english_R':
                embedding[ri].copy_(original_embedding[tokens.index('ɹ')])
        for index, row in enumerate(rows, 1):
            ids = torch.tensor(row['token_ids'], device='cuda')
            original_ids = ids.clone()
            if condition == 'r_input_as_l':
                ids[ids == ri] = li
            if condition == 'r_duration_2':
                def scaled_duration(hs, masks=None):
                    durations = original_duration(hs, masks)
                    durations[:, original_ids == ri] = (durations[:, original_ids == ri] * 2).clamp_min(4)
                    return durations
                model.tts.generator.duration_predictor.inference = scaled_duration
            else:
                model.tts.generator.duration_predictor.inference = original_duration
            with torch.inference_mode():
                if condition == 'reference_conditioning':
                    signal = torch.tensor(read_audio(row['audio_path'], config.feats_extract_conf['fs']), device='cuda')[None]
                    features = model.collect_feats(text=ids[None], text_lengths=torch.tensor([len(ids)], device='cuda'),
                                                   speech=signal, speech_lengths=torch.tensor([signal.shape[1]], device='cuda'))
                    normalized = {}
                    for key, module in [('feats', 'normalize'), ('pitch', 'pitch_normalize'), ('energy', 'energy_normalize')]:
                        normalized[key] = getattr(model, module)(features[key], features[key + '_lengths'])[0][0]
                    result = model.tts.inference(text=ids, use_teacher_forcing=True, **normalized)
                else:
                    result = model.tts.inference(text=ids)
            save(condition, index, row, result, config.feats_extract_conf['fs'])
        print('Generated', condition, flush=True)
    model.tts.generator.duration_predictor.inference = original_duration
    with torch.no_grad():
        embedding.copy_(original_embedding)
    del embedding, model
    torch.cuda.empty_cache()

    if args.old_config:
        from espnet2.bin.tts_inference import Text2Speech
        from kaldiio import load_scp
        import yaml
        if yaml.safe_load(args.old_config.read_text())['token_list'] != tokens:
            raise ValueError('Old model comparison requires identical token IDs')
        old = Text2Speech(train_config=args.old_config, model_file=args.old_checkpoint, device='cuda')
        old.model.eval().requires_grad_(False)
        vectors = load_scp(str(args.xvectors))
        for condition in ['scratch_478k', 'scratch_r_input_as_l']:
            for index, row in enumerate(rows, 1):
                ids = torch.tensor(row['token_ids'], device='cuda')
                if condition == 'scratch_r_input_as_l':
                    ids[ids == ri] = li
                vector = torch.tensor(np.asarray(vectors[row['utterance_id']]).copy(), device='cuda').squeeze()
                with torch.inference_mode():
                    result = old.model.tts.inference(text=ids, spembs=vector)
                save(condition, index, row, result, int(old.fs))
            print('Generated', condition, flush=True)
        del old
    (args.output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in emitted))
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        checkpoint_sha256=sha256(args.checkpoint), frontend_sha256=sha256(args.bundle / 'frontend.json'),
        old_checkpoint_sha256=sha256(args.old_checkpoint) if args.old_checkpoint else None,
        changed_training_files=False, interventions='in-memory only; no optimizer steps; no model saved',
        reference_panel_sha256=sha256(args.panel / 'selection.jsonl'),
        conditions=sorted({row['condition'] for row in emitted}), seed=777), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'checkpoint', 'panel', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('old-config', 'old-checkpoint', 'xvectors'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--train-data', type=Path, help='Instead probe 12 fixed training recordings with reference conditioning')
    args = parser.parse_args()
    if args.old_config and not (args.old_checkpoint and args.xvectors):
        parser.error('Old model comparison requires checkpoint and speaker vectors')
    generate(args)


if __name__ == '__main__':
    main()
