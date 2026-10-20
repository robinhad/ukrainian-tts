"""Export fixed single-voice held-out samples with the transfer frontend and MOS."""
import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from training.scripts.generate_checkpoint_listening import select, write_pages
from .attention import enable_flash_attention
from .convert import load_strict, sha256
from .frontend import TransferFrontend
from .prepare import read_audio


def select_heldout(evaluation, used, speaker, source, count):
    selected = evaluation[evaluation.speaker_id.eq(speaker) & evaluation.source_id.eq(source)]
    if selected.utterance_id.duplicated().any() or not selected.split.str.endswith('_eval').all():
        raise ValueError('Expected unique evaluation records')
    if (set(selected.utterance_id) & set(used.utterance_id)
            or set(selected.audio_sha256) & set(used.audio_sha256)):
        raise ValueError('Evaluation overlaps training or development')
    return select(selected.to_dict('records'), count)


def generate(args):
    checkpoint = args.checkpoint.resolve(strict=True)
    match = re.fullmatch(r'(\d+)epoch', checkpoint.stem)
    if not match:
        raise ValueError('Use an epoch checkpoint from the 1,000-steps-per-epoch transfer run')
    step = int(match[1]) * 1000
    output = args.output / f'listening_{step}'
    if output.exists():
        raise ValueError(f'Listening output already exists: {output}')
    train = pd.read_parquet(args.voice_data / 'train/selection.parquet')
    dev = pd.read_parquet(args.voice_data / 'dev/selection.parquet')
    voice = json.loads((args.voice_data / 'selection.json').read_text())
    speaker, source = voice['speaker_id'], voice['source_id']
    selected = select_heldout(pd.read_parquet(args.evaluation), pd.concat([train, dev]),
                              speaker, source, args.count)
    originals = pd.read_parquet(args.originals).set_index('utterance_id', verify_integrity=True)
    # Validate inputs before allocating a model or creating any output.
    for row in selected:
        raw = originals.loc[row['utterance_id']]
        if (raw.source_id != source or raw.speaker_id != speaker
                or raw.text_sanitized != row['text_sanitized']):
            raise ValueError('Original and processed references disagree')
        for path, digest in [(row['audio_path'], row['audio_sha256']),
                             (raw.audio_path, raw.audio_sha256)]:
            if sha256(Path(path)) != digest:
                raise ValueError('Reference checksum mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(777)
    np.random.seed(777)
    checkpoint_hash = sha256(checkpoint)
    model, config = load_strict(args.bundle, args.device, checkpoint)
    frontend = TransferFrontend(config.token_list)
    frontend.verify(args.bundle / 'frontend.json')
    encoded = [frontend.encode(row['text_sanitized']) for row in selected]
    if args.device == 'cuda':
        enable_flash_attention(model)
    model.eval()
    rate = config.feats_extract_conf['fs']
    (output / 'audio').mkdir(parents=True)
    rows = []
    for index, (row, (phones, ids)) in enumerate(zip(selected, encoded), 1):
        with torch.inference_mode():
            wav = model.tts.inference(
                text=torch.tensor(ids, device=args.device, dtype=torch.long))['wav'].flatten().cpu().numpy()
        if not wav.size or not np.isfinite(wav).all() or not np.any(wav):
            raise ValueError('Empty, silent or nonfinite synthesis')
        if np.max(np.abs(wav)) >= 1:
            raise ValueError('Synthesis would clip PCM; do not silently normalize listening examples')
        paths = {role: f'audio/{index:02d}_{role}.wav'
                 for role in ('synthesis', 'reference', 'processed')}
        sf.write(output / paths['synthesis'], wav, rate, subtype='PCM_24')
        raw = originals.loc[row['utterance_id']]
        shutil.copyfile(raw.audio_path, output / paths['reference'])
        sf.write(output / paths['processed'], read_audio(row['audio_path'], rate), rate, subtype='FLOAT')
        rows.append(dict(utterance_id=row['utterance_id'], source_id=source,
                         text=row['text_sanitized'], duration_seconds=len(wav) / rate,
                         phonemes=phones, token_ids=ids, **paths,
                         audio_sha256={role: sha256(output / path) for role, path in paths.items()}))
        print(f'Generated {index}/{len(selected)} · {len(wav) / rate:.2f}s', flush=True)
    report = dict(model_kind='phonetic_transfer', checkpoint_step=step,
                  checkpoint_sha256=checkpoint_hash, frontend_sha256=sha256(args.bundle / 'frontend.json'),
                  sample_count=len(rows), sample_rate=rate, speaker_id=speaker, source_id=source,
                  train_items=len(train), train_hours=float(train.duration.sum() / 3600),
                  seed=777, selection='fixed seeded held-out evaluation texts; no quality-based selection',
                  speaker_conditioning='none; single learned voice',
                  postprocessing='none; synthesis PCM24 serialization only',
                  attention_backend='flash2' if args.device == 'cuda' else 'espnet')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'selection.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    if args.skip_quality:
        write_pages(output, rows, step)
    else:
        from training.scripts.score_checkpoint_listening import score_listening
        score_listening(output, args.quality_config)
    print(f'Listening page: {output / "preview.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'checkpoint', 'voice-data', 'evaluation', 'originals', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--quality-config', type=Path, default=Path('training/conf/quality.yaml'))
    parser.add_argument('--skip-quality', action='store_true')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    generate(args)


if __name__ == '__main__':
    main()
