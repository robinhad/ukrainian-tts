"""Independent phone recognition for a matched rhotic diagnostic audio manifest.

Recognizer substitutions are diagnostic evidence, not a human pronunciation score.
Input JSONL: id, condition, audio_path, expected_phones. Runtime paths stay local.
"""
import argparse
import itertools
import json
from pathlib import Path
import unicodedata

import numpy as np
import torch

from .prepare import read_audio

RECOGNIZER = 'facebook/wav2vec2-xlsr-53-espeak-cv-ft'
REVISION = '2c733782da5604684829819a5eb744c193fe9398'


def normalize(phones):
    return [p.lstrip('ˈˌ').replace('͡', '').replace('g', 'ɡ') for p in phones
            if p and not all(unicodedata.category(c).startswith('P') for c in p)
            and not p.startswith('(')]


def align(reference, hypothesis):
    """Levenshtein token alignment, preserving deletions and insertions."""
    cost = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)
    cost[:, 0] = np.arange(len(reference) + 1)
    cost[0] = np.arange(len(hypothesis) + 1)
    for i, a in enumerate(reference, 1):
        for j, b in enumerate(hypothesis, 1):
            cost[i, j] = min(cost[i-1, j-1] + (a != b), cost[i-1, j] + 1, cost[i, j-1] + 1)
    result, i, j = [], len(reference), len(hypothesis)
    while i or j:
        if i and j and cost[i, j] == cost[i-1, j-1] + (reference[i-1] != hypothesis[j-1]):
            result.append((i-1, j-1)); i -= 1; j -= 1
        elif i and cost[i, j] == cost[i-1, j] + 1:
            result.append((i-1, None)); i -= 1
        else:
            result.append((None, j-1)); j -= 1
    return list(reversed(result))


def score(manifest, output, device):
    from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
    from huggingface_hub import hf_hub_download
    from .convert import sha256
    torch.set_num_threads(2)
    extractor = AutoFeatureExtractor.from_pretrained(RECOGNIZER, revision=REVISION)
    vocab_path = hf_hub_download(RECOGNIZER, 'vocab.json', revision=REVISION)
    vocabulary = json.loads(Path(vocab_path).read_text())
    tokens = {v: k for k, v in vocabulary.items()}
    model = Wav2Vec2ForCTC.from_pretrained(RECOGNIZER, revision=REVISION).to(device).eval()
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as stream:
        for index, row in enumerate(rows, 1):
            audio = read_audio(row['audio_path'], 16000)
            batch = extractor(audio, sampling_rate=16000, return_tensors='pt')
            with torch.inference_mode():
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits[0]
            predicted = logits.argmax(-1).cpu().tolist()
            phones, spans = [], []
            cursor = 0
            for token, frames in itertools.groupby(predicted):
                length = len(list(frames))
                if token != model.config.pad_token_id:
                    phones.append(tokens[token])
                    spans.append([cursor * .02, (cursor + length) * .02])
                cursor += length
            expected = normalize(row['expected_phones'])
            aligned = align(expected, phones)
            rhotics = [dict(reference_index=i, expected=expected[i],
                            recognized=phones[j] if j is not None else '<deleted>',
                            span=spans[j] if j is not None else None)
                       for i, j in aligned if i is not None and expected[i] in {'r', 'rʲ', 'l', 'ɹ'}]
            result = dict(id=row['id'], condition=row['condition'], expected=expected,
                          recognized=phones, recognized_spans=spans, rhotics=rhotics,
                          audio_sha256=sha256(Path(row['audio_path'])),
                          recognizer=RECOGNIZER, recognizer_revision=REVISION)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n'); stream.flush()
            print(index, '/', len(rows), row['condition'], row['id'],
                  [(r['expected'], r['recognized']) for r in rhotics], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    score(args.manifest, args.output, args.device)


if __name__ == '__main__':
    main()
