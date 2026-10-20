"""Inference using exactly the frontend and ID inventory used for preparation."""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from .convert import load_strict
from .frontend import TransferFrontend


def synthesize(bundle, text, checkpoint=None, device='cpu', flash_attention=False):
    # TTSTask's inference wrapper allows non-strict loads; explicitly use the
    # GAN training task and strict state loading instead, then JETS inference.
    model, args = load_strict(bundle, device, checkpoint)
    if flash_attention:
        from .attention import enable_flash_attention
        enable_flash_attention(model)
    frontend = TransferFrontend(args.token_list)
    frontend.verify(bundle / 'frontend.json')
    _, ids = frontend.encode(text)
    model.eval()
    with torch.no_grad():
        result = model.tts.inference(text=torch.tensor(ids, dtype=torch.long, device=device))
    return result['wav'].detach().cpu().numpy(), args.feats_extract_conf['fs']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--flash-attention', action='store_true')
    args = parser.parse_args()
    waveform, sample_rate = synthesize(args.bundle, args.text, args.checkpoint, args.device,
                                      args.flash_attention)
    sf.write(args.output, waveform, sample_rate)


if __name__ == '__main__':
    main()
