# CRISP-DM cycle 1

## Business understanding

The MVP proves the complete Ukrainian single-speaker JETS contour, not perceptual
quality. The fixed architecture is ESPnet2 GAN-TTS/JETS, raw 24 kHz mono waveform,
eSpeak-ng Ukrainian phonemes with stress and punctuation, and no separate vocoder,
verbalizer or stress model.

Success requires deterministic non-empty phonemes, valid ESPnet data directories,
token and pitch/energy statistics, stable generator/discriminator updates, a real
checkpoint, and a finite non-empty 24 kHz mono WAV produced from that checkpoint.

## Constraints and risks

- Dataset: pinned Lada single-speaker corpus; smoke subset only before MVP PASS.
- GPU: one RTX 3090 selected by UUID; second GPU is not used in cycle 1.
- Initial training is FP32 with 15% VRAM reserve and no AMP.
- JETS alignment can fail on noisy text/audio pairs or very small batches.
- eSpeak/ESPnet version or language-data drift invalidates frontend snapshots.

## Environment observed before execution

- Ubuntu 22.04, Python 3.10.12, 24 logical CPUs.
- 125 GiB RAM total; 122 GiB available during preflight.
- 336 GiB free disk during preflight.
- 2x RTX 3090, driver 590.48.01; both idle with about 24.1 GiB free VRAM.
- CUDA toolkit 13.2 is installed; the isolated PyTorch runtime uses CUDA 12.8 wheels.

## Data understanding, preparation, modeling, evaluation and deployment

These sections are updated only from actual artifacts and commands. Current status:
implementation prepared; environment bootstrap and data materialization not yet run.
