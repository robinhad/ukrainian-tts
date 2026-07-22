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

## Data understanding and preparation

The local environment is bootstrapped from pinned ESPnet and eSpeak-ng revisions.
The Ukrainian frontend uses eSpeak-ng 1.52.0 with language-data SHA-256
`924ed10e1c4f6f41ac603ceb3204d1b20877d75a9f8e71d392d7fdc98aa80d6d`.
The 50-case regression snapshot covers ordinary text, stress-sensitive words,
apostrophes, hyphens, numbers, dates, time, abbreviations, names, toponyms,
Latin/mixed text and complex punctuation. On 2026-07-22 all 18 frontend and
configuration tests passed in 0.80 seconds with no empty token sequences.

The pinned Lada Parquet was materialized deterministically into 256 train, 32 dev
and 32 eval utterances. All 320 audio/text pairs exist and are unique, totalling
0.474 hours. Durations range from 3.36 to 10.38 seconds with a 5.4 second median.
Model copies are mono PCM WAV at 24 kHz and received no signal processing. QC marked
28 source recordings as potentially clipped; the flag is retained in the manifest.
ESPnet validated every generated data directory without dropping an utterance.

## Modeling

Stages 5 and 6 produced a 153-entry phoneme token list (0.0% OOV) plus speech,
text, pitch and energy statistics. A CUDA dry run constructed the complete FP32
JETS model (83.31M trainable parameters), both AdamW optimizers and all extractors.
The first launch exposed that ESPnet's GAN trainer does not support gradient
accumulation; `accum_grad` was changed from 2 to the supported value 1. The repeated
run completed 100 FP32 iterations plus validation without NaN or OOM. Peak cached
VRAM was 5.938 GiB. A real 1-epoch checkpoint was saved with SHA-256
`6092bf244c08cf89971deb653c760f61c5a56bba63eb4dafafb36ed88c02685c`.

## Evaluation

ESPnet inference generated all 32 fixed smoke-eval utterances. Independent QC found
32 valid mono 24 kHz finite, non-empty and non-zero WAVs, no output clipping, and
durations from 1.653 to 4.096 seconds. Median GPU real-time factor was 0.0185. This
is a contour validation after only 100 iterations; no perceptual-quality claim is
made.

## Deployment

The local `python -m training.inference.synthesize` entrypoint loaded the same
frontend, token list, config and checkpoint and generated a 2.955-second raw WAV at
24 kHz with metadata JSON and RTF 0.112. It performs no publication mastering.
