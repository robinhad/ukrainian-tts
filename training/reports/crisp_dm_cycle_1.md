# CRISP-DM cycle 1

## Business understanding

The MVP proves that the complete Ukrainian single-speaker JETS pipeline operates.
It does not prove perceptual quality. The fixed architecture has these components:

- ESPnet2 GAN-TTS with JETS.
- Raw mono audio at 24 kHz.
- Ukrainian eSpeak-ng phonemes with stress and punctuation.
- No separate vocoder, verbalizer, or stress model.

The MVP is successful when the pipeline meets these conditions:

- The frontend makes deterministic and non-empty phonemes.
- The ESPnet data directories are valid.
- The pipeline makes token, pitch, and energy statistics.
- The generator and the discriminator update without an error.
- The training stage saves a real checkpoint.
- The checkpoint makes a finite and non-empty mono WAV at 24 kHz.

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

The local environment uses pinned ESPnet and eSpeak-ng revisions.
The Ukrainian frontend uses eSpeak-ng 1.52.0 with language-data SHA-256
`924ed10e1c4f6f41ac603ceb3204d1b20877d75a9f8e71d392d7fdc98aa80d6d`.
The 50-case regression snapshot includes ordinary text and stress-sensitive words.
It also includes apostrophes, hyphens, numbers, dates, times, abbreviations, names,
toponyms, Latin text, mixed text, and complex punctuation. On 2026-07-22, all 18
frontend and configuration tests passed in 0.80 seconds. No token sequence was empty.

The data stage made 256 train, 32 dev, and 32 eval utterances from the pinned Lada
Parquet file. All 320 audio and text pairs exist and are unique. Their total duration
is 0.474 hours. The durations are 3.36 to 10.38 seconds. The median is 5.4 seconds.
The model copies are mono PCM WAV files at 24 kHz. The data stage did not apply
signal processing. QC marked 28 source recordings as possibly clipped. The manifest
keeps this flag. ESPnet accepted all the utterances in each data directory.

## Modeling

Stages 5 and 6 made a phoneme token list with 153 entries and 0.0% OOV. The stages
also made the speech, text, pitch, and energy statistics. A CUDA dry run constructed
the FP32 JETS model with 83.31 million trainable parameters. It also constructed
the two AdamW optimizers and all the extractors. The first launch showed that the
ESPnet GAN trainer does not support gradient accumulation. The configuration now
uses the supported value `accum_grad: 1`. The repeated run completed 100 FP32
iterations and validation. It had no NaN or OOM error. Peak cached VRAM was 5.938
GiB. The run saved a real 1-epoch checkpoint with SHA-256
`6092bf244c08cf89971deb653c760f61c5a56bba63eb4dafafb36ed88c02685c`.

## Evaluation

ESPnet inference made all 32 fixed smoke-eval utterances. Independent QC accepted
all 32 WAV files. Each file is finite, non-empty, non-zero, mono, and 24 kHz. QC did
not find output clipping. The durations are 1.653 to 4.096 seconds. The median GPU
real-time factor is 0.0185. This test only validates the pipeline after 100
iterations. It does not validate perceptual quality.

## Deployment

The local `python -m training.inference.synthesize` entry point used the same
frontend, token list, configuration, and checkpoint. It made a 2.955-second raw WAV
at 24 kHz. It also made the metadata JSON. The RTF was 0.112. The entry point did
not apply publication mastering.
