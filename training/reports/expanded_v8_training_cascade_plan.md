# V8 Training-Cascade Fine-tune

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Objective

Make a new training corpus from the clean boundary-trimmed input for the
current 74,156-file non-VOA corpus. Train JETS for 100,000 new optimizer steps
from the v7 validation-mel best checkpoint.

## Audio Process

Use this fixed order:

1. Use the existing boundary-trimmed mono 24 kHz input.
2. Apply ClearerVoice `MossFormer2_SE_48K`.
3. Apply Sidon. Sidon has an internal 50 Hz input filter.
4. Apply light de-essing.
5. Apply FFmpeg de-clicking.
6. Apply FFmpeg peak limiting.
7. Apply default DeepFilterNet3. Do not use its optional post-filter or an
   attenuation limit.
8. Apply 85 percent Xiph RNNoise and 15 percent DeepFilterNet3 input.
9. Apply one linear gain to match source integrated loudness. Limit the gain
   at -0.1 dBFS.
10. Restore the exact source sample count and write a physical mono 24 kHz PCM
    24-bit WAV file atomically.

Do not use compression, dynamic loudness normalization, VAD removal, or a
second boundary trim.

## Speaker Embeddings

Keep the existing exact 50/50 assignment. Reuse 37,078 vectors that use audio
before processing. Calculate 37,078 vectors from the final v8 audio. Use both
RTX 3090 cards for the clean-vector calculation.

## Model

Use v7 epoch 93 as the initial checkpoint. Its validation mel loss is 33.045.
Its SHA-256 value is
`bec6bb4587a16250e39ce554cb6f2a6563f4d67fc825fc711a2c07436560129a`.
Load model weights only. Make new optimizer, scheduler, epoch, and step states.
Use two GPUs and FP32.

## Gates and Monitoring

Do not start full training until dataset validation, the exact 50/50 embedding
check, statistics, 100-step smoke training, and smoke inference have PASS
status. Stop new work if free disk space is less than 30 GiB.

Write preprocessing and training records every 15 minutes. Write an end-to-end
pipeline status at intervals of not more than 30 minutes. Each record includes
GPU use, power, VRAM, temperature, free disk space, progress, and a Kyiv ETA.
Preserve 25K, 50K, 75K, and 100K model files.

The preprocessing uses four persistent workers on each GPU. This setting
processed approximately 2.7 files per second in calibration. A test with eight
workers on each GPU processed only 1.1 to 1.6 files per second because of model
contention. Thus, the pipeline uses the faster eight-worker total.
