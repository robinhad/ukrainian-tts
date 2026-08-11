# V7 De-click and Peak-limit Fine-tune

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Objective

Make a new training corpus from the completed v6 enhanced audio. Apply the
final de-click and peak-limit stage. Keep the v6 corpus unchanged. Train JETS
for 100,000 new optimizer steps from the v6 validation-mel best checkpoint.

## Audio process

Use the 74,156 v6 Sidon and de-essed WAV files as the source. Keep the source
sample rate and channel count. Use 32-bit planar floating-point audio in the
filters. Use this FFmpeg filter graph:

```text
aformat=sample_fmts=fltp,adeclick=w=55:o=75:a=2:t=4:b=2,alimiter=limit=0.891251:attack=5:release=80:level=false:latency=true
```

Write physical PCM 24-bit WAV files in a new v7 directory. Use atomic output
replacement. Do not change a v6 source file.

## Speaker embeddings

Keep the existing exact 50/50 assignment. Reuse 37,078 vectors that use audio
before processing. Calculate 37,078 vectors from the final v7 audio. Use both
RTX 3090 cards for the clean-vector calculation.

## Model

Use this initial checkpoint:

```text
training/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k/best_checkpoints/94epoch.pth
```

Its SHA-256 is
`3a8f3f40d00abfcaadc3f457a7485ddcc816cbb8a26d0cec193eeae39c7d7d48`.
The v6 validation mel loss at epoch 94 is `34.084`. Load model weights only.
Make new optimizer, scheduler, epoch, and step states. Use two GPUs and FP32.

## Gates and monitoring

Do not start full training until dataset validation, the exact 50/50 embedding
check, statistics, 100-step smoke training, and smoke inference have PASS
status. Stop new work if free disk space is less than 30 GiB.

Write a pipeline status at intervals of not more than 30 minutes. Include the
phase, progress, GPU power, VRAM, temperature, free disk space, and Kyiv ETA.
During full training, write metrics and ETA at intervals of not more than 15
minutes. Preserve 25K, 50K, 75K, and 100K model files.
