# Expanded-v4 Trim-Only Fine-Tune

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

## Purpose

This run must reduce metallic and hoarse sound artifacts. It must isolate the effect of audio processing.

## Data

The run uses the accepted expanded-v3 utterance IDs. It excludes 1,357 clipped files and 5 files that are shorter than 2 seconds. It excludes files that are longer than 20 seconds. The expected corpus has 207,505 utterances and 467.127 hours.

The training audio is the existing 24 kHz mono PCM WAV after one boundary-silence trim. The pipeline does not apply DeepFilterNet, a high-pass filter, de-essing, compression, or loudness normalization.

The speaker embedding set has an exact 50/50 split. One half uses original audio before the boundary trim. The other half uses the clean training WAV after the boundary trim. FFmpeg decodes compressed original files in memory. The pipeline does not make a second audio corpus.

## Model

The run initializes the JETS model from `best_checkpoints/367epoch.pth`. The checkpoint SHA-256 is `b7f093cd5c437f9b0122a13c48018c33b154698bb8149d26544401e99126d617`.

The run loads generator and discriminator weights. It does not load old speech, pitch, or energy normalization buffers. It creates new optimizer, scheduler, epoch, and step states. It uses the same 138-token list as epoch 367.

The target is 100,000 new steps. The run uses two NVIDIA RTX 3090 GPUs, FP32, 2,000,000 batch bins, eight data workers, and no gradient accumulation.

## Commands

Prepare and test a small corpus:

```bash
GPU_UUIDS=GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184 \
  training/scripts/prepare_expanded_v4_trim_only.sh smoke
GPU_UUIDS=GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184 \
  training/scripts/run_expanded_v4_trim_only_smoke.sh
```

Prepare the full corpus and start the run:

```bash
GPU_UUIDS=GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184 \
  training/scripts/prepare_expanded_v4_trim_only.sh full
GPU_UUIDS=GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184 \
  training/scripts/launch_expanded_v4_trim_only_training.sh 100000
```

## Safety

The run must stop if the free disk space is 30 GiB or less. After the raw embedding extraction passes, a separate monitor can delete only named source-cache directories. The canonical training WAV corpus and the expanded-v3 500K experiment are not cleanup targets.

The status monitor writes step count, loss metrics, GPU memory, GPU utilization, GPU power, temperature, free disk space, and the Kyiv ETA every 15 minutes.

## Smoke Result

The smoke preparation gates passed. The embedding extraction made 160 pre-trim vectors and 160 post-trim vectors. The 100-step dual-GPU fine-tune completed without NaN. It created 32 valid 24 kHz mono WAV files. The automatic validator reported no clipping warnings and a median real-time factor of 0.013.
