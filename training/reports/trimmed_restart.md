# Silence-trimmed training restart

## Objective

Remove leading and trailing silence from the model audio copies. Keep the raw
audio unchanged. Start a new JETS experiment from random initialization. Use both
RTX 3090 GPUs for full training.

## Fixed trim configuration

| Parameter | Value |
|---|---:|
| Detector | Relative frame RMS, version 1 |
| Threshold | 40 dB below maximum frame RMS |
| Frame length | 1024 samples |
| Hop length | 256 samples |
| Boundary padding | 100 ms |
| Sample rate | 24 kHz |

The trim operation does not use VAD. It does not apply denoise, normalization,
compression, de-essing, or mastering.

## Smoke result

- Audio preparation: PASS for 320 files.
- Files with applied trimming: 320.
- Mean removed duration: 2.647 s.
- Total duration before trimming: 0.474 h.
- Total duration after trimming: 0.238 h.
- Corrupt files: 0.
- Dataset validation: PASS.
- Tokenization: PASS with 0.0% OOV.
- Statistics: PASS.
- Training: PASS for 100 iterations.
- Train/validation generator loss: 121.344/89.308.
- Peak cached VRAM: 9.803 GiB.
- Inference: PASS for 32/32 WAV files.
- WAV clipping warnings: 0.

## Full-corpus result

- Audio files: 6,787.
- Files with applied trimming: 6,787.
- Total duration before trimming: 10.343 h.
- Total duration after trimming: 5.565 h.
- Total removed duration: 4.778 h.
- Mean removed duration: 2.534 s.
- Median removed duration: 2.539 s.
- Maximum removed duration: 9.324 s.
- Corrupt files: 0.
- Split sizes: 6,461 train, 146 development, and 180 evaluation files.
- Split leakage: 0.
- Tokenization OOV rate: 0.0%.
- Token list: 261 lines.
- Pitch, energy, and speech statistics: PASS.

## Dual-GPU calibration result

| Batch bins | cuDNN benchmark | Result | Peak cached VRAM | Elapsed time |
|---:|---|---|---:|---:|
| 4,000,000 | true | STOP | Not recorded | Stopped before checkpoint |
| 4,000,000 | false | PASS | 20.535 GiB | 385 s |
| 4,200,000 | false | PASS | 21.859 GiB | 404 s |

The first run took 8--11 seconds per batch. It did not improve useful GPU power.
The operator stopped the run before it saved a checkpoint.

The 4,200,000 run used up to about 22.85 GiB of device memory on GPU 0. It left
less than 10 percent free device memory. It did not give a useful power gain.

The long run started with `batch_bins: 4000000` and `cudnn_benchmark: false`.
Both GPUs reached 100 percent compute use. During epoch 4, one GPU used 23.19
GiB and had only 935 MiB free. The operator stopped the run after the 3k
checkpoint. The same experiment resumed with `batch_bins: 3800000`. This change
keeps the model and optimizer state but restores a safer memory reserve. Do not
use a checkpoint from the untrimmed experiment for this restart.
