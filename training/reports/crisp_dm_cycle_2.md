# CRISP-DM cycle 2

## Business understanding

The production objective is stable Ukrainian pronunciation in the Lada voice. The
pipeline must keep the voice of one speaker. Local inference must be reproducible.
The training stage must make progressive checkpoints. Select a checkpoint from the
fixed-set listening results and the inference stability. Do not select a checkpoint
from the training loss only. The fixed JETS, eSpeak-ng, and 24 kHz architecture does
not change.

## Data understanding

The pinned source contains 6962 rows. Data preparation selected 6787 utterances.
It removed 174 duplicate sanitized texts and one duration outlier. The selected
corpus contains 10.343 hours. The durations are 2.64 to 11.22 seconds. The median
is 5.4 seconds. The split contains 6461 train, 146 dev, and 180 eval utterances.

The source does not contain a document ID. Thus, the split procedure uses each
contiguous block of 50 numeric source files as one related group. A source group
does not occur in more than one split. An exact audio hash or text hash does not
occur in more than one split. The proxy groups can still cause residual leakage.

All model copies are mono PCM WAV files at 24 kHz. Data preparation did not apply
mastering or normalization. QC set 366 clipping flags. The frontend found 259
tokens. Of these tokens, 104 occur no more than ten times. See
`reports/full_data_analysis.json` for the detailed coverage data.

## Data preparation

ESPnet accepted all the train, dev, and eval directories. Full tokenization reported
0.0% OOV. The statistics stage made the speech, pitch, and energy statistics. The
deterministic frontend snapshot contains 500 corpus and curated sentences. The
first suite had 18 passing tests in 5.06 seconds. After the local entry-point fix,
the expanded suite had 19 passing tests in 5.10 seconds. The final suite had 21
passing tests in 5.18 seconds after the 25k evaluation.

## Modeling

FP32 batch calibration used 200 real iterations at every point:

| batch_bins | Peak cached VRAM | Train time | Result |
|---:|---:|---:|---|
| 1,000,000 | 6.027 GiB | 2m04s | PASS |
| 2,000,000 | 17.266 GiB | 3m40s | PASS |
| 2,500,000 | 12.535 GiB | 4m22s | PASS; non-monotonic grouping |
| 3,000,000 | 20.727 GiB | 5m07s | PASS; about 12% VRAM reserve |

The selected value is `batch_bins=3,000,000`. A 200-iteration AMP run was finite
and faster. Its validation generator loss was 140.497. The FP32 loss was 77.744 at
the same batch size. Thus, long training does not use AMP. The resumable runner
saves a checkpoint after each 1000 iterations.

The full-corpus FP32 sanity milestone completed 1000 iterations on GPU0. The wall
time was 25 minutes and 38 seconds. The trainer time was 25 minutes and 13 seconds.
The run saved `1epoch.pth` with SHA-256 `66e06306...5910f`. The train generator
loss was 82.403. The validation generator loss was 84.718. All generator,
discriminator, alignment, pitch, and energy losses were finite. Peak cached VRAM
was 20.727 GiB.

Post-1k full training used both RTX 3090 cards. ESPnet used single-node DDP. Before
the launch, the resource gate runs a CUDA matrix operation on both pinned GPU UUIDs.
The first DDP attempt failed before the first batch. The generated activation script
reset `CUDA_VISIBLE_DEVICES` to GPU0. The template now keeps an explicit multi-GPU
selection. The retry initialized both NCCL ranks. The first 50 batches took
approximately 0.92 seconds per batch. One GPU took approximately 1.50 seconds per
batch. NCCL cannot use direct P2P on this host. It uses shared-memory transport.
This condition can decrease performance, but it did not stop training.

The dual-GPU FP32 run resumed from 1k and completed 25,000 iterations. The command
ran for 25,316 seconds and exited with status 0. The final train and validation
generator losses were 61.883 and 71.475. All reported losses were finite. The peak
cached VRAM was 15.551 GiB. The 25k checkpoint SHA-256 is
`d1ee89bd...46ff4dd`.

The fixed validation set gave these generator losses for the retained candidates:

| Checkpoint | Validation generator loss | SHA-256 prefix | Result |
|---:|---:|---|---|
| 15k | 68.848 | `ee289c8a` | Retained |
| 17k | 70.752 | `ac0e7895` | Retained |
| 23k | 69.162 | `f8cd1c9d` | Retained |
| 25k | 71.475 | `d1ee89bd` | Retained target milestone |

Loss alone does not select the release checkpoint. A listening test must compare
the retained candidates.

The power monitor recorded 231 samples during the run. GPU0 had a mean power of
188.91 W, a maximum power of 235.28 W, and a maximum temperature of 86 C. GPU1
had a mean power of 212.18 W, a maximum power of 253.82 W, and a maximum
temperature of 78 C. Neither GPU reached the 95 C slowdown threshold.

## Evaluation and deployment

The 1k checkpoint made all 180 fixed eval utterances in 15 seconds. Independent
validation accepted each output. The 5k, 15k, 17k, 23k, and 25k checkpoints then
used the same fixed eval set. Each run made 180 WAV files and exited with status 0.
Independent validation accepted all 900 new files. Each file is finite, non-empty,
mono, and 24 kHz. No run had a clipping warning.

| Checkpoint | WAV result | Duration range | Median RTF |
|---:|---|---:|---:|
| 1k | 180/180 PASS | 1.184--10.037 s | 0.00893 |
| 5k | 180/180 PASS | 2.240--7.829 s | 0.00757 |
| 15k | 180/180 PASS | 2.016--7.360 s | 0.00788 |
| 17k | 180/180 PASS | 1.557--8.139 s | 0.00785 |
| 23k | 180/180 PASS | 1.600--8.309 s | 0.00770 |
| 25k | 180/180 PASS | 2.816--7.424 s | 0.00845 |

The 25k local entry point made a 4.757-second raw WAV. Its RTF was 0.0754 and its
peak absolute sample was 0.457. It also wrote the frontend, checkpoint, and config
metadata. The frontend uses the repository-local eSpeak runtime. It rejects a
different version or data hash. Release packaging and perceptual checkpoint
selection are pending.
