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
the expanded suite had 19 passing tests in 5.10 seconds.

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

Post-1k full training uses both RTX 3090 cards. ESPnet uses single-node DDP. Before
the launch, the resource gate runs a CUDA matrix operation on both pinned GPU UUIDs.
The first DDP attempt failed before the first batch. The generated activation script
reset `CUDA_VISIBLE_DEVICES` to GPU0. The template now keeps an explicit multi-GPU
selection. The retry initialized both NCCL ranks. The first 50 batches took
approximately 0.92 seconds per batch. One GPU took approximately 1.50 seconds per
batch. NCCL cannot use direct P2P on this host. It uses shared-memory transport.
This condition can decrease performance, but it does not stop training.

## Evaluation and deployment

The 1k checkpoint made all 180 fixed eval utterances in 15 seconds. Independent
validation accepted each output. Each output is finite, non-empty, mono, and 24
kHz. There were no clipping warnings. The durations are 1.184 to 10.037 seconds.
The median RTF is 0.00893.

The local entry point also made a 3.477-second WAV with an RTF of 0.103. The first
test from a new shell found that the entry point did not find the pinned eSpeak
environment. The frontend now finds the repository-local runtime. It rejects a
different version or data hash. Release packaging and perceptual checkpoint
selection are pending.
