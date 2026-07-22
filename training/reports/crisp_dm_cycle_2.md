# CRISP-DM cycle 2

## Business understanding

The production-oriented objective is stable Ukrainian pronunciation in the Lada
single-speaker voice, reproducible local inference, and progressive checkpoints
that can be selected by fixed-set listening and inference stability rather than
training loss alone. The fixed JETS/eSpeak-ng/24 kHz architecture is unchanged.

## Data understanding

The pinned source contains 6962 rows. Preparation selected 6787 utterances after
excluding 174 duplicate sanitized texts and one duration outlier. The resulting
corpus contains 10.343 hours with durations from 2.64 to 11.22 seconds (median
5.4 seconds). The group-aware split is 6461 train, 146 dev, and 180 eval. Because
the source exposes no document ID, contiguous blocks of 50 numeric source files
are treated as related groups; no source group or exact audio/text hash crosses a
split. This approximation is a documented residual leakage risk.

All model copies are mono PCM 24 kHz WAV without mastering or normalization. There
are 366 clipping QC flags. Detailed duration/text/token/character/ratio coverage is
stored in `reports/full_data_analysis.json`. The frontend observed 259 tokens, 104
of them occurring at most ten times.

## Data preparation

ESPnet validated all train/dev/eval directories. Full tokenization reported 0.0%
OOV and full speech, pitch and energy statistics were aggregated. The deterministic
frontend snapshot now contains 500 corpus and curated sentences; all 18 automated
tests passed in 5.06 seconds.

## Modeling

FP32 batch calibration used 200 real iterations at every point:

| batch_bins | Peak cached VRAM | Train time | Result |
|---:|---:|---:|---|
| 1,000,000 | 6.027 GiB | 2m04s | PASS |
| 2,000,000 | 17.266 GiB | 3m40s | PASS |
| 2,500,000 | 12.535 GiB | 4m22s | PASS; non-monotonic grouping |
| 3,000,000 | 20.727 GiB | 5m07s | PASS; about 12% VRAM reserve |

The selected value is `batch_bins=3,000,000`. A subsequent 200-iteration AMP run
was finite and faster, but its validation generator loss was 140.497 versus 77.744
for FP32 at the same calibrated batch size. AMP is therefore disabled for the long
run. The resumable runner checkpoints every 1000 iterations.

## Evaluation and deployment

Full-corpus milestone evaluation begins after the 1k checkpoint and reuses the same
180-utterance eval split. Release packaging remains pending checkpoint evaluation.
