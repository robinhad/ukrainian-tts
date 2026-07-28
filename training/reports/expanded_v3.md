# Expanded-v3 implementation

## Objective

Build a fresh multi-source Ukrainian JETS model. Use clean training audio. Stop
at 25,000 optimizer steps for a listening decision.

## Source control

The source registry uses a default-deny policy. It excludes YODAS data,
MLCommons data, VoxLingua data, ASMR data, Common Voice derivatives, GPL audio,
non-commercial data, and test-only data.

The registry permits direct Common Voice 26 as a specific source. It does not
permit a Common Voice derivative. The pipeline needs the direct archive before
the full run.

The user gave a commercial training grant for `robinhad/VOA-ukr`. The pipeline
does not permit raw redistribution. The inspected rows use a dash as the
transcript. Therefore, the pipeline treats this source as unlabeled.

Per-file sources need a permitted license and a license evidence URL for each
record. The permitted set is CC0, CC BY, CC BY-SA, or public domain. LibriVox
also needs a public-domain text check for the target jurisdiction.

## Unlabeled audio

UA-SER supplies human transcripts and speaker IDs. The pipeline does not send
UA-SER through ASR.

For unlabeled sources, the pipeline uses the pinned NVIDIA Sortformer model for
diarization. Feature extraction uses source chunks of no more than 60 seconds.
The process keeps source-relative time values. It keeps only intervals with one
active speaker. It merges same-speaker intervals across a silent gap of no more
than 0.5 seconds. It splits intervals longer than 20 seconds at a local
low-energy point.

The pipeline uses the pinned NVIDIA Parakeet model for speech recognition. It
keeps Ukrainian hypotheses with a mean confidence of at least 0.50. Ukrainian
letters must be at least 80 percent of all letters. A pseudo-labeled record can
only enter the training split.

The process writes a progress record after each source file. A restart skips
each completed source file. The collector also uses an existing batch when its
report, record count, shard range, and audio paths are complete. Version
`v4-c050-d20-g050-defer-long` uses new markers and outputs. It cannot reuse
pseudo-label completion markers from the 0.80-confidence run.

The pipeline saves segments longer than 20 seconds in `deferred_too_long`.
It does not add these segments to the training manifest. A later process can
split or review these files.

## Clean audio

The pipeline removes boundary silence. It then applies DeepFilterNet3, a
high-pass filter, light de-essing, gentle compression, and two-pass EBU R128
normalization. The source file does not change.

## Speaker embeddings

The pipeline assigns one deterministic audio variant to each record. Half of
all embeddings use source audio. The other half use clean audio. The difference
for each speaker stratum is no more than one record.

The smoke run made 159 raw vectors and 159 clean vectors. Each vector has 192
values.

## Smoke result

| Item | Result |
|---|---:|
| Source records before final audio QC | 360 |
| Clean retained records | 318 |
| Train records | 268 |
| Development records | 19 |
| Evaluation records | 31 |
| Source groups in two splits | 0 |
| Token-list lines | 55 |
| Token OOV rate | 0 percent |
| JETS parameters | 83.33 million |
| Training iterations | 100 |
| Valid inference WAV files | 31 of 31 |
| Median inference RTF | 0.0142 |

The smoke checkpoint is a real ESPnet checkpoint. The smoke run used both RTX
3090 cards. One sample showed 226 W on GPU 0 and 217 W on GPU 1. No NaN, OOM,
or critical runtime error occurred.

The smoke result proves pipeline operation. It does not prove good sound
quality.

## Full-run status

The completed 25K run is technically valid, but it does not meet the new VOA
duration gate. The manifest has 76,578 records and 95.260 hours.
The train, development, and evaluation splits have 73,755, 1,403, and 1,420
records.

The manifest contains only 2.981 VOA hours. The required minimum is 800 hours.
The old pseudo-label process accepted 2,947 of 1,349,403 candidate outcomes.
It rejected 757,829 candidates at the uncalibrated 0.80 confidence threshold.
Do not use the current manifest as the full large-corpus training set.

The hybrid embedding set has 38,289 source-audio vectors and 38,289 clean-audio
vectors. Each vector has 192 values. The token list has 110 lines. The pitch
and energy statistics exist.

The dual-GPU JETS run completed 25,000 iterations. It ended with exit code 0.
No NaN or OOM occurred. The final validation generator loss is 60.947. The
final validation mel loss is 44.106. Peak cached GPU memory is 21.381 GiB.

The five-voice evaluation made five valid 24 kHz mono WAV files. All five files
use the required Kamianets-Podilskyi sentence. Automatic checks have `PASS`.
The human listening check is open.

## Disk control

The warning threshold is 80 GiB. The cleanup trigger is 60 GiB during data
preparation. The process keeps completed source batches when free space is more
than 60 GiB. At 60 GiB, it removes the oldest completed batch until free space
is more than 60 GiB. If no completed batch is available, the preparation stops.
The training stop threshold is 60 GiB. The training monitor checks this
threshold at intervals of no more than five minutes.
