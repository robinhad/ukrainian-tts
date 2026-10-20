# English → Ukrainian JETS phonetic transfer

This is a separate weights-only transfer path. It does not resume the Ukrainian
478K run, change its listening page, or launch training during conversion/tests.
The inspected Ukrainian checkpoint has a 100×256 text embedding and 549 tensors;
its 24 kHz multispeaker recipe is incompatible with the English donor's settings.

The approved donor is [imdanboy/jets](https://huggingface.co/imdanboy/jets), trained
on English LJSpeech, revision `1db95c26516c44e6789bf06417c51e89400b190b` (model card:
CC BY 4.0). It has 78 ARPAbet tokens, a 256-dimensional embedding, and a
single-speaker architecture. The converted model has 100 independent Ukrainian
rows. All **538 other tensors remain bit-for-bit identical**. Audio stays at
**22,050 Hz, FFT 1024, hop 256, 80 mel bins**, including the donor's separate
feature/loss mel settings, pitch and energy extraction, normalization statistics,
vocoder and discriminator. Do not use the existing 24 kHz recipe to train this bundle.
A mixed-speaker dataset can be supplied, but this donor has no speaker conditioning.

The initial mapping contains **41 equivalent-sound, 32 PHOIBLE-nearest,
13 semantic special/punctuation, and 14 random rows**. See the complete
[mapping JSON](../reports/phonetic_transfer_mapping.json),
[mapping CSV](../reports/phonetic_transfer_mapping.csv), and
[GPU smoke results](../reports/phonetic_transfer_smoke.json).

## Frontend and mapping

Both preparation and inference use `TransferFrontend`, wrapping the repository's
pinned eSpeak-ng Ukrainian frontend. A phone is a whole eSpeak token: `t̪`, `tʃ`,
`iː`, `sʲ`, and `ˈiː` are never split into characters. Stress-marked vowels retain
separate IDs. Punctuation is separate. Written combining acute accents are
handled explicitly because eSpeak otherwise ignores them: `за́мок` and `замо́к`
produce different stressed vowel phones. Ambiguous written-to-phone vowel
alignment fails instead of silently moving stress. Unmarked text retains
eSpeak's predicted stress; pronunciation accuracy is not guaranteed.

`ukrainian_inventory.yaml` preserves the existing checkpoint's 100-token order,
including historical foreign-phone and language-marker tokens. Preparation
rejects unseen phones instead of substituting `<unk>`. For a different corpus,
expand the inventory before conversion and regenerate the bundle and prepared
splits. Inventory, frontend source hashes, eSpeak version and language-data hash
are stored in `frontend.json` and verified on preparation, training and inference.
Changing frontend code also requires regenerating that snapshot and the data.

English ARPAbet uses broad American IPA (`R→ɹ`, `CH→tʃ`, `AH0→ə`, `AH1→ˈʌ`,
`ER0→ɚ`). Stress 0/1/2 becomes none/primary/secondary. ARPAbet does not encode
vowel length, so length is not invented to create exact matches. Every donor
conversion is listed under `source_inventory` in the JSON report.

Exact base IPA is preferred, then matching stress and donor token order.
Otherwise the converter uses [PHOIBLE v2.0](https://github.com/phoible/dev/tree/862bec9af5db42e3c9ceedeaa378bf4c6fa0ec8b)
articulatory features. It requires matching syllabicity, ≥90% known-feature
coverage, and ≤0.15 normalized categorical mismatch. Tone and stress are
excluded from feature distance; stress breaks ties and mismatches are reported.
IPA normalization only removes tie bars, normalizes Unicode, and canonicalizes
`g/ɡ`; it does not erase length, palatalization or stress. Static features are an
initialization heuristic, not acoustic equivalence: for example `r→L` is the
nearest row under this metric, and monophthong/diphthong features may tie.
Approximate matches are always labeled `phoible`, even at zero feature distance.

The table is pinned by SHA-256, downloaded rather than vendored. PHOIBLE's
[upstream documentation and attribution](https://github.com/phoible/dev/tree/862bec9af5db42e3c9ceedeaa378bf4c6fa0ec8b)
and [repository license](https://github.com/phoible/dev/blob/862bec9af5db42e3c9ceedeaa378bf4c6fa0ec8b/LICENSE)
remain authoritative for those inputs. The mapping code is independent of its
upstream scripts. Missing features or unreliable matches use deterministic
random normal rows with the donor embedding's standard deviation. No guessed
articulatory vectors are substituted.

Special IDs copy by meaning. Available punctuation copies directly; quotation
variants share the donor's quotation/apostrophe initialization, and ellipsis
uses its `..` row. Unsupported punctuation is random and reported. Even when
two tokens copy one English row, they occupy separate trainable storage and
can diverge during fine-tuning.

## Reproduce conversion

Use the existing environment from `training/scripts/bootstrap_env.sh`, including
the Ukrainian ESPnet tokenizer patch. From the repository root:

```bash
source training/activate.sh
TRANSFER_ROOT=training/quality_runs/phonetic_transfer
python -m training.scripts.verify_local_frontend --output "$TRANSFER_ROOT/frontend"
export UKTTS_ESPEAK_DATA_HASH_FILE="$TRANSFER_ROOT/frontend/ESPEAK_NG_DATA_HASH"
python -m training.transfer.download --output "$TRANSFER_ROOT"
DONOR_DIR="$TRANSFER_ROOT/donor/exp/tts_train_jets_raw_phn_tacotron_g2p_en_no_space"
python -m training.transfer.convert \
  --source-config "$DONOR_DIR/config.yaml" \
  --source-model "$DONOR_DIR/train.total_count.ave_5best.pth" \
  --source-root "$TRANSFER_ROOT/donor" \
  --target-config training/transfer/ukrainian_inventory.yaml \
  --features "$TRANSFER_ROOT/phoible-segments-features.tsv" \
  --output "$TRANSFER_ROOT/converted"
```

Choose a new output directory on subsequent conversions. The bundle includes
`model.pth`, complete ESPnet `config.yaml`, normalization files, `tokens.txt`,
`frontend.json`, and mapping JSON/CSV. Inputs and downloads are recorded in
`inputs.lock.json`; custom donor files and features are also hashed in the report.
`--max-distance` and `--seed` control mapping; `--lr-factor` defaults to 0.1.

The published donor's YAML advertises scale-discriminator weight normalization
that was not applied in the old implementation. Conversion detects plain weights
and disables those config flags to preserve the **actual donor architecture**;
this is the [ESPnet #5240 compatibility case](https://github.com/espnet/espnet/pull/5240).
It is resolved before optimizers are built, without changing any weight tensor.

## Prepare pairs and configure fine-tuning

Supply separate train/held-out JSONL files, one object per line:

```json
{"id":"example-001","text":"Привіт, Україно!","audio_path":"data/example-001.wav"}
```

Existing V12 parquet manifests are accepted directly through `--pairs`, using
their `utterance_id`, `text_sanitized`, and `audio_path` columns. Keep the existing
`quality_v12_train.parquet` / `quality_v12_dev.parquet` split.
Preparation phonemizes text, mixes channels to mono if necessary, and resamples
only to the donor's 22.05 kHz. It does not rerun enhancement, trimming, or loudness
normalization. WAVs use floating-point storage to avoid an extra quantization step.

```bash
python -m training.transfer.prepare --bundle "$TRANSFER_ROOT/converted" \
  --pairs train.jsonl --output "$TRANSFER_ROOT/train"
python -m training.transfer.prepare --bundle "$TRANSFER_ROOT/converted" \
  --pairs valid.jsonl --output "$TRANSFER_ROOT/valid"
# Writes a reviewable config; does not train.
python -m training.transfer.train --bundle "$TRANSFER_ROOT/converted" \
  --train "$TRANSFER_ROOT/train" --valid "$TRANSFER_ROOT/valid" \
  --output "$TRANSFER_ROOT/config_review" --write-config-only
```

Training consumes space-separated phones with `g2p: null`, `cleaner: null`;
inference applies the same frontend and sends the resulting IDs directly to
JETS. Never send raw Ukrainian text to this bundle's generic ESPnet preprocessor.
Prepared text/manifest checksums and frontend fingerprints are checked before
training. Relative normalization paths in the bundle are resolved by the wrappers.

Full-model fine-tuning is configured with **fresh AdamW optimizers at 2e-5 for
both generator and discriminator**, all parameters trainable, no resume, and
strict weights-only initialization. Donor AdamW settings and exponential scheduler
are retained, with a fresh scheduler clock. Default limits are 50 epochs of 1000
iterations, batch size 4; review these before a production run. Validation uses
actual JETS `generator_loss`/`generator_g_mel_loss` metrics. W&B defaults to
`ukrainian-tts`, metrics only (`wandb_model_log_interval: -1`).

The following command is provided for a later authorized run; **it was not run**:

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=64G \
  python -m training.transfer.train --bundle "$TRANSFER_ROOT/converted" \
  --train "$TRANSFER_ROOT/train" --valid "$TRANSFER_ROOT/valid" \
  --output training/exp/english_ukrainian_transfer
```

Use a fresh output directory. This entrypoint intentionally refuses to resume an
existing optimizer. A later continuation should be designed separately from
initial transfer.

### Authorized 20K-step run

The SLURM launcher prepares the existing V12 ≥3.5 non-VOA train/dev manifests,
then fine-tunes for 20,000 generator/discriminator iterations (20 × 1000).
It uses batch size 8, fresh optimizers at 2e-5, and keeps three best checkpoints.
It sends only numeric TensorBoard metrics to the `ukrainian-tts` W&B project
through the existing metrics supervisor; native ESPnet W&B uploads are disabled.
Resource telemetry and progress/ETA in Europe/Kyiv are recorded every 60 seconds.
The memory guard stops the process if the configured unified-memory reserve is
breached; monitoring reports errors but does not silently restart a failed run.

Submit **from the repository root**:

```bash
sbatch --job-name=jets-phonetic-transfer-20k --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=2-00:00:00 \
  --output=training/quality_runs/phonetic_transfer/slurm-%j.log \
  training/scripts/run_phonetic_transfer.sh
```

`STEPS`, `BATCH_SIZE`, `TRANSFER_ROOT`, and `TTS_EXP` can be supplied through the
environment. Default output is `training/exp/phonetic_transfer_20000`; runtime logs
and monitoring records are under `training/quality_runs/phonetic_transfer/run_20000`.
For direct invocations, `training.transfer.train --steps 20000 --batch-size 8`
sets the same iteration budget. `--external-metrics` is intended for use under
the numeric-only W&B supervisor.

### Single-voice replacement: Tetiana

The mixed-speaker attempt was stopped at the user's request. Its replacement
starts again from the converted English checkpoint with fresh optimizers for
the full 20,000 steps. No mixed-speaker training updates are carried forward.

The selected voice is [Tetiana](https://huggingface.co/datasets/speech-uk/opentts-tetiana),
the named speaker with the greatest retained duration in the current V12 train
manifest. Selection requires **both** `speaker_id=tetiana` and
`source_id=opentts_tetiana`; a dataset label or an anonymous per-file identifier
is not treated as proof of a shared speaker.

| Split | Recordings | Duration |
|---|---:|---:|
| Train | 553 | 39.47 minutes |
| Validation | 13 | 0.91 minutes |
| Existing evaluation split, excluded from training | 37 | 2.60 minutes |

The existing processing and SigMOS ≥3.5 filter are retained. The selector keeps
the original train/dev split, rejects overlapping IDs/audio hashes, and reuses
the already resampled audio. Single-voice metadata is checked again at training
startup. This is a small adaptation set; evaluate the held-out voice samples
before judging the final checkpoint.

```bash
sbatch --job-name=jets-tetiana-transfer-20k --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=2-00:00:00 \
  --export=ALL,SPEAKER=tetiana,SOURCE_ID=opentts_tetiana,STEPS=20000 \
  --output=training/quality_runs/phonetic_transfer/slurm-tetiana-%j.log \
  training/scripts/run_phonetic_transfer.sh
```

This writes to `training/exp/phonetic_transfer_tetiana_20000`, with separate
`run_tetiana_20000` monitoring and `wandb_tetiana_20000` numeric-metrics state.
The reproducible subset command is `python -m training.transfer.select_voice`;
its `selection.json` records the selected speaker, source, counts, and hours.
`RUN_TAG` can override the log/metrics namespace for a new experiment.

## FlashAttention training and checkpoint resume

Install FlashAttention-2 after the normal training bootstrap. The installer
defaults to the official
[prebuilt 2.8.3 wheel, release v2.8.3.post1](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post1)
for Linux ARM64, Python 3.12, Torch 2.9, CUDA 13 and CXX11 ABI, pinned by SHA256.
It checks this environment and installs without changing Torch or dependencies.
Other environments must select a matching wheel or explicitly opt into a
source build (`FLASH_ATTN_BUILD_SOURCE=1`), which requires CUDA/Ninja and builds
upstream 2.8.3.post2.

```bash
bash training/scripts/install_flash_attention.sh
source training/activate.sh
export UKTTS_ESPEAK_DATA_HASH_FILE=training/quality_runs/phonetic_transfer/frontend/ESPEAK_NG_DATA_HASH
srun --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 \
  env UKTTS_TEST_FLASH=1 python -m pytest training/tests/test_transfer_attention.py -q
srun --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 \
  python -m training.transfer.smoke --bundle "$TRANSFER_ROOT/converted" \
  --pairs smoke_pair.jsonl --device cuda --flash-attention \
  --output "$TRANSFER_ROOT/flash_smoke.json"
```

Set `FLASH_ATTENTION=1` on the SLURM launcher for new runs. For an existing
experiment, stop its current job after a complete epoch checkpoint has been
saved, then submit with the same experiment, voice, step target, and metrics
namespace, plus `RESUME=1,FLASH_ATTENTION=1`. Never run two trainers against
the same output directory.

```bash
sbatch --job-name=jets-tetiana-transfer-flash --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=2-00:00:00 \
  --export=ALL,SPEAKER=tetiana,SOURCE_ID=opentts_tetiana,STEPS=20000,RESUME=1,FLASH_ATTENTION=1 \
  --output=training/quality_runs/phonetic_transfer/slurm-tetiana-flash-%j.log \
  training/scripts/run_phonetic_transfer.sh
```

`--steps` remains the **total** target. Resume restores the model, both
optimizers, both schedulers and the step/epoch reporter from `checkpoint.pth`;
it fails if state is absent or data/training settings change. The original
`finetune.yaml` is preserved, and resume writes `finetune-resume.yaml`.
The same W&B metrics stream and minute-by-minute Kyiv ETA monitor continue.

The adapter replaces computation in the donor's eight plain transformer
attention modules without changing parameter names or shapes. Only Q/K/V inputs
to the fused operation use BF16; projections, model parameters, optimizer state
and GAN losses stay FP32. Training retains 0.2 attention dropout. Padding masks
remove keys only: padded query outputs are retained because subsequent
convolutions can consume them. Inference uses zero attention dropout. CUDA
errors fail the run instead of silently reverting to unfused attention.
Each module logs its first successful FlashAttention call.

The CUDA tests compare outputs and gradients with ESPnet attention, including
padded query positions, check parameter identity, exercise training dropout,
and require actual FlashAttention forward/backward kernels in a CUDA profile.
The full-model smoke additionally checks both GAN backward passes and synthesis.
Weights remain loadable by the ordinary CPU/GPU inference path; add
`--flash-attention --device cuda` to `training.transfer.infer` to use the same
accelerated attention path. Both paths retain the identical Ukrainian frontend.

Validation on 2026-10-20: the official ARM64 wheel passed both attention CUDA
tests (padded/unpadded outputs and gradients, dropout, profiled forward/backward
kernels), both resume/ETA tests, and the 20 existing transfer/frontend/metrics
tests. The full JETS smoke passed both GAN backward passes and inference; its
sanitized results are in
[`phonetic_transfer_flash_smoke.json`](../reports/phonetic_transfer_flash_smoke.json).
The Tetiana run resumed at step 1,000 with both Adam counters at 1,000 and both
scheduler epochs at 1. All eight attention modules confirmed successful
training calls with dropout 0.2.

Early observed throughput was 0.8955 seconds/step over resumed steps
1,011–1,100, versus 0.913 seconds/step for the initial 1,000-step epoch:
about 2% faster. This is an end-to-end observation across different batches,
not a controlled attention benchmark; the earlier epoch also overlapped CPU
compilation. FlashAttention does not accelerate the entire GAN pipeline.

## Smoke tests and inference

```bash
python -m pytest training/tests/test_phonetic_transfer.py -q
srun --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:15:00 \
  python -m training.transfer.smoke --bundle "$TRANSFER_ROOT/converted" \
  --pairs smoke_pair.jsonl --device cuda --output "$TRANSFER_ROOT/smoke_result.json"
python -m training.transfer.infer --bundle "$TRANSFER_ROOT/converted" \
  --text 'Привіт, Україно!' --device cuda --output "$TRANSFER_ROOT/example.wav"
```

Validation on 2026-10-20 passed 15 unit/frontend tests, all 550 existing
frontend regression cases, and the GPU smoke test. The generated fine-tuning
configuration also passed the actual ESPnet argument parser. No training was launched.

The smoke pair should be a real, transcribed Ukrainian recording of a few
seconds. The test verifies every loaded tensor, the trainer's initialization
loader, the real prepared-data iterator, identical training/inference token IDs, explicit
stress, fresh optimizer
state and coverage of every parameter, finite GAN losses/gradients, nonzero text
embedding and backbone gradients, and a finite synthesized waveform. It calls
backward for both GAN halves, **never optimizer.step()**, and verifies that all
parameters remain unchanged. The unit test also updates one copied row in a toy
embedding to prove rows are independent. This is not an audio-quality evaluation.

For later trained weights, use the same bundle with
`--checkpoint training/exp/english_ukrainian_transfer/50epoch.pth` (adjust epoch).
The wrapper strictly loads the entire generator/discriminator state. An English
model with initialized Ukrainian rows is not yet a Ukrainian speech model;
intelligibility must be evaluated after fine-tuning.

## Single-voice listening page

The transfer checkpoint uses a different rate and frontend from the older
multi-speaker run. Generate its listening set with the transfer entrypoint:

```bash
source training/activate.sh
TRANSFER_ROOT=training/quality_runs/phonetic_transfer
export UKTTS_ESPEAK_DATA_HASH_FILE="$TRANSFER_ROOT/frontend/ESPEAK_NG_DATA_HASH"
# Use a normal GPU allocation when idle, or share your running training
# allocation briefly by supplying its ID in TRAINING_JOB_ID.
srun --jobid="$TRAINING_JOB_ID" --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=2 --mem=16G \
  python -m training.transfer.listening \
  --bundle "$TRANSFER_ROOT/converted" \
  --checkpoint training/exp/phonetic_transfer_tetiana_20000/latest.pth \
  --voice-data "$TRANSFER_ROOT/voices/tetiana" \
  --evaluation training/data/quality_v12/manifests/quality_v12_eval.parquet \
  --originals training/data/quality_v10_raw/frontend/quality_v10_eval.parquet \
  --output "$TRANSFER_ROOT/voices/tetiana"
```

This pins the resolved epoch checkpoint, selects ten evaluation texts with seed
777, checks source/speaker identity and exclusion from train/dev, verifies audio
hashes, and uses the training frontend and FlashAttention inference path.
Outputs are placed in `listening_<step>/`: `preview.html` embeds all audio for
standalone listening, `index.html` uses relative WAV paths, and
`listening_set.zip` contains the complete set. Existing outputs are protected
against accidental mixing. Audio, references, selection, hashes, per-file
SigMOS/Audiobox PQ and medians are included. Quality scoring is report-only;
samples are never selected by their score. The generated waveform is not
enhanced, trimmed, or normalized. No reference speaker embedding is used.

The initial 5,000-step page has median synthesized SigMOS 3.209, versus 3.470
for the same ten original references and 3.504 for their processed references.
These are model-predicted scores, not human listening judgments.

## Recalculate statistics for the selected voice

The initial transfer used the donor's normalization. The Tetiana continuation
uses statistics recalculated from all 553 prepared **training** recordings
(39.469 minutes, 204,253 frames); development and evaluation are excluded.
Mel, continuous log-F0 and energy come from ESPnet's actual `collect_feats`
path, with the same 22,050 Hz audio, FFT, hop, mel range and pitch settings.
Moments are accumulated in float64, weighted by real frames, excluding padding.
This changes normalization, not audio files or extraction settings.

| Feature | Donor mean | Tetiana mean | Donor std | Tetiana std |
|---|---:|---:|---:|---:|
| Continuous log-F0 | 5.293362 | 5.169609 | 0.246970 | 0.258810 |
| Energy | 21.824111 | 12.286066 | 19.813511 | 12.421743 |

All 80 mel means/stds, counts, input hashes and extraction provenance are in
[`phonetic_transfer_tetiana_statistics.json`](../reports/phonetic_transfer_tetiana_statistics.json).
These are feature units, not Hz or dB loudness. Changing normalization during
fine-tuning changes the targets seen by learned predictors; quality improvement
must be measured, not assumed.

```bash
source training/activate.sh
TRANSFER_ROOT=training/quality_runs/phonetic_transfer
export UKTTS_ESPEAK_DATA_HASH_FILE="$TRANSFER_ROOT/frontend/ESPEAK_NG_DATA_HASH"
srun --jobid="$TRAINING_JOB_ID" --overlap --ntasks=1 --cpus-per-task=2 \
  python -m training.transfer.statistics collect \
  --bundle "$TRANSFER_ROOT/converted" \
  --voice-data "$TRANSFER_ROOT/voices/tetiana" \
  --output "$TRANSFER_ROOT/converted_tetiana_stats" --device cuda
```

Use a new bundle directory. After stopping the trainer at a completed epoch
checkpoint, explicitly migrate its state into a new experiment directory:

```bash
python -m training.transfer.statistics migrate \
  --source training/exp/phonetic_transfer_tetiana_20000 \
  --bundle "$TRANSFER_ROOT/converted_tetiana_stats" \
  --output training/exp/phonetic_transfer_tetiana_stats_20000

export TRANSFER_BUNDLE="$TRANSFER_ROOT/converted_tetiana_stats"
export TTS_EXP=training/exp/phonetic_transfer_tetiana_stats_20000
export SPEAKER=tetiana SOURCE_ID=opentts_tetiana
export RUN_TAG=tetiana_20000 STEPS=20000 BATCH_SIZE=8
export RESUME=1 FLASH_ATTENTION=1
sbatch --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=48:00:00 \
  --output="$TRANSFER_ROOT/slurm-tetiana-stats-%j.log" \
  training/scripts/run_phonetic_transfer.sh
```

Migration replaces exactly six normalization buffers in the checkpoint.
Learned weights, both optimizer states, schedulers and accumulated training
step count remain intact. Previous validation history is archived under
`valid_previous_normalization`; best-checkpoint ranking and averaging use only
the new phase. The transfer launcher handles sparse validation phases in the
ESPnet reporter. Original checkpoints and their bundle remain unchanged.
Scalar history continues in the same metrics-only W&B run.

Applied on 2026-10-20 at step 6,000. An independent recursive comparison
confirmed 533 unchanged model tensors, exactly six changed buffers, identical
optimizer/scaler/scheduler state, Adam counters of 6,000 and scheduler epochs
of 6. The audit is in
[`phonetic_transfer_tetiana_statistics_migration.json`](../reports/phonetic_transfer_tetiana_statistics_migration.json).
The migrated checkpoint passed strict loading, the prepared-data iterator,
shared frontend IDs, finite generator/discriminator backward passes and finite
synthesis with FlashAttention:
[`phonetic_transfer_tetiana_stats_6000_smoke.json`](../reports/phonetic_transfer_tetiana_stats_6000_smoke.json).
Smoke-test optimizers are temporary and take zero steps; the training
continuation restores the original optimizer states. A numeric
`train/normalization_phase=1` marker at step 6,000 marks the change in W&B.

On 2026-10-20, a further 30,000 steps were scheduled after successful completion
of the 20,000-step run, for **50,000 total steps**. The continuation uses the
same Tetiana data, recalculated statistics bundle, experiment directory,
FlashAttention, batch size and metrics-only W&B stream. It restores the final
trainer checkpoint, including both optimizer states and scheduler clocks.
Keep `RUN_TAG=tetiana_20000` and the existing `TTS_EXP` despite the larger target;
those names identify the ongoing run. With the exports above, submit:

```bash
STEPS=50000 sbatch --dependency="afterok:$CURRENT_TRAINING_JOB_ID" \
  --kill-on-invalid-dep=yes --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
  --mem=96G --time=48:00:00 \
  --output="$TRANSFER_ROOT/slurm-tetiana-to50k-%j.log" \
  training/scripts/run_phonetic_transfer.sh
```

The dependency prevents two trainers from writing to the same experiment.
The 50K resume configuration was checked against the saved 18K state: both Adam
counters were 18,000, both scheduler epochs were 18, and normalization matched
the recalculated bundle. At approximately 0.893 seconds/step, the additional
30K steps take about 7 hours 27 minutes, excluding startup and checkpoint overhead.

For inference, listening-page generation and smoke tests, pair new checkpoints
with `converted_tetiana_stats`; use `converted` for pre-migration checkpoints.
Strict loading and resume reject mismatched statistics, allowing only FP32
roundoff from published averaged checkpoints. Training and inference keep the
same verified Ukrainian phoneme frontend.

```bash
python -m pytest training/tests/test_transfer_statistics.py -q
srun --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:15:00 \
  python -m training.transfer.smoke \
  --bundle "$TRANSFER_ROOT/converted_tetiana_stats" \
  --checkpoint "$TTS_EXP/latest.pth" \
  --pairs "$TRANSFER_ROOT/smoke_pair.jsonl" \
  --device cuda --flash-attention --output "$TRANSFER_ROOT/statistics_smoke.json"
```

## Rhotic diagnostic: transfer confuses r and l

The 2026-10-21 investigation used the fixed ten-item Tetiana listening panel
(seven sentences contain 16 `r` positions), plus twelve fixed training items
containing 24 `r` positions. Independent, unconstrained phoneme recognition used
[Facebook Wav2Vec2 XLSR phoneme recognition](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
at revision `2c733782da5604684829819a5eb744c193fe9398`. Audio was resampled to
16 kHz for this recognizer. Greedy output phones were aligned to the same
expected phone sequence in every condition; no transcript-constrained decoder
or language model was used. These labels are diagnostic evidence, not a human
pronunciation score. Full counts, hashes and per-file phone outputs are in
[`phonetic_transfer_rhotic_diagnostic.json`](../reports/phonetic_transfer_rhotic_diagnostic.json)
and the adjacent `phonetic_transfer_rhotic_{history,interventions,training}.jsonl`.

| Held-out condition | Recognized r | Recognized l | Other/deleted |
|---|---:|---:|---:|
| Original recordings | 16 | 0 | 0 |
| Processed recordings | 16 | 0 | 0 |
| Ukrainian model trained from scratch, 478K | 16 | 0 | 0 |
| Transfer, 43K | 2 | 11 | 3 |
| Transfer, input r IDs replaced with l | 2 | 11 | 3 |
| Scratch model, input r IDs replaced with l | 0 | 12 | 4 |
| Transfer, r/l embedding difference multiplied by 8 | 4 | 11 | 1 |
| Transfer, predicted r duration doubled | 3 | 8 | 5 |

The mapping initialized **both `r` and `l` from the same donor `L` vector**.
The rows have separate storage, but began numerically identical. At 43K their
cosine similarity is still **0.999245** (L2 distance 0.822); the scratch model's
similarity is -0.029460 (distance 25.465). Replacing input `r` with `l` in the
transfer model preserves the recognizer's outcome at **all 16 individual r
positions**. The same intervention in the scratch model changes its correct
`r` pronunciation. This identifies weak learned r/l distinction as the current
failure mechanism, consistent with the initialization collision. The relative
roles of initialization, learning rate and training data would require a
controlled retraining comparison to separate causally.

The problem is present before the 6K normalization migration: the 5K transfer
outputs contain zero recognized `r`, fifteen `l` and one deletion. It also
occurs on training items: references contain 24/24 recognized `r`, ordinary
transfer synthesis 10/24, and synthesis conditioned on reference-derived
alignment durations, pitch and energy 12/24. Thus source preprocessing and
predicted duration alone do not explain the failure. Median predicted `r`
duration is 69.66 ms for transfer and 64 ms for the scratch model on this panel.

Simply replacing the trained r vector with the donor-initialized English `R`
row is not a fix: thirteen of sixteen positions become English-like `ɹ`.
Embedding extrapolation is also not a validated fix. All interventions were
in memory on a pinned 43K checkpoint; no optimizer steps or model saves were
performed. Twenty-one no-r control waveforms remained exactly unchanged.
The running training job and frontend were not modified.

Reproduce in a GPU allocation (or use `srun --jobid=... --overlap --ntasks=1`
to share the current allocation). Pin a copy of the checkpoint outside the
trainer's pruning directory first. The diagnostic root contains a local
`baseline_manifest.jsonl` with `id`, `condition`, `audio_path`, and
`expected_phones` from the saved listening selections. Runtime paths and audio
remain untracked.

```bash
python -m training.transfer.rhotic_diagnostic --manifest "$DIAGNOSTIC_ROOT/baseline_manifest.jsonl" \
  --output "$DIAGNOSTIC_ROOT/baseline_scores.jsonl"
python -m training.transfer.rhotic_probes \
  --bundle "$TRANSFER_ROOT/converted_tetiana_stats" \
  --checkpoint "$DIAGNOSTIC_ROOT/43epoch.pth" \
  --panel "$TRANSFER_ROOT/voices/tetiana/listening_42000" \
  --output "$DIAGNOSTIC_ROOT/probes" \
  --old-config training/exp_quality_v12/tts_jets_quality_v12_50k/config.yaml \
  --old-checkpoint training/exp_quality_v12/tts_jets_quality_v12_50k/478epoch.pth \
  --xvectors training/dump_quality_v12/xvector/quality_v12_eval/xvector.scp
python -m training.transfer.rhotic_diagnostic --manifest "$DIAGNOSTIC_ROOT/probes/manifest.jsonl" \
  --output "$DIAGNOSTIC_ROOT/probes/scores.jsonl"
```

For training-set probes, use a new output directory and add
`--train-data "$TRANSFER_ROOT/voices/tetiana/train"`, omitting the three old-model
arguments. Score that manifest into `train_probes/scores.jsonl`.
`python -m training.transfer.rhotic_report --help` describes the final aggregation;
it writes `index.html` and standalone `preview.html` with 56 listening players.
The tests in `training/tests/test_rhotic_diagnostic.py` check phone boundaries,
rhotic distinctions, insertions/deletions and r/l count separation.
