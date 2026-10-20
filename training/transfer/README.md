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
environment. Default output is `training/exp/phonetic_transfer_20k`; runtime logs
and monitoring records are under `training/quality_runs/phonetic_transfer/run_20000`.
For direct invocations, `training.transfer.train --steps 20000 --batch-size 8`
sets the same iteration budget. `--external-metrics` is intended for use under
the numeric-only W&B supervisor.

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
