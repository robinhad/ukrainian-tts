# Ukrainian single-speaker JETS training

This directory contains the reproducible ESPnet2 GAN-TTS pipeline. The pipeline is
separate from the legacy multi-speaker inference code. It does not use a verbalizer,
a separate stress model, denoising, compression, or mastering.

The project documents use ASD-STE100 Simplified Technical English style. No
approved STE checker has certified these documents.

## Fixed versions

- ESPnet `v.202604-patch1`, commit `cff0a07`
- eSpeak-ng `1.52.0`, commit `4870adf`
- PyTorch/torchaudio `2.9.1`, CUDA 12.8 wheels
- Dataset `speech-uk/opentts-lada`, revision
  `729289b58251da4a21ce85f8808dd908f28b0d7f`

## Prepare the environment

```bash
cd training
./scripts/bootstrap_env.sh
source ./activate.sh
python scripts/validate_espeak.py --write-hash
python -m training.frontend.snapshot \
  --cases tests/frontend/regression.tsv \
  --snapshot tests/frontend/expected_phonemes.json \
  --cache data/phonemes.sqlite3 --update
```

The bootstrap installs files only in `training/.venv` and `training/vendor`. It
applies `patches/espnet-espeak-ng-ukrainian.patch` to the pinned ESPnet source.

## Audio trimming

Audio preparation removes leading and trailing silence from each model copy. It
does not change the raw OGG file. The detector uses relative frame RMS with these
fixed values:

- Threshold: 40 dB below the maximum frame RMS.
- Frame length: 1024 samples.
- Hop length: 256 samples.
- Safety padding: 100 ms at each active boundary.

The manifest records the original duration, the removed duration, both trim
boundaries, and the trim configuration hash. Trimming does not use VAD,
normalization, denoise, compression, or mastering.

## Smoke cycle

The smoke cycle uses GPU0. The code selects GPU0 by its UUID. `run_logged.py`
records the exit status for each long command. It prints a heartbeat at intervals
of five minutes or less. The heartbeat contains the Europe/Kyiv time and the ETA.

```bash
cd training
source ./activate.sh
./scripts/run_smoke_test.sh
```

Use these commands to resume an ESPnet stage:

```bash
./espnet_recipe/run.sh --stage 1 --stop_stage 6
./espnet_recipe/run.sh --stage 7 --stop_stage 7 --train_args "--max_epoch 1"
./espnet_recipe/run.sh --stage 8 --stop_stage 8 \
  --train_args "--max_epoch 1" \
  --inference_model train.total_count.ave.pth
```

Do not run full-corpus preparation or long training unless every critical gate in
`reports/scale_readiness.md` is `PASS`.

## Local inference

The entry point finds the pinned eSpeak-ng installation in this repository. You
can run it from a new shell after the bootstrap. Source `activate.sh` before you
run a recipe command.

```bash
training/.venv/bin/python -m training.inference.synthesize \
  --text "Український синтез мовлення працює офлайн." \
  --output training/eval/generated/example.wav \
  --config training/exp_full/tts_jets_uk_24k_full/config.yaml \
  --checkpoint training/exp_full/milestones/25k.pth
```

Keep the raw WAV separate from the publication WAV.

## Full-corpus training after MVP PASS

Calibration selected `batch_bins=3000000`. This value kept approximately 12% of
the VRAM free for each process. The 200-iteration AMP test increased the validation
loss. Thus, full training uses FP32. Smoke tests use one RTX 3090. Full training
uses both RTX 3090 cards. Use these commands to resume the same experiment:

```bash
cd training
./scripts/run_full_training.sh 1000
./scripts/run_full_training.sh 25000
./scripts/run_milestone_inference.sh latest.pth
./scripts/run_full_training.sh 50000
./scripts/run_full_training.sh 100000
```

Use the same command for the 200k and 400k targets. Continue only if the fixed-set
listening results and the inference diagnostics improve.

## Restart with trimmed audio

These commands keep the first full-corpus experiment. They make new data,
statistics, and checkpoints with the `trimmed` suffix.

```bash
training/scripts/run_trimmed_smoke_test.sh
training/scripts/prepare_trimmed_full.sh
BATCH_BINS=3800000 training/scripts/run_trimmed_training.sh 25000
training/scripts/finalize_trimmed_training.sh 25epoch.pth 25k
```

Calibrate `BATCH_BINS` before a long run. Use both GPUs. Increase the value until
each GPU uses approximately 85--90% of its VRAM. Keep at least 10% free VRAM. Do
not increase the GPU power limit and do not overclock the GPUs. Keep
`cudnn_benchmark=false` for this variable-length workload. The benchmark mode
made the calibration slower.

The finalization command keeps the 25k checkpoint. It runs inference on all 180
fixed evaluation items. It validates each WAV and makes one local inference
example and one 20-item listening set. It also summarizes sampled GPU power,
utilization, temperature, and device memory. Run this command only after the
25k training command exits with status 0.

The short calibration selected 4,000,000 batch bins. The complete data sequence
later used 23.19 GiB on one GPU and left only 935 MiB free. The run resumed from
the 3k checkpoint with 3,800,000 batch bins. This value keeps a safer device
memory reserve for the long run.

## Make the listening set

This command makes links to 20 raw references and the 15k, 23k, and 25k outputs.
It does not copy the large audio files. Listen to the same utterance in each
directory before you select a release checkpoint.

```bash
training/.venv/bin/python training/scripts/build_listening_set.py \
  --manifest training/data/full/manifests/eval.parquet \
  --raw-dir training/data/full/raw \
  --candidate jets_15k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_15k/eval/wav \
  --candidate jets_23k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_23k/eval/wav \
  --candidate jets_25k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_25k/eval/wav \
  --count 20 \
  --output training/eval/generated/listening_25k
```

## Get the training status

Use this command to get the progress, the Europe/Kyiv ETA, the checkpoint list,
the error count, and the status of each GPU. The GPU status includes power draw,
power limit, temperature, memory use, and compute use.

```bash
python scripts/training_status.py \
  --log exp_full/tts_jets_uk_24k_full/train.log \
  --output reports/training_status.jsonl
```

## View the training metrics

ESPnet writes TensorBoard event files through PyTorch. It does not use the
TensorFlow training runtime. Use this command from the repository root:

```bash
training/.venv/bin/tensorboard \
  --logdir training/exp_full_trimmed/tts_jets_uk_24k_trimmed/tensorboard
```

The completed trimmed run has 29 train scalar tags and 16 validation scalar
tags through step 25,000. The validation mel loss fell from 56.198 at 1k to
33.549 at 25k.

## Train with Common Voice and Lada

This iteration uses Common Voice and Lada audio. It uses one 192-value ECAPA
embedding for each utterance. It uses both RTX 3090 GPUs for training.

The public Common Voice source does not contain stable client IDs. Therefore,
the data pipeline does not claim a common speaker identity for two Common Voice
files. The pipeline uses `lada` as the speaker ID for Lada data.

Run these commands from the repository root:

```bash
training/scripts/run_multispeaker_smoke_test.sh
training/scripts/prepare_multispeaker_full.sh
training/scripts/calibrate_multispeaker_batch.sh 4500000 200
training/scripts/run_multispeaker_training.sh 25000
training/scripts/finalize_multispeaker_training.sh 25epoch.pth 25k
```

The preparation command makes 24 kHz mono PCM WAV model copies. It removes only
leading and trailing silence. It uses a 40 dB relative frame-RMS threshold and
100 ms boundary padding. It does not use MFA, VAD, denoise, compression,
de-essing, loudness normalization, or mastering.

The preparation command uses the pinned Common Voice and Lada revisions. Set
`DMYTRO_MANIFEST` only when a valid Dmytro manifest and its audio files are
available. If this variable is not set, Dmytro is not in the train set.

The selected long-run value is `batch_bins=4500000`. Larger tested values did
not keep the required VRAM reserve. The training command uses FP32 and writes
TensorBoard metrics through PyTorch.

Use this command to run the external five-minute monitor:

```bash
training/scripts/monitor_multispeaker_training.sh \
  TRAIN_PID 25000 300
```

The monitor writes current progress, power, memory, temperature, and the Kyiv
ETA to `training/reports/training_status_multispeaker.jsonl`.

The finalization command starts only after the 25k checkpoint exists. It
validates all 1,677 fixed evaluation files for the 1k, 5k, and 25k
checkpoints. It also makes a listening set with 10 Common Voice references and
10 Lada references. This set contains all three model candidates. The command
makes one Lada example and one zero-shot Dmytro example. The zero-shot example
does not make Dmytro a trained speaker. It makes a release candidate with
model, frontend, statistics, reports, licenses, cards, and SHA-256 checksums.

Use this command to view the current model metrics:

```bash
training/.venv/bin/tensorboard \
  --logdir training/exp_multispeaker_full/tts_jets_uk_24k_multispeaker/tensorboard
```
