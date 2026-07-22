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
  --output eval/generated/example.wav \
  --config exp/<experiment>/config.yaml \
  --checkpoint exp/<experiment>/<checkpoint>.pth
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

## Get the training status

Use this command to get the progress, the Europe/Kyiv ETA, the checkpoint list,
the error count, and the status of each GPU. The GPU status includes power draw,
power limit, temperature, memory use, and compute use.

```bash
python scripts/training_status.py \
  --log exp_full/tts_jets_uk_24k_full/train.log \
  --output reports/training_status.jsonl
```
