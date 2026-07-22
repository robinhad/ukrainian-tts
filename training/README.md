# Ukrainian single-speaker JETS training

This directory contains the reproducible ESPnet2 GAN-TTS pipeline. It is independent
from the repository's legacy multi-speaker inference code and deliberately performs
no verbalization, separate stress prediction, denoising, compression or mastering.

## Fixed versions

- ESPnet `v.202604-patch1`, commit `cff0a07`
- eSpeak-ng `1.52.0`, commit `4870adf`
- PyTorch/torchaudio `2.9.1`, CUDA 12.8 wheels
- Dataset `speech-uk/opentts-lada`, revision
  `729289b58251da4a21ce85f8808dd908f28b0d7f`

## Bootstrap

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

The bootstrap installs only into `training/.venv` and `training/vendor`. It applies
`patches/espnet-espeak-ng-ukrainian.patch` to the pinned local ESPnet checkout.

## Smoke cycle

GPU 0 is selected by UUID. Every long command is wrapped by `run_logged.py`, which
records the exit status and prints a heartbeat at least every five minutes.
Each heartbeat includes the current Europe/Kyiv time and an estimated completion time.

```bash
cd training
source ./activate.sh
./scripts/run_smoke_test.sh
```

The ESPnet stages can also be resumed independently:

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

The entrypoint discovers the pinned repository-local eSpeak-ng installation, so it
can be invoked from a fresh shell after bootstrap (sourcing `activate.sh` remains
recommended for all recipe commands):

```bash
training/.venv/bin/python -m training.inference.synthesize \
  --text "Український синтез мовлення працює офлайн." \
  --output eval/generated/example.wav \
  --config exp/<experiment>/config.yaml \
  --checkpoint exp/<experiment>/<checkpoint>.pth
```

The raw WAV remains separate from any publication mastering.

## Full-corpus training after MVP PASS

Calibration selected `batch_bins=3000000` with about 12% VRAM reserve. FP32 is kept
because the 200-iteration AMP check degraded validation loss. Training uses one RTX
3090 and resumes in the same experiment directory:

```bash
cd training
./scripts/run_full_training.sh 1000
./scripts/run_full_training.sh 25000
./scripts/run_milestone_inference.sh latest.pth
./scripts/run_full_training.sh 50000
./scripts/run_full_training.sh 100000
```

Further 200k/400k extensions use the same command and are allowed only while fixed
eval listening and inference diagnostics improve.
