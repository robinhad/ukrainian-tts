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
./espnet_recipe/run.sh --stage 8 --stop_stage 8
```

Do not run full-corpus preparation or long training unless every critical gate in
`reports/scale_readiness.md` is `PASS`.

## Local inference

```bash
python -m training.inference.synthesize \
  --text "Український синтез мовлення працює офлайн." \
  --output eval/generated/example.wav \
  --config exp/<experiment>/config.yaml \
  --checkpoint exp/<experiment>/<checkpoint>.pth
```

The raw WAV remains separate from any publication mastering.
