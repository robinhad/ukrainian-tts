# Training steps

Use these steps for the Ukrainian JETS pipeline. Run the commands from the
repository root. Do not use this procedure for VITS or Tacotron2.

## 1. Prepare the local environment

The bootstrap command installs the dependencies in `training/.venv` and
`training/vendor`. It does not install a system Python package.

```sh
training/scripts/bootstrap_env.sh
source training/activate.sh
python training/scripts/validate_espeak.py
```

## 2. Run the frontend tests

```sh
pytest -q training/tests
```

## 3. Run the smoke pipeline

This command uses GPU0. It prepares the smoke data, trains JETS, and makes the
smoke WAV files.

```sh
training/scripts/run_smoke_test.sh
```

Do not start full training if a critical gate in
`training/reports/scale_readiness.md` is not `PASS`.

## 4. Run full training

The full-training command uses both pinned RTX 3090 cards. The command checks
both cards, available RAM, and available disk space before it starts.

```sh
training/scripts/run_full_training.sh 25000
```

Use the same command with `50000` or `100000` only after you evaluate the fixed
evaluation set for the preceding milestone.

## 5. Make the milestone WAV files

```sh
training/scripts/run_milestone_inference.sh latest.pth
```

## 6. Make one local WAV file

The output is a raw mono 24 kHz WAV file. This command does not apply mastering.

```sh
training/.venv/bin/python -m training.inference.synthesize \
  --text "Український синтез мовлення працює офлайн." \
  --output training/eval/generated/example.wav \
  --config training/exp_full/tts_jets_uk_24k_full/config.yaml \
  --checkpoint training/exp_full/tts_jets_uk_24k_full/latest.pth
```
