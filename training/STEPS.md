# Training steps

Use these steps for the Ukrainian JETS pipeline. Run the commands from the
repository root. Do not use this procedure for VITS or Tacotron2.

## Expanded-v3 controlled run

Do not start the expanded-v3 25K run after only a smoke PASS. First, prepare all
enabled sources and create the full readiness report.

```sh
export MDC_COMMON_VOICE_ROOT=/path/to/direct/common-voice/uk
training/scripts/bootstrap_nemo_env.sh
training/scripts/prepare_expanded_v3.sh full
jq .status training/reports/expanded_v3_full_scale_readiness.json
training/scripts/launch_expanded_v3_training.sh
```

The required status is `PASS`. The launcher uses two GPUs. It saves 1K, 5K,
15K, and 25K milestones. It then makes five listening voices with the fixed
Kamianets-Podilskyi sentence.

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

For the silence-trimmed restart, use this command:

```sh
training/scripts/run_trimmed_smoke_test.sh
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

For a new run that uses trimmed audio, use these commands:

```sh
training/scripts/prepare_trimmed_full.sh
BATCH_BINS=4000000 training/scripts/run_trimmed_training.sh 25000
```

## 5. Make the milestone WAV files

```sh
training/scripts/run_milestone_inference.sh latest.pth
```

For the silence-trimmed restart, run this command only after 25k training exits
with status 0:

```sh
training/scripts/finalize_trimmed_training.sh 25epoch.pth 25k
```

This command keeps the checkpoint and validates all fixed evaluation WAV files.

## 6. Make one local WAV file

The output is a raw mono 24 kHz WAV file. This command does not apply mastering.

```sh
training/.venv/bin/python -m training.inference.synthesize \
  --text "Український синтез мовлення працює офлайн." \
  --output training/eval/generated/example.wav \
  --config training/exp_full/tts_jets_uk_24k_full/config.yaml \
  --checkpoint training/exp_full/tts_jets_uk_24k_full/latest.pth
```

## 7. Make the listening set

Use the same 20 utterances for each candidate. Do not select a release checkpoint
from validation loss alone.

```sh
training/.venv/bin/python training/scripts/build_listening_set.py \
  --manifest training/data/full/manifests/eval.parquet \
  --raw-dir training/data/full/raw \
  --candidate jets_15k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_15k/eval/wav \
  --candidate jets_23k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_23k/eval/wav \
  --candidate jets_25k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_25k/eval/wav \
  --count 20 \
  --output training/eval/generated/listening_25k
```
