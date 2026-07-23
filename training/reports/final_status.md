# Final status

## Виконано

The reproducible Ukrainian JETS pipeline is on branch `autotrain`. All new
training code is under `training/`. The pipeline made silence-trimmed model
copies and kept all raw source files unchanged. The full run reached 25,000
iterations on two RTX 3090 GPUs. The command returned exit status 0.

The pipeline uses Ukrainian text, the Lada speaker, 24 kHz audio, ESPnet2 JETS,
GAN-TTS, phoneme tokens, and pinned eSpeak-ng 1.52.0. It does not use a separate
verbalizer, stress model, G2P model, vocoder, VAD, or forced aligner. It does not
apply denoise, normalization, compression, de-essing, or mastering to training
audio.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS for the silence-trimmed smoke subset.
- Data Preparation: PASS for 320 smoke files.
- Modeling: PASS for 100 iterations.
- Evaluation: PASS for 32 of 32 smoke-eval WAV files.
- Deployment: PASS for the local raw-WAV entry point.

## CRISP-DM cycle 2

- Business Understanding: PASS.
- Data Understanding: PASS for 6,787 selected utterances.
- Data Preparation: PASS for trimmed audio, split, token list, and statistics.
- Modeling: PASS for 25,000 FP32 iterations on two GPUs.
- Evaluation: PASS for 180 of 180 fixed-set WAV files.
- Deployment: PASS for local inference. Perceptual release selection is pending.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Frontend tests | PASS | Final suite has 28 passing tests |
| eSpeak-ng pin | PASS | Version 1.52.0 and the language-data hash are pinned |
| Non-empty phonemes | PASS | Regression and corpus checks found no empty sequence |
| ESPnet data directories | PASS | Train, development, and evaluation directories pass |
| Token list | PASS | Full tokenization has 0.0% OOV |
| Statistics | PASS | Speech, pitch, and energy statistics exist |
| JETS construction | PASS | JETS and both AdamW optimizers initialize |
| Stable training | PASS | 25,000 iterations; no NaN, OOM, or critical error |
| Checkpoint | PASS | The 25k checkpoint exists and has a verified SHA-256 |
| Inference WAV | PASS | 180 of 180 fixed-eval files pass |
| Reproduction commands | PASS | README and scripts contain the commands |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/run_trimmed_smoke_test.sh` | 0 | 100 iterations and 32 of 32 valid WAV files | `reports/smoke_trimmed_inference.json` |
| `training/scripts/prepare_trimmed_full.sh` | 0 | 6,787 files; 5.565 h after trim; 0.0% OOV; statistics complete | `reports/full_trimmed_dataset.json` |
| `BATCH_BINS=3800000 training/scripts/run_trimmed_training.sh 25000` | 0 | 25,000 iterations on two GPUs in 32,318 s | `exp_full_trimmed/tts_jets_uk_24k_trimmed/train.log` |
| `training/scripts/finalize_trimmed_training.sh 25epoch.pth 25k` | 0 | Checkpoint copy, 180-file inference, WAV checks, local example, and listening set | `reports/full_trimmed_inference_25k.json` |
| `source training/activate.sh && pytest -q training/tests` | 0 | 28 tests passed in 5.55 s | `tests/` |
| `python training/scripts/check_resources.py --mode full --require-torch --workspace training --output training/reports/resource_usage.jsonl` | 0 | 121.89 GiB RAM and 284.78 GiB disk free after the run | `reports/resource_usage.jsonl` |

The final train and validation generator losses are 47.795 and 47.112. The final
train and validation mel losses are 33.677 and 33.549. Peak cached VRAM is
22.178 GiB. Validation mel loss fell by 40.3 percent from 1k to 25k.

The GPU monitor collected 140 samples. GPU0 mean and maximum power were 197.11 W
and 264.08 W. GPU1 mean and maximum power were 221.82 W and 261.39 W. Both
cards reached 100 percent sampled utilization.

The TensorBoard audit found 29 train scalar tags and 16 validation scalar tags.
Both event streams reach step 25,000. The fixed-eval run accepted 180 of 180
mono 24 kHz files. It found no clipping warning. Duration is 1.013 to 5.376
seconds. Median RTF is 0.01338.

## Створені артефакти

- Trimmed manifests: `training/data/full_trimmed/manifests/`.
- ESPnet data directories: `training/espnet_recipe/data/{train,dev,eval}/`.
- JETS config: `training/espnet_recipe/conf/tuning/train_jets_uk_24k.yaml`.
- ESPnet patch: `training/patches/espnet-espeak-ng-ukrainian.patch`.
- Token list: `training/dump_full_trimmed/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `training/exp_full_trimmed/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Checkpoint: `training/exp_full_trimmed/milestones/25k.pth`.
- Checkpoint SHA-256: `58f4673676cd382d1ae2bc6c5a7a80e809ccce9e2b3dea42edef6cae177f9d75`.
- TensorBoard events: `training/exp_full_trimmed/tts_jets_uk_24k_trimmed/tensorboard/{train,valid}/`.
- Fixed-eval WAV files: `training/exp_full_trimmed/tts_jets_uk_24k_trimmed/decode_jets_milestone_25k/eval/wav/`.
- Evaluation report: `training/reports/full_trimmed_inference_25k.json`.
- Power report: `training/reports/power_summary_trimmed.json`.
- Local example: `training/eval/generated/example_trimmed_25k.wav` and adjacent JSON.
- Listening set: `training/eval/generated/listening_trimmed_25k/`.
- CRISP-DM reports: `training/reports/crisp_dm_cycle_1.md` and
  `training/reports/crisp_dm_cycle_2.md`.

## Відомі проблеми

- The source metadata has no document IDs. The split uses contiguous 50-file
  blocks as proxy groups. Residual relation leakage is possible.
- Automatic WAV checks do not measure naturalness or pronunciation.
- NCCL cannot use direct P2P on this host. It uses shared-memory transport.
- AMP gave a worse short-run validation result. The full run uses FP32.
- One card reached 85 C. This is below the configured 95 C stop limit, but
  cooling limits sustained power.
- The Git remote contains an embedded credential. Do not print it. Rotate it.

## Наступна одна дія

Listen to the 20 pairs in `training/eval/generated/listening_trimmed_25k/`.
Record the perceptual result before the release package is made.
