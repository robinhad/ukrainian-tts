# Final status

## Виконано

The reproducible Ukrainian JETS pipeline is on branch `autotrain`. All new
training code is under `training/`. The full-corpus run reached 25,000 iterations
on two RTX 3090 GPUs. The command exited with status 0. The 25k checkpoint and all
evaluation reports are complete.

The pipeline uses Ukrainian text, the Lada speaker, 24 kHz audio, ESPnet2 JETS,
GAN-TTS, and the pinned eSpeak-ng frontend. Training audio has no mastering,
compression, de-essing, denoise, or loudness normalization.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS for the smoke subset.
- Data Preparation: PASS for the smoke subset.
- Modeling: PASS for the 100-iteration smoke run.
- Evaluation: PASS for 32 smoke-eval WAV files.
- Deployment: PASS for the local raw-WAV entry point.

## CRISP-DM cycle 2

- Business Understanding: PASS.
- Data Understanding: PASS for the pinned full corpus.
- Data Preparation: PASS. ESPnet token and statistics stages completed.
- Modeling: PASS for batch calibration, 1k sanity, and the 25k dual-GPU run.
- Evaluation: PASS for fixed-set inference and automatic WAV checks.
- Deployment: PASS for the 25k local raw-WAV entry point. Release selection is
  pending the listening test.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Frontend tests | PASS | 21 tests passed in 5.18 s |
| eSpeak-ng pin | PASS | Version 1.52.0 and data hash are pinned |
| Non-empty phonemes | PASS | Regression and corpus checks found no empty sequence |
| ESPnet data directories | PASS | Smoke and full directories passed validation |
| Token list | PASS | Full tokenization had 0.0% OOV |
| Statistics | PASS | Speech, pitch, and energy statistics exist |
| JETS construction | PASS | JETS and both AdamW optimizers initialized |
| Stable training | PASS | 25,000 iterations; no NaN, OOM, or runtime error |
| Checkpoint | PASS | 25k checkpoint exists and has a verified SHA-256 |
| Inference WAV | PASS | 25k fixed eval passed for 180/180 files |
| Reproduction commands | PASS | README, STEPS, and command log contain the commands |

## Фактичні запуски

- Host resource preflight: exit 0. PyTorch found two RTX 3090 GPUs. The CUDA
  matrix check passed on both GPU UUIDs.
- Environment and frontend tests: exit 0. The final suite reported `21 passed in
  5.18s`.
- Full data preparation: exit 0. It selected 6787 of 6962 rows and made a
  6461/146/180 group split.
- Full audio validation: exit 0. It validated 10.343 hours and reported 366
  non-blocking clipping flags.
- ESPnet stages 1--6: exit 0. They made the data directories, token list, speech
  statistics, pitch statistics, and energy statistics.
- FP32 calibration: exit 0 at 1M, 2M, 2.5M, and 3M `batch_bins`. The run selected
  3M. The AMP comparison was finite but had a worse validation loss.
- Full 1k training: exit 0. Train/validation generator loss was 82.403/84.718.
- First dual-GPU 25k launch: exit 1 before a batch. The activation template reset
  `CUDA_VISIBLE_DEVICES`. The local fix now keeps an explicit multi-GPU value.
- Dual-GPU 25k retry: exit 0. The command ran for 25,316 seconds and reached
  25,000 iterations. Final train/validation generator loss was 61.883/71.475.
  Peak cached VRAM was 15.551 GiB.
- GPU power monitor: 231 samples. GPU0 mean/max power was 188.91/235.28 W and its
  maximum temperature was 86 C. GPU1 mean/max power was 212.18/253.82 W and its
  maximum temperature was 78 C.
- TensorBoard event audit: exit 0. The train log has 29 scalar tags. The valid log
  has 16 scalar tags. Both logs reached step 25,000.
- 25k ESPnet inference: exit 0 in 14.745 s. It made all 180 fixed eval files.
- 25k WAV validation: exit 0. It accepted 180/180 mono 24 kHz finite, non-zero
  files. It found no clipping warning. Duration was 2.816--7.424 s. Median RTF
  was 0.00845.
- Candidate inference: exit 0 for 5k, 15k, 17k, and 23k. Each checkpoint made
  180/180 valid files without a clipping warning.
- 25k local inference: exit 0. It made a 4.757-second mono 24 kHz WAV and JSON
  metadata. RTF was 0.0754. Peak absolute sample was 0.457.
- Post-training resource check: exit 0. Both GPUs were idle. The host had 122.06
  GiB available RAM and 314.79 GiB free disk.

## Створені артефакти

- Full manifests: `training/data/full/manifests/`.
- ESPnet data directories: `training/espnet_recipe/data/{train,dev,eval}/`.
- JETS config: `training/espnet_recipe/conf/tuning/train_jets_uk_24k.yaml`.
- ESPnet patch: `training/patches/espnet-espeak-ng-ukrainian.patch`.
- Full token list: `training/dump_full/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Full statistics: `training/exp_full/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Retained checkpoints: `training/exp_full/milestones/{5k,15k,17k,23k,25k}.pth`.
- TensorBoard events: `training/exp_full/tts_jets_uk_24k_full/tensorboard/{train,valid}/`.
- 25k checkpoint SHA-256: `d1ee89bdb99fe40a24a9c44ebfc4004e5ef18b4fc00c08658832ba36846ff4dd`.
- Fixed eval WAV files: `training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_*k/eval/wav/`.
- Evaluation reports: `training/reports/full_inference_{1k,5k,15k,17k,23k,25k}.json`.
- GPU power summary: `training/reports/power_summary.json`.
- 25k local example: `training/eval/generated/example_25k.wav` and adjacent JSON.
- Listening set: `training/eval/generated/listening_25k/`.
- CRISP-DM reports: `training/reports/crisp_dm_cycle_1.md` and
  `training/reports/crisp_dm_cycle_2.md`.

## Відомі проблеми

- The source metadata has no document IDs. The split uses contiguous 50-file
  blocks as proxy groups. Residual relation leakage is possible.
- The full corpus has 366 clipping flags. The pipeline did not modify these files.
- The frontend has 104 phoneme tokens that occur no more than ten times.
- NCCL cannot use direct P2P on this host. It used shared-memory transport.
- AMP was finite but reduced the short-run validation result. Full training used
  FP32.
- Validation loss is best at 15k and second best at 23k. Automatic WAV checks do
  not measure naturalness. A listening test must select the release checkpoint.
- The Git remote contains an embedded credential. Do not print it. Rotate it.

## Наступна одна дія

Listen to the 20 items in `training/eval/generated/listening_25k/`. Compare the
raw reference with 15k, 23k, and 25k. Record one checkpoint selection before the
release package is made.
