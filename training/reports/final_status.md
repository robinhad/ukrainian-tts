# Final status

## Виконано

Pipeline implementation is in progress on branch `autotrain`. This report contains
results only for commands that ran.

## CRISP-DM cycle 1

- Business Understanding: PASS
- Data Understanding: PASS for smoke subset
- Data Preparation: PASS for smoke subset
- Modeling: PASS for 100-iteration smoke run
- Evaluation: PASS for 32 smoke-eval WAVs
- Deployment: PASS for local raw-WAV entrypoint

## CRISP-DM cycle 2

- Business Understanding: PASS
- Data Understanding: PASS for the complete pinned corpus
- Data Preparation: PASS; full ESPnet token/statistics stages completed
- Modeling: batch calibration and 1k full-corpus sanity milestone PASS; 25k pending
- Evaluation: 1k fixed-set inference PASS for 180/180 WAVs
- Deployment: release candidate pending checkpoint evaluation

## MVP-gate

See `scale_readiness.md` for evidence-backed statuses.

## Фактичні запуски

- Host resource preflight: exit 0. Two RTX 3090 GPUs were idle. The RAM and disk checks passed.
- Minimal sanitation tests: exit 0; 10 passed.
- First bootstrap attempt: exit 1. PyTorch 2.9.1+cu128 was installed. The PyPI `espnet==202604` requirement was not valid. The pipeline now installs ESPnet from commit `cff0a07`.
- Second bootstrap attempt: exit 1. ESPnet was installed and eSpeak was built. eSpeak made a static library. The next configuration enabled the shared library that phonemizer requires.
- Environment bootstrap retry: exit 0; pinned ESPnet and shared eSpeak-ng installed.
- Frontend regression suite: exit 0; 18 tests passed in 0.80 seconds.
- Smoke subset materialization: the first implementation wrote all 320 records. It exited 250 during Hugging Face streaming shutdown. The pinned Parquet implementation then exited 0.
- Audio preparation: exit 0; 320 mono PCM 24 kHz WAV files created.
- Manifest validation: exit 0; 320 utterances, 0.474 hours, no errors, 28 clipping flags.
- ESPnet stages 1--6: exit 0 after fixes to `run.pl`, `resampy`, and `--srctexts`. The stages made the data directories, a 153-token list, and the statistics.
- JETS dry run: exit 0; 83.31M-parameter model and both optimizers constructed on CUDA.
- First training launch: exit 1 before the first batch. ESPnet GANTrainer rejected `accum_grad > 1`. The configuration now uses `accum_grad: 1`.
- Smoke training retry: exit 0; 100 train iterations, 5 validation batches, peak cached VRAM 5.938 GiB, no NaN/OOM, checkpoint saved.
- ESPnet smoke inference: exit 0; 32 WAVs generated.
- Independent WAV validation: exit 0; 32/32 mono 24 kHz finite/nonzero outputs, no clipping, median RTF 0.0185.
- Local inference entrypoint: exit 0; 2.955-second WAV plus JSON metadata, RTF 0.112.
- Full materialization: exit 0; 6787/6962 rows selected, 6461/146/180 group split.
- Full audio preparation and validation: exit 0; 10.343 hours, no hard errors, 366 clipping flags.
- Full frontend snapshot: exit 0; 500 cases; all 18 tests passed in 5.06 seconds.
- Full ESPnet stages 1--6: exit 0; data directories, token list and pitch/energy statistics produced.
- FP32 batch calibration: 1M, 2M, 2.5M and 3M each completed 200 iterations; 3M selected with 20.727 GiB peak cache and about 12% VRAM reserve.
- AMP check: exit 0 and finite. The AMP validation generator loss was 140.497. The FP32 loss was 77.744. Long training does not use AMP.
- Full 1k training: exit 0; 1000 iterations in 25m38s, train/valid generator loss 82.403/84.718, peak cached VRAM 20.727 GiB, checkpoint saved with no NaN/OOM.
- Full 1k ESPnet inference: exit 0 in 15s; all 180 fixed eval utterances generated.
- Full 1k independent WAV validation: exit 0; 180/180 mono 24 kHz finite/nonzero WAVs, no clipping warnings, median RTF 0.00893.
- First local inference from a new shell: exit 1. The entry point did not find pinned eSpeak data outside the recipe environment. It now finds the repository-local data and rejects a different version or hash.
- Local inference retry with eSpeak variables explicitly unset: exit 0; 3.477-second mono 24 kHz WAV, RTF 0.103, metadata written.
- Post-fix automated tests: exit 0; 19 passed in 5.10 seconds.

## Створені артефакти

- Frontend, regression corpus, dataset/QC scripts, ESPnet recipe and JETS config are tracked under `training/`.
- Runtime manifests: `training/data/manifests/`; ESPnet data: `training/espnet_recipe/data/`.
- Token list: `training/dump/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `training/exp/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Checkpoint: `training/exp/tts_train_jets_uk_24k_raw_phn_espeak_ng_ukrainian_max_epoch1/1epoch.pth` (runtime artifact).
- Generated eval WAVs: the matching `decode_jets_train.total_count.ave/smoke_eval/wav/` directory.
- Local example: `training/eval/generated/example.wav` and adjacent metadata JSON.
- Evaluation report: `training/reports/smoke_inference.json`.
- Full manifests: `training/data/full/manifests/`; detailed report: `training/reports/full_data_analysis.json`.
- Full token/statistics artifacts: `training/dump_full/token_list/` and `training/exp_full/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Full 1k checkpoint: `training/exp_full/tts_jets_uk_24k_full/1epoch.pth` (runtime artifact).
- Full 1k eval WAVs: `training/exp_full/tts_jets_uk_24k_full/decode_jets_latest/eval/wav/`.
- Full 1k evaluation report: `training/reports/full_inference_1k.json`.
- Full 1k local example: `training/eval/generated/full_1k.wav` and adjacent metadata JSON.

## Відомі проблеми

- Twenty-eight smoke files have a clipping QC flag.
- The 100-iteration checkpoint proves operability only; expected speech quality is low.
- The source metadata does not have document IDs. The split procedure uses contiguous 50-file blocks as proxy groups.
- The full corpus has 366 clipping flags and 104 phoneme tokens occurring at most ten times.
- AMP was stable but degraded the 200-iteration validation metric and is disabled.
- The 1k checkpoint is technically valid. It is not selected for perceptual quality. The 25k checkpoint requires a listening evaluation.
- The Git remote contains an embedded credential. Do not print it. Rotate the credential.

## Наступна одна дія

`cd training && ./scripts/run_full_training.sh 25000`
