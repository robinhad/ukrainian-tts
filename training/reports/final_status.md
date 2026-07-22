# Final status

## Виконано

Pipeline implementation is in progress on branch `autotrain`. This report is updated
after each executed stage and does not claim unexecuted training or synthesis.

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
- Modeling: batch calibration PASS; 1k/25k+ long run pending
- Evaluation: fixed 180-utterance eval set prepared; full milestone NOT RUN
- Deployment: release candidate pending checkpoint evaluation

## MVP-gate

See `scale_readiness.md` for evidence-backed statuses.

## Фактичні запуски

- Host resource preflight: exit 0; two idle RTX 3090 GPUs, RAM and disk thresholds pass.
- Minimal sanitation tests: exit 0; 10 passed.
- First bootstrap attempt: exit 1 after PyTorch 2.9.1+cu128 installed; removed an invalid PyPI `espnet==202604` requirement because the pinned ESPnet release is installed from commit `cff0a07`.
- Second bootstrap attempt: ESPnet installed and eSpeak built, then exit 1 because eSpeak defaulted to a static library; enabled the shared library required by phonemizer.
- Environment bootstrap retry: exit 0; pinned ESPnet and shared eSpeak-ng installed.
- Frontend regression suite: exit 0; 18 tests passed in 0.80 seconds.
- Smoke subset materialization: first implementation wrote all 320 records but exited 250 in Hugging Face streaming shutdown; pinned Parquet implementation then exited 0.
- Audio preparation: exit 0; 320 mono PCM 24 kHz WAV files created.
- Manifest validation: exit 0; 320 utterances, 0.474 hours, no errors, 28 clipping flags.
- ESPnet stages 1--6: after fixing `run.pl`, `resampy`, and `--srctexts`, exit 0; data directories, 153-token list and statistics created.
- JETS dry run: exit 0; 83.31M-parameter model and both optimizers constructed on CUDA.
- First training launch: exit 1 before the first batch because ESPnet GANTrainer rejects `accum_grad > 1`; changed it to 1.
- Smoke training retry: exit 0; 100 train iterations, 5 validation batches, peak cached VRAM 5.938 GiB, no NaN/OOM, checkpoint saved.
- ESPnet smoke inference: exit 0; 32 WAVs generated.
- Independent WAV validation: exit 0; 32/32 mono 24 kHz finite/nonzero outputs, no clipping, median RTF 0.0185.
- Local inference entrypoint: exit 0; 2.955-second WAV plus JSON metadata, RTF 0.112.
- Full materialization: exit 0; 6787/6962 rows selected, 6461/146/180 group split.
- Full audio preparation and validation: exit 0; 10.343 hours, no hard errors, 366 clipping flags.
- Full frontend snapshot: exit 0; 500 cases; all 18 tests passed in 5.06 seconds.
- Full ESPnet stages 1--6: exit 0; data directories, token list and pitch/energy statistics produced.
- FP32 batch calibration: 1M, 2M, 2.5M and 3M each completed 200 iterations; 3M selected with 20.727 GiB peak cache and about 12% VRAM reserve.
- AMP check: exit 0 and finite, but validation generator loss 140.497 versus 77.744 for FP32; AMP rejected.

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

## Відомі проблеми

- Twenty-eight smoke files carry a clipping QC flag and require review before scaling.
- The 100-iteration checkpoint proves operability only; expected speech quality is low.
- Source metadata lacks document IDs; contiguous 50-file blocks are the documented proxy for related recordings.
- The full corpus has 366 clipping flags and 104 phoneme tokens occurring at most ten times.
- AMP was stable but degraded the 200-iteration validation metric and is disabled.
- The repository Git remote contains an embedded credential; it must not be printed and should be rotated.

## Наступна одна дія

`cd training && ./scripts/run_full_training.sh 1000`
