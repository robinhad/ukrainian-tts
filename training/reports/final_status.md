# Final status

## Виконано

Pipeline implementation is in progress on branch `autotrain`. This report is updated
after each executed stage and does not claim unexecuted training or synthesis.

## CRISP-DM cycle 1

- Business Understanding: PASS
- Data Understanding: PASS for smoke subset
- Data Preparation: PASS for smoke subset
- Modeling: model construction PASS; training NOT RUN
- Evaluation: NOT RUN
- Deployment: implementation in progress

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

## Створені артефакти

- Frontend, regression corpus, dataset/QC scripts, ESPnet recipe and JETS config are tracked under `training/`.
- Runtime manifests: `training/data/manifests/`; ESPnet data: `training/espnet_recipe/data/`.
- Token list: `training/dump/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `training/exp/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- No checkpoint or generated WAV exists yet.

## Відомі проблеми

- Twenty-eight smoke files carry a clipping QC flag and require review before scaling.
- GPU training and inference have not run yet.
- The repository Git remote contains an embedded credential; it must not be printed and should be rotated.

## Наступна одна дія

`cd training && source ./activate.sh && ./espnet_recipe/run.sh --stage 7 --stop_stage 7 --train_args "--max_epoch 1"`
