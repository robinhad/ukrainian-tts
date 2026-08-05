# Final Status

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

## Виконано

The expanded-v4 trim-only pipeline prepared 207,505 utterances. The total
duration is 467.127 hours. Each training segment is from 2 to 20 seconds.

The pipeline used boundary silence trim. It did not use DeepFilterNet. It did
not use a high-pass filter, de-essing, compression, or R128 normalization.

The pipeline made 103,752 speaker embeddings from audio before trim. It made
103,753 speaker embeddings from audio after trim. Each vector has 192 values.
The training audio is the clean trim-only version.

ESPnet made a 138-token list and the required statistics. JETS started from
the epoch 367 model weights. It used new optimizer and step states. It used
two RTX 3090 GPUs and completed 100,000 new optimizer steps. Training ended
with exit code 0 at 23:41 Kyiv time on 2026-08-05.

The evaluation made 1,400 WAV files for each of the 25K, 50K, 75K, and 100K
milestones. All 5,600 files passed the automatic checks. The listening
evaluation made five more WAV files. All five files passed the automatic
checks.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS.
- Data Preparation: PASS.
- Modeling: PASS.
- Evaluation: PASS.
- Deployment: PASS.

The smoke cycle completed 100 optimizer steps on two GPUs. It made 32 valid
WAV files before the full run started.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Source policy | PASS | The approved expanded-v3 sources supply this data revision. |
| Trim-only training data | PASS | 207,505 records and 467.127 hours passed validation. |
| Segment duration | PASS | The minimum is 2.0 seconds. The maximum is 20.0 seconds. |
| Split isolation | PASS | No source-group leakage or duplicate audio hash exists. |
| Hybrid speaker embeddings | PASS | 103,752 pre-trim and 103,753 post-trim vectors exist. |
| Frontend and phonemes | PASS | eSpeak-ng 1.52.0 made non-empty phonemes. |
| Token list | PASS | The fixed token list has 138 entries. |
| Pitch and energy statistics | PASS | ESPnet made all required statistics. |
| Dual-GPU smoke run | PASS | The smoke run completed 100 steps and made a checkpoint. |
| Dual-GPU full run | PASS | JETS completed 100,000 new optimizer steps. |
| Finite model values | PASS | The audit found 549 finite model tensors. |
| 100K checkpoint | PASS | The optimizer and report counters are 100,000. |
| Milestone match | PASS | The 100K milestone matches the checkpoint model exactly. |
| Fixed-set inference | PASS | 5,600 of 5,600 WAV files passed. |
| Five-voice inference | PASS | Five of five WAV files passed. |
| Monitor interval | PASS | The largest measured interval is 15.011 minutes. |
| Disk reserve | PASS | The minimum measured reserve is 37.45 GiB. |
| Test suite | PASS | All 101 tests passed. |
| Human listening | NOT RUN | A person must check rasp, metallic sound, and robotic sound. |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/prepare_expanded_v4_trim_only.sh smoke` | 0 | The pipeline made the smoke data and statistics. | `training/data/expanded_v4_trim_only_smoke/` |
| `training/scripts/run_expanded_v4_trim_only_smoke.sh` | 0 | The run completed 100 steps and made 32 valid WAV files. | `training/reports/expanded_v4_trim_only_smoke_checkpoint.json` |
| `training/scripts/prepare_expanded_v4_trim_only.sh full` | 0 | The pipeline made 207,505 records, hybrid embeddings, and statistics. | `training/reports/expanded_v4_trim_only_full_scale_readiness.json` |
| `training/scripts/launch_expanded_v4_trim_only_training.sh 100000` | 0 | JETS completed 100,000 steps. | `training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/train.log` |
| `training/scripts/evaluate_expanded_v4_trim_only_milestone.sh 25k` | 0 | All 1,400 fixed-set WAV files passed. | `training/reports/expanded_v4_trim_only_inference_25k.json` |
| `training/scripts/evaluate_expanded_v4_trim_only_milestone.sh 50k` | 0 | All 1,400 fixed-set WAV files passed. | `training/reports/expanded_v4_trim_only_inference_50k.json` |
| `training/scripts/evaluate_expanded_v4_trim_only_milestone.sh 75k` | 0 | All 1,400 fixed-set WAV files passed. | `training/reports/expanded_v4_trim_only_inference_75k.json` |
| `training/scripts/evaluate_expanded_v4_trim_only_milestone.sh 100k` | 0 | All 1,400 fixed-set WAV files passed. | `training/reports/expanded_v4_trim_only_inference_100k.json` |
| `training/scripts/generate_five_voice_expanded_v4_trim_only_eval.sh 100k` | 0 | Five listening WAV files passed. | `training/reports/five_voice_expanded_v4_trim_only_100k.json` |
| `training/.venv/bin/python -m pytest -q training/tests` | 0 | All 101 tests passed in 10.37 seconds. | `training/tests/` |

## Training metrics

The last validation generator loss is 57.751. The last validation mel loss is
39.409. The last validation alignment loss is 4.396. The last validation
discriminator loss is 2.465.

The best validation generator loss is 56.384 at epoch 28. The best validation
mel loss is 38.839 at epoch 28. The best validation alignment loss is 4.396
at epoch 100. The pipeline kept these checkpoints.

The sampled peak power was 246.55 W on GPU 0 and 240.33 W on GPU 1. Both GPUs
reached 100 percent compute use. The sampled peak VRAM was 23,974 MiB on GPU 0
and 24,104 MiB on GPU 1. The training log reports 23.061 GiB of peak cached
memory. GPU 0 reached 87 degrees C. GPU 1 reached 78 degrees C.

The minimum measured free disk space was 37.45 GiB. This value stayed above
the 30 GiB cleanup limit.

## Створені артефакти

- Manifest: `training/data/expanded_v4_trim_only/manifests/all.parquet`.
- ESPnet data: `training/espnet_recipe/data/expanded_v4_trim_only_*`.
- Token list: `training/dump_expanded_v4_trim_only/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `training/exp_expanded_v4_trim_only/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Final checkpoint: `training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/milestones/100k.pth`.
- Validation-best checkpoints: `training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/best_checkpoints/`.
- Fixed evaluation reports: `training/reports/expanded_v4_trim_only_inference_*k.json`.
- Listening report: `training/reports/five_voice_expanded_v4_trim_only_100k.json`.
- Listening WAV files: `training/eval/generated/five_voice_expanded_v4_trim_only_100k/`.
- Runtime report: `training/reports/expanded_v4_trim_only_training_monitor_summary.json`.
- Checkpoint audit: `training/reports/expanded_v4_trim_only_100k_checkpoint.json`.

The final checkpoint SHA-256 is
`35d576109acd745511c8692a167f504380d485344e09e9be363449012fd9b2b3`.

## Відомі проблеми

- Human listening is not complete. The automatic checks cannot measure naturalness.
- The user must check if the new audio has rasp, metallic sound, or robotic sound.
- GPU 0 reached 87 degrees C. The stop limit was 90 degrees C.
- Flash Attention is not installed. ESPnet used its standard attention code.
- The release status is `NOT_READY` until human listening is complete.

## Наступна одна дія

Listen to the five WAV files in
`training/eval/generated/five_voice_expanded_v4_trim_only_100k/`. Report which
voice has the least rasp, metallic sound, and robotic sound.
