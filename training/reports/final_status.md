# Final Status

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

## Виконано

The expanded-v6 pipeline prepared 74,156 utterances. The total duration is
92.266 hours. The training split has 71,334 utterances and 89.223 hours. The
development split has 1,404 utterances. The evaluation split has 1,418
utterances. The corpus has no VOA records.

The pipeline used boundary silence trim, Sidon, and light de-essing. Sidon has
an internal 50 Hz input filter. The pipeline did not use DeepFilterNet, an
additional high-pass filter, compression, loudness normalization, or limiting.
All training files are 24 kHz mono PCM16 WAV files.

The pipeline used an exact 50/50 speaker-embedding mix. It reused 37,078
vectors from audio before Sidon. It calculated 37,078 vectors from Sidon and
de-essed audio. Each vector has 192 values. The training audio is the Sidon and
de-essed version.

ESPnet made the token list and all required statistics. JETS started from the
expanded-v5 epoch 81 model weights. It used new optimizer and step states. It
used two RTX 3090 GPUs in FP32 mode and completed 100,000 new optimizer steps.
Training ended with exit code 0 at 11:18:55 Kyiv time on 2026-08-11.

The evaluation made 1,418 WAV files for each of the 25K, 50K, 75K, and 100K
milestones. All 5,672 files passed the automatic checks. The listening
evaluation made five more WAV files. All five files passed the automatic
checks and are physical files, not symbolic links.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS.
- Data Preparation: PASS.
- Modeling: PASS.
- Evaluation: PASS.
- Deployment: PASS.

The smoke cycle completed 100 optimizer steps on two GPUs. Its checkpoint
audit passed. Smoke inference made 1,418 valid WAV files.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Dataset policy | PASS | The corpus has 74,156 records and has no VOA records. |
| Sidon and de-essing audio | PASS | All files use the fixed v6 processing profile. |
| Dataset validation | PASS | All 74,156 records passed. No file has clipping. |
| Split isolation | PASS | The dataset validator found no split error. |
| Exact 50/50 embeddings | PASS | 37,078 raw and 37,078 clean vectors exist. |
| Frontend and phonemes | PASS | eSpeak-ng 1.52.0 made non-empty phonemes. |
| Token list | PASS | The token-list SHA-256 is `dc4ea2634513b87e2e70b27d9545f5dd14cea91dd99391f615a24ded5ee964f2`. |
| Pitch and energy statistics | PASS | ESPnet made all required statistics. |
| Dual-GPU smoke run | PASS | The smoke run completed 100 steps. |
| Smoke inference | PASS | All 1,418 smoke WAV files passed. |
| Dual-GPU full run | PASS | JETS completed 100,000 new optimizer steps. |
| Finite model values | PASS | The audit found 549 finite model tensors. |
| 100K checkpoint | PASS | The optimizer and report counters are 100,000. |
| Milestone match | PASS | The 100K milestone matches the checkpoint model. |
| Fixed-set inference | PASS | 5,672 of 5,672 WAV files passed. |
| Five-voice inference | PASS | Five of five WAV files passed. |
| Monitor interval | PASS | The largest measured interval is 15.011 minutes. |
| Disk reserve | PASS | The minimum measured reserve is 102.67 GiB. |
| Test suite | PASS | All 124 tests passed. |
| Human listening | NOT RUN | A person must check voice quality. |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/prepare_expanded_v6_sidon_deess_novoa.sh` | 0 | The pipeline made 74,156 records, embeddings, and statistics. | `training/reports/expanded_v6_sidon_deess_novoa_full_scale_readiness.json` |
| `training/scripts/run_expanded_v6_sidon_deess_novoa_smoke.sh` | 0 | The run completed 100 steps and inference passed. | `training/reports/expanded_v6_sidon_deess_novoa_smoke_checkpoint.json` |
| `training/scripts/launch_expanded_v6_sidon_deess_novoa_training.sh 100000` | 0 | JETS completed 100,000 new steps. | `training/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k/train.log` |
| `training/scripts/evaluate_expanded_v6_sidon_deess_novoa_milestone.sh 25k` | 0 | All 1,418 fixed-set WAV files passed. | `training/reports/expanded_v6_sidon_deess_novoa_inference_25k.json` |
| `training/scripts/evaluate_expanded_v6_sidon_deess_novoa_milestone.sh 50k` | 0 | All 1,418 fixed-set WAV files passed. | `training/reports/expanded_v6_sidon_deess_novoa_inference_50k.json` |
| `training/scripts/evaluate_expanded_v6_sidon_deess_novoa_milestone.sh 75k` | 0 | All 1,418 fixed-set WAV files passed. | `training/reports/expanded_v6_sidon_deess_novoa_inference_75k.json` |
| `training/scripts/evaluate_expanded_v6_sidon_deess_novoa_milestone.sh 100k` | 0 | All 1,418 fixed-set WAV files passed. | `training/reports/expanded_v6_sidon_deess_novoa_inference_100k.json` |
| `training/scripts/generate_five_voice_expanded_v6_eval.sh 100k` | 0 | Five listening WAV files passed. | `training/reports/five_voice_expanded_v6_sidon_deess_novoa_100k.json` |
| `training/scripts/audit_finetune_checkpoint.py` | 0 | The step, tensor, and milestone checks passed. | `training/reports/expanded_v6_sidon_deess_novoa_100k_checkpoint.json` |
| `training/scripts/summarize_tensorboard.py` | 0 | Two TensorBoard runs and all scalar tags passed. | `training/reports/expanded_v6_sidon_deess_novoa_tensorboard_metrics.json` |
| `training/.venv/bin/python -m pytest -q training/tests` | 0 | All 124 tests passed in 10.43 seconds. | `training/tests/` |

## Training metrics

The final validation generator loss is 55.860. The final validation mel loss
is 35.059. The final validation alignment loss is 3.995. The final validation
discriminator loss is 2.004.

The minimum validation generator loss is 54.422 at epoch 42. The minimum
validation mel loss is 34.084 at epoch 94. The minimum validation alignment
loss is 3.992 at epoch 88. The pipeline kept these checkpoints.

The train generator loss decreased from 75.321 to 57.693. The train mel loss
decreased from 54.303 to 36.570. These values are TensorBoard epoch summaries
for steps 10 and 100,000.

GPU 0 had a mean sampled power of 198.63 W and a maximum of 242.65 W. GPU 1
had a mean sampled power of 205.59 W and a maximum of 242.33 W. Both GPUs
reached 100 percent compute use. The maximum sampled VRAM was 21,734 MiB on
GPU 0 and 19,716 MiB on GPU 1. The training log reports 20.742 GiB of peak
cached memory. GPU 0 reached 87 degrees C. GPU 1 reached 77 degrees C.

The minimum measured free disk space was 102.67 GiB. This value stayed above
the 30 GiB cleanup limit.

## Створені артефакти

- Manifest: `training/data/expanded_v6_sidon_deess_novoa/manifests/all.parquet`.
- ESPnet data: `training/espnet_recipe/data/expanded_v6_sidon_deess_novoa_*`.
- Token list: `training/dump_expanded_v6_sidon_deess_novoa/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `training/exp_expanded_v6_sidon_deess_novoa/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Resume checkpoint: `training/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k/checkpoint.pth`.
- Final model: `training/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k/milestones/100k.pth`.
- Validation-best models: `training/exp_expanded_v6_sidon_deess_novoa/tts_jets_uk_24k_expanded_v6_sidon_deess_novoa_from_v5e81_100k/best_checkpoints/`.
- Fixed evaluation reports: `training/reports/expanded_v6_sidon_deess_novoa_inference_*k.json`.
- Listening report: `training/reports/five_voice_expanded_v6_sidon_deess_novoa_100k.json`.
- Listening WAV files: `training/eval/generated/five_voice_expanded_v6_sidon_deess_novoa_100k/`.
- TensorBoard summary: `training/reports/expanded_v6_sidon_deess_novoa_tensorboard_metrics.json`.
- Runtime summary: `training/reports/expanded_v6_sidon_deess_novoa_training_monitor_summary.json`.
- Checkpoint audit: `training/reports/expanded_v6_sidon_deess_novoa_100k_checkpoint.json`.

The final model SHA-256 is
`d4ed14c8bfb828c97fffb7ca33065a882c6ba342829eb5a00f3dda8b4205d9d5`.

## Відомі проблеми

- Human listening is not complete. Automatic checks cannot measure naturalness.
- The user must check for rasp, metallic sound, and robotic sound.
- GPU 0 reached 87 degrees C. The stop limit was 90 degrees C.
- Flash Attention is not installed. ESPnet used its standard attention code.
- DIO reported some fully unvoiced frames. Training continued without NaN.
- The release status is `NOT_READY` until human listening is complete.

## Наступна одна дія

Listen to the five WAV files in
`training/eval/generated/five_voice_expanded_v6_sidon_deess_novoa_100k/`.
Report which voice has the least rasp, metallic sound, and robotic sound.
