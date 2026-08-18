# Scale Readiness

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

## Cycle 1 gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Host resources | PASS | Two RTX 3090 GPUs and 128 GiB RAM were available. | `training/reports/resource_preflight_2gpu.json` | Preserve the host record. |
| Frontend tests | PASS | The tests cover NFC, apostrophes, deterministic output, stress, and version mismatch. | `training/tests/frontend/` | Repeat after a frontend change. |
| eSpeak-ng version | PASS | The fixed version is 1.52.0. | `training/vendor/ESPEAK_NG_VERSION` | Reject version drift. |
| Non-empty phonemes | PASS | The regression set has no empty sequence. | `training/tests/frontend/expected_phonemes.json` | Preserve the snapshot. |
| ESPnet data | PASS | The Kaldi data directories passed validation. | `training/espnet_recipe/data/smoke_*` | Regenerate on a new host. |
| Token list | PASS | ESPnet made the token list with 0 percent OOV. | `training/dump/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve the file. |
| Statistics | PASS | Speech, pitch, and energy statistics exist. | `training/exp/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve the files. |
| JETS construction | PASS | ESPnet made the model and both optimizers. | `training/exp/` | Keep the fixed config. |
| Smoke training | PASS | The model ran finite iterations and made a checkpoint. | `training/reports/crisp_dm_cycle_1.md` | Use the full gate. |
| Smoke inference | PASS | The smoke WAV files are valid 24 kHz mono files. | `training/reports/smoke_inference.json` | Use the full gate. |

All critical cycle 1 gates have `PASS`.

## Expanded-v4 trim-only 100K gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Trim-only manifest | PASS | 207,505 records and 467.127 hours passed validation. | `training/data/expanded_v4_trim_only/manifests/all.parquet` | Preserve its hash. |
| Processing profile | PASS | Boundary trim is on. DeepFilterNet, high-pass, de-essing, compression, and R128 are off. | `training/reports/expanded_v4_trim_only_full_dataset.json` | Keep the profile fixed. |
| Active duration range | PASS | All records are from 2 to 20 seconds. | `training/reports/expanded_v4_trim_only_full_validation.json` | Keep long audio out of this run. |
| Split isolation | PASS | The report has no source-group leakage or duplicate audio hash. | `training/reports/expanded_v4_trim_only_full_analysis.json` | Repeat after a data change. |
| Hybrid embeddings | PASS | 103,752 pre-trim and 103,753 post-trim vectors exist. | `training/reports/expanded_v4_trim_only_full_hybrid_embeddings.json` | Preserve the ECAPA revision. |
| Token list | PASS | The token list has 138 entries and matches the epoch 367 list. | `training/dump_expanded_v4_trim_only/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve it with the model. |
| Statistics | PASS | The required ESPnet statistics exist. | `training/exp_expanded_v4_trim_only/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve them with the model. |
| Dual-GPU smoke run | PASS | The smoke run completed 100 new steps. | `training/reports/expanded_v4_trim_only_smoke_checkpoint.json` | Preserve the report. |
| Dual-GPU full run | PASS | Training completed 100,000 new steps with exit code 0. | `training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/train.log` | Use listening results for selection. |
| Runtime health | PASS | No NaN, OOM, or critical runtime error occurred. | `training/reports/training_status_expanded_v4_trim_only_100k.jsonl` | Preserve the log. |
| Monitor interval | PASS | The largest measured interval is 15.011 minutes. | `training/reports/expanded_v4_trim_only_training_monitor_summary.json` | Keep the 30-minute limit. |
| GPU memory | PASS | The training log reports 23.061 GiB peak cached memory. | `training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/train.log` | Do not increase the batch size. |
| GPU temperature | PASS | The sampled maximum values are 87 and 78 degrees C. | `training/reports/expanded_v4_trim_only_training_monitor_summary.json` | Keep the 90-degree stop limit. |
| Disk reserve | PASS | The minimum measured reserve was 37.45 GiB. | `training/reports/expanded_v4_trim_only_training_monitor_summary.json` | Keep the 30 GiB limit. |
| 100K checkpoint | PASS | The counters are 100,000 and all 549 model tensors are finite. | `training/reports/expanded_v4_trim_only_100k_checkpoint.json` | Preserve the file outside Git. |
| Milestone match | PASS | The 100K milestone matches the checkpoint model exactly. | `training/reports/expanded_v4_trim_only_100k_checkpoint.json` | Preserve its SHA-256. |
| Validation-best checkpoints | PASS | The pipeline kept the best generator, mel, and alignment epochs. | `training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/best_checkpoints/validation_best.json` | Compare them by listening. |
| Fixed-set inference | PASS | All 5,600 WAV files passed. | `training/reports/expanded_v4_trim_only_inference_*k.json` | Preserve the reports. |
| Five-voice inference | PASS | Five of five WAV files passed. | `training/reports/five_voice_expanded_v4_trim_only_100k.json` | Start human listening. |
| Full tests | PASS | All 101 tests passed. | `training/tests/` | Repeat after a code change. |
| Human listening | NOT RUN | Automatic tests do not measure naturalness. | `training/eval/generated/five_voice_expanded_v4_trim_only_100k/` | Check rasp, metallic sound, and robotic sound. |

The 100K training and automatic evaluation are complete. The automatic gates
have `PASS`. Release promotion is not ready because human listening has
`NOT RUN`.

## Expanded-v8 training-cascade 100K gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Source corpus | PASS | The corpus has 74,156 records, 92.266 hours, and no VOA records. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_full_dataset.json` | Preserve the manifest and profile hash. |
| Processing profile | PASS | The pipeline used ClearerVoice, Sidon, de-essing, de-clicking, limiting, DeepFilterNet3, RNNoise85, and a peak-safe loudness match. | `training/data/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/manifests/all.parquet` | Do not change the order for this model. |
| PCM24 and physical output | PASS | All 74,156 final paths refer to physical mono 24 kHz PCM24 WAV files. | `training/data/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/audio_24k/` | Preserve the files outside Git. |
| Degenerate-output control | PASS | Nine files used the recorded fallback. No final file was too quiet or clipped. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_full_dataset.json` | Review these files during listening. |
| Dataset validation | PASS | All 74,156 records passed. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_full_validation.json` | Repeat after a data change. |
| Exact 50/50 embeddings | PASS | 37,078 raw vectors and 37,078 clean vectors exist. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_full_hybrid_embeddings.json` | Preserve the assignment and ECAPA revision. |
| Frontend | PASS | eSpeak-ng 1.52.0 made non-empty Ukrainian phonemes. | `training/vendor/ESPEAK_NG_VERSION` | Reject version drift. |
| Token list | PASS | The SHA-256 is `dc4ea2634513b87e2e70b27d9545f5dd14cea91dd99391f615a24ded5ee964f2`. | `training/dump_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve it with the model. |
| Statistics | PASS | ESPnet made 252 required statistics files. | `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve them with the model. |
| Dual-GPU smoke run | PASS | The smoke run completed 100 new steps. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_smoke_checkpoint.json` | Preserve the report. |
| Smoke inference | PASS | All 1,418 smoke WAV files passed. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_smoke_inference.json` | Preserve the report. |
| Dual-GPU full run | PASS | Training completed 100,000 new steps with exit status 0. | `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/train.log` | Use listening results for selection. |
| Runtime health | PASS | No NaN, OOM, or critical runtime error occurred. | `training/reports/training_status_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k.jsonl` | Preserve the log. |
| Monitor interval | PASS | The largest measured training interval was 15.011 minutes. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_training_monitor_summary.json` | Keep the 30-minute limit. |
| GPU memory | PASS | The training log reports 20.715 GiB peak cached memory. | `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/train.log` | Keep the tested batch size. |
| GPU temperature | PASS | The sampled maximum values are 85 and 76 degrees C. No thermal slowdown occurred. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_training_monitor_summary.json` | Keep the 90-degree stop limit. |
| Disk reserve | PASS | The minimum measured reserve was 42.19 GiB. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_training_monitor_summary.json` | Keep the 30 GiB limit. |
| 100K checkpoint | PASS | The counters are 100,000 and all 549 model tensors are finite. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k_checkpoint.json` | Preserve the file outside Git. |
| Milestone match | PASS | The 100K model matches the checkpoint model exactly. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k_checkpoint.json` | Preserve its SHA-256. |
| Validation-best checkpoints | PASS | The pipeline kept the best generator, mel, and alignment epochs. | `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/best_checkpoints/validation_best.json` | Compare them by listening. |
| Fixed-set inference | PASS | All 5,672 WAV files passed. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_inference_*k.json` | Preserve the reports. |
| Five-voice inference | PASS | Five physical WAV files and metadata passed. | `training/eval/generated/five_voice_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k/` | Start human listening. |
| Full tests | PASS | All 159 tests passed. | `training/tests/` | Repeat after a code change. |
| Human listening | NOT RUN | Automatic tests do not measure naturalness. | `training/eval/generated/five_voice_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k/` | Check rasp, metallic sound, and robotic sound. |

The v8 100K training and automatic evaluation are complete. All automatic
gates have `PASS`. Release promotion is not ready because human listening has
`NOT RUN`.
