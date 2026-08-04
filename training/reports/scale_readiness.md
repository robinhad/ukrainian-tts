# Scale readiness

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify the report.

## Cycle 1 gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Host resources | PASS | Two RTX 3090 GPUs and 128 GiB RAM were available. | `reports/resource_preflight_2gpu.json` | Preserve the host record. |
| Frontend tests | PASS | The tests cover NFC, apostrophes, deterministic output, stress, and version mismatch. | `tests/frontend/` | Repeat after a frontend change. |
| eSpeak-ng version | PASS | The fixed version is 1.52.0. | `vendor/ESPEAK_NG_VERSION` | Reject version drift. |
| Non-empty phonemes | PASS | The regression set has no empty sequence. | `tests/frontend/expected_phonemes.json` | Preserve the snapshot. |
| ESPnet data | PASS | The Kaldi data directories passed validation. | `espnet_recipe/data/smoke_*` | Regenerate on a new host. |
| Token list | PASS | ESPnet made the token list with 0 percent OOV. | `dump/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve the file. |
| Statistics | PASS | Speech, pitch, and energy statistics exist. | `exp/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve the files. |
| JETS construction | PASS | ESPnet made the model and both optimizers. | `exp/` | Keep the fixed config. |
| Smoke training | PASS | The model ran finite iterations and made a checkpoint. | `reports/crisp_dm_cycle_1.md` | Use the full gate. |
| Smoke inference | PASS | The smoke WAV files are valid 24 kHz mono files. | `reports/smoke_inference.json` | Use the full gate. |

All critical cycle 1 gates have `PASS`.

## Expanded-v3 500K gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Source policy | PASS | The default-deny review permits all included sources. | `conf/expanded_v3_sources.yaml` | Keep each revision fixed. |
| Clean manifest | PASS | 208,867 records and 469.755 hours passed validation. | `data/expanded_v3/manifests/all.parquet` | Preserve its SHA-256. |
| Active duration range | PASS | All records are from 2 to 20 seconds. | `reports/expanded_v3_full_validation.json` | Keep deferred long audio out of this model. |
| Split isolation | PASS | The report has no source-group leakage or duplicate audio hash. | `reports/expanded_v3_full_analysis.json` | Repeat after a data change. |
| Hybrid embeddings | PASS | 104,433 raw and 104,434 clean vectors exist. | `reports/expanded_v3_full_hybrid_embeddings.json` | Preserve the ECAPA revision. |
| Token list | PASS | The token list has 138 entries and 0 percent OOV. | `dump_expanded_v3/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve it with the model. |
| Statistics | PASS | The required ESPnet statistics exist. | `exp_expanded_v3/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve them with the model. |
| Dual-GPU training | PASS | Training completed 500,000 iterations with exit code 0. | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/train.log` | Use listening results for selection. |
| Runtime health | PASS | No NaN, OOM, or critical runtime error occurred. | `reports/training_status_expanded_v3_500k.jsonl` | Preserve the log. |
| GPU memory | PASS | The training log reports 23.062 GiB peak cached memory. | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/train.log` | Do not increase the batch size. |
| Disk reserve | PASS | The minimum reported reserve was 48.93 GiB. The limit was 30 GiB. | `reports/training_status_expanded_v3_500k.jsonl` | Keep the 30 GiB limit. |
| 500K checkpoint | PASS | The milestone and epoch files are byte-identical. | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/milestones/500k.pth` | Preserve the file outside Git. |
| Validation-best checkpoints | PASS | The pipeline kept the best generator, mel, and alignment epochs. | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/best_checkpoints/validation_best.json` | Compare them by listening. |
| Fixed-set inference | PASS | 1,418 of 1,418 WAV files passed. Median RTF is 0.00669. | `reports/expanded_v3_inference_500k.json` | Preserve the report. |
| Five-voice inference | PASS | Five of five WAV files passed. | `reports/five_voice_expanded_v3_500k.json` | Do the human listening check. |
| Artifact checksums | PASS | `sha256sum -c` passed for every listed file. | `reports/expanded_v3_500k_artifacts.sha256` | Preserve the checksum file. |
| Old 800-hour VOA target | FAIL | The accepted VOA subset has 377.489 hours. | `reports/expanded_v3_full_scale_readiness.json` | Treat this as an authorized run exception. |
| Human listening | NOT RUN | Automatic checks cannot measure naturalness. | `eval/generated/five_voice_expanded_v3_500k/` | Listen before release. |

The 500K training and automatic evaluation are complete. The old 800-hour VOA
target remains open. The user authorized training on the accepted dataset.
Release promotion is not ready until the human listening check is complete.
