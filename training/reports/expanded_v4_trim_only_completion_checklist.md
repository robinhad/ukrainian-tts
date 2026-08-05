# Expanded-v4 Trim-Only Completion Checklist

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

## Scope

This run uses the expanded-v4 trim-only data. The run does not use DeepFilterNet. It does not use a high-pass filter, de-essing, compression, or R128 normalization.

The run starts from the epoch 367 model weights. The run uses new optimizer, scheduler, epoch, and step states. The target is 100,000 new optimizer steps.

## Data and Frontend Gates

| Gate | Status | Evidence |
|---|---|---|
| The corpus has 207,505 accepted utterances. | PASS | `expanded_v4_trim_only_full_dataset.json` |
| The accepted duration is 467.127 hours. | PASS | `expanded_v4_trim_only_full_analysis.json` |
| The run excludes 1,357 clipped utterances and 5 short utterances. | PASS | `expanded_v4_trim_only_full_dataset.json` |
| The run has no source-group split leakage. | PASS | `expanded_v4_trim_only_full_analysis.json` |
| The training WAV files are mono PCM WAV at 24 kHz. | PASS | `expanded_v4_trim_only_full_validation.json` |
| The speaker embeddings have 103,752 pre-trim vectors and 103,753 post-trim vectors. | PASS | `expanded_v4_trim_only_full_hybrid_embeddings.json` |
| Each accepted utterance has one 192-value speaker vector. | PASS | `expanded_v4_trim_only_full_hybrid_embeddings.json` |
| The token list has 138 tokens and matches the epoch 367 token list. | PASS | `expanded_v4_trim_only_full_scale_readiness.json` |
| Speech, pitch, and energy statistics exist. | PASS | `expanded_v4_trim_only_full_scale_readiness.json` |

All paths in this table are relative to `training/reports/`.

## Model Gates

| Gate | Status | Evidence |
|---|---|---|
| The dual-GPU 100-step smoke run completed. | PASS | `expanded_v4_trim_only_smoke_checkpoint.json` |
| The smoke checkpoint has finite model values. | PASS | `expanded_v4_trim_only_smoke_checkpoint.json` |
| The smoke inference made 32 valid WAV files. | PASS | `expanded_v4_trim_only_smoke_inference.json` |
| The full run made the 25K milestone. | PASS | `expanded_v4_trim_only_25k_checkpoint.json` |
| The full run made and audited the 50K milestone. | PASS | `expanded_v4_trim_only_50k_checkpoint.json` |
| The full run made and audited the 75K milestone. | PASS | `expanded_v4_trim_only_75k_checkpoint.json` |
| The full run completed 100,000 new optimizer steps. | IN PROGRESS | The full training process is active. |
| The 100K checkpoint has finite model values. | NOT RUN | This gate runs after step 100,000. |
| The fixed evaluation set has valid 24 kHz mono WAV files. | NOT RUN | This gate runs after training. |
| The five fixed voices have valid WAV files and model hashes. | NOT RUN | This gate runs after training. |

## Runtime Gates

| Gate | Status | Evidence |
|---|---|---|
| Both RTX 3090 GPUs are in the run. | PASS | `training_status_expanded_v4_trim_only_100k.jsonl` |
| The monitor interval is not more than 30 minutes. | PASS TO DATE | `expanded_v4_trim_only_training_monitor_summary.json` |
| The runtime log has no OOM, NaN, or critical error. | PASS TO DATE | `training_status_expanded_v4_trim_only_100k.jsonl` |
| Free disk space stays above 30 GiB. | PASS TO DATE | `expanded_v4_trim_only_disk.jsonl` |

## Completion Rule

Do not mark this run complete until all `IN PROGRESS` and `NOT RUN` gates have a final result. After training, run the fixed evaluation set and the five-voice evaluation. Then update this checklist and `final_status.md`.
