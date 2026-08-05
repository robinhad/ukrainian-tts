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
| The full run completed 100,000 new optimizer steps. | PASS | `expanded_v4_trim_only_100k_checkpoint.json` |
| The 100K checkpoint has finite model values. | PASS | The audit found 549 finite model tensors. |
| The 100K milestone matches the checkpoint model. | PASS | The audit found no missing, unexpected, or different model values. |
| The fixed evaluation set has valid 24 kHz mono WAV files. | PASS | The 25K, 50K, 75K, and 100K reports each contain 1,400 valid WAV files. |
| The five fixed voices have valid WAV files and model hashes. | PASS | `five_voice_expanded_v4_trim_only_100k.json` |
| A person checked speech quality. | NOT RUN | The user must listen to the five 100K WAV files. |

## Runtime Gates

| Gate | Status | Evidence |
|---|---|---|
| Both RTX 3090 GPUs are in the run. | PASS | `training_status_expanded_v4_trim_only_100k.jsonl` |
| The monitor interval is not more than 30 minutes. | PASS | The largest measured interval is 15.011 minutes. |
| The runtime log has no OOM, NaN, or critical error. | PASS | The monitor found zero error matches and zero critical samples. |
| Free disk space stays above 30 GiB. | PASS | The minimum measured free space is 37.45 GiB. |
| GPU temperatures stay below 90 degrees C. | PASS | The measured maximum values are 87 and 78 degrees C. |
| The full test suite passes. | PASS | All 101 tests passed on 2026-08-06. |

## Completion Rule

The training and automatic evaluation are complete. Human listening is not complete. Do not promote this model as a release until a person checks the five listening WAV files.
