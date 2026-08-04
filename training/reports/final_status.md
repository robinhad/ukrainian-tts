# Final status

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify the report.

## Виконано

The expanded-v3 pipeline prepared 208,867 clean utterances. The total duration
is 469.755 hours. The training manifest contains segments from 2 to 20 seconds.
It does not contain the deferred segments that are longer than 20 seconds.

The pipeline made 104,433 speaker embeddings from source audio. It made
104,434 speaker embeddings from clean audio. All vectors have 192 values.

ESPnet made a 138-token list and all pitch and energy statistics. JETS used two
RTX 3090 GPUs and completed 500,000 iterations. Training ended with exit code
0 at 17:46 Kyiv time on 2026-08-04. The launcher ended with exit code 0 at
17:50 Kyiv time.

The finalizer made 1,418 fixed-set WAV files and five listening WAV files. All
1,423 files passed the automatic audio checks. The files are mono, 24 kHz,
finite, non-empty, and below digital clipping.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS.
- Data Preparation: PASS.
- Modeling: PASS.
- Evaluation: PASS.
- Deployment: PASS.

The smoke cycle proved the complete technical path before the long run.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Source policy | PASS | The default-deny source review passed. |
| Clean training data | PASS | 208,867 records and 469.755 hours passed validation. |
| Segment duration | PASS | The minimum is 2.0 seconds. The maximum is 20.0 seconds. |
| Split isolation | PASS | No source-group leakage or duplicate audio hash exists. |
| Hybrid speaker embeddings | PASS | 104,433 raw and 104,434 clean vectors exist. |
| Frontend and phonemes | PASS | eSpeak-ng 1.52.0 made non-empty phonemes. |
| Token list | PASS | The token list has 138 entries and 0 percent OOV. |
| Pitch and energy statistics | PASS | ESPnet made all required statistics. |
| Dual-GPU training | PASS | JETS completed 500,000 iterations with exit code 0. |
| Finite losses | PASS | The log has no NaN, OOM, or critical runtime error. |
| 500K checkpoint | PASS | The epoch file and milestone file are byte-identical. |
| Fixed-set inference | PASS | 1,418 of 1,418 WAV files passed. |
| Five-voice inference | PASS | Five of five WAV files passed. |
| Checksums | PASS | The final checksum file passes `sha256sum -c`. |
| Old 800-hour VOA target | FAIL | The accepted VOA subset has 377.489 hours. The user authorized this run on the available data. |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `process_voa_streaming.sh` | 0 | The pipeline accepted 134,711 VOA records. | `data/expanded_v3/manifests/all.parquet` |
| `run_expanded_v3.sh --stage 4 --stop_stage 6 --nj 10` | 0 | ESPnet made the token list and statistics. | `dump_expanded_v3/`, `exp_expanded_v3/tts_stats_raw_phn_espeak_ng_ukrainian/` |
| `launch_expanded_v3_training.sh 500000` | 0 | JETS completed 500,000 iterations. | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/train.log` |
| `evaluate_expanded_v3_milestone.sh 500k` | 0 | 1,418 fixed-set WAV files passed. | `reports/expanded_v3_inference_500k.json` |
| `generate_five_voice_expanded_v3_eval.sh` | 0 | Five listening WAV files passed. | `reports/five_voice_expanded_v3_500k.json` |
| `finalize_expanded_v3_500k.sh` | 0 | Both evaluations and all checksums completed. | `reports/expanded_v3_500k_artifacts.sha256` |
| Independent WAV audit | 0 | All 1,423 WAV files passed a second check. | `eval/generated/five_voice_expanded_v3_500k/` |
| `python -m pytest -q training/tests` | 0 | All 94 tests passed in 7.93 seconds. | `tests/` |

## Training metrics

The last validation generator loss is 57.506. The last validation mel loss is
40.105. The last validation alignment loss is 4.321. The last validation
discriminator loss is 2.195.

The best validation generator loss is 56.028 at epoch 367. The best validation
mel loss is 39.860 at epoch 443. The best validation alignment loss is 4.316
at epochs 488 and 492. The pipeline kept all of these checkpoints.

The sampled peak power was 243.70 W on GPU 0 and 241.63 W on GPU 1. Both GPUs
reached 100 percent compute use. The sampled peak VRAM was 23,948 MiB on GPU 0
and 24,062 MiB on GPU 1. The training log reports 23.062 GiB of peak cached
memory. GPU 0 reached 88 degrees C. GPU 1 reached 78 degrees C.

The minimum reported free disk space was 48.93 GiB. This value stayed above
the 30 GiB cleanup threshold.

## Створені артефакти

- Manifest: `data/expanded_v3/manifests/all.parquet`.
- ESPnet data: `espnet_recipe/data/expanded_v3_*`.
- Token list: `dump_expanded_v3/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `exp_expanded_v3/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Final checkpoint: `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/milestones/500k.pth`.
- Validation-best checkpoints: `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k/best_checkpoints/`.
- Fixed evaluation: `reports/expanded_v3_inference_500k.json`.
- Listening evaluation: `reports/five_voice_expanded_v3_500k.json`.
- Listening WAV files: `eval/generated/five_voice_expanded_v3_500k/`.
- Checksums: `reports/expanded_v3_500k_artifacts.sha256`.
- Runtime status: `reports/training_status_expanded_v3_500k.jsonl`.

The final checkpoint SHA-256 is
`9987ed08937235abd14be9519628312441e5aefdd3d19a584a54bc623668063d`.

## Відомі проблеми

- The old VOA target is 800 hours. The accepted VOA subset has 377.489 hours.
- The user reported metallic sound in an earlier model. The 500K model needs a
  human listening check.
- GPU 0 reached 88 degrees C during training.
- Flash Attention is not installed. ESPnet used its standard attention code.
- The automatic tests do not measure naturalness or speaker preference.

## Наступна одна дія

Listen to the five files in `eval/generated/five_voice_expanded_v3_500k/`.
Select the best voice before you make a release candidate.
