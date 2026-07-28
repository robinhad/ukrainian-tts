# Final status

## Виконано

The historical expanded-v3 pipeline completed the 25K listening gate. A later
source-duration audit found that this run used only 2.981 VOA hours. The
intended minimum is 800 hours. Therefore, the historical model is not the
intended large-corpus model.

The cleanup trigger was 60 GiB. The data collector removed completed source
batches only when the free space reached this trigger. The training monitor
kept a minimum reserve of 75.49 GiB.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS.
- Data Preparation: PASS.
- Modeling: PASS.
- Evaluation: PASS.
- Deployment: PASS.

The full data has 76,578 utterances and 95.260 hours. The train, development,
and evaluation splits have 73,755, 1,403, and 1,420 utterances.

## MVP-gate

| Gate | Status |
|---|---|
| Source policy | PASS |
| Frontend regression | PASS |
| Non-empty phonemes | PASS |
| Clean full data | FAIL |
| VOA retained duration | FAIL |
| Exact 50/50 hybrid embeddings | PASS |
| ESPnet token list | PASS |
| Pitch and energy statistics | PASS |
| JETS construction | PASS |
| Dual-GPU training | PASS |
| No NaN or OOM | PASS |
| 25K checkpoint | PASS |
| Five valid 24 kHz mono WAV files | PASS |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `prepare_expanded_v3.sh full` and resume stages | 0 | 76,578 clean records passed the final gate | `reports/expanded_v3_full_scale_readiness.json` |
| `run_expanded_v3.sh --stage 4 --stop_stage 6` | 0 | The token list and all statistics exist | `dump_expanded_v3/`, `exp_expanded_v3/tts_stats_raw_phn_espeak_ng_ukrainian/` |
| ESPnet stage 7 | 0 | 25,000 iterations completed at 03:43 EEST | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k/train.log` |
| Initial automatic five-voice step | 1 | The selector read the wrong aggregate archive | `reports/expanded_v3_25k_launcher.log` |
| `MODEL_FILE=milestones/25k.pth generate_five_voice_expanded_v3_eval.sh` | 0 | Five of five WAV files passed | `reports/five_voice_expanded_v3_25k.json` |
| `pytest -q training/tests` | 0 | 81 tests passed | `tests/` |
| Revised expanded-v3 readiness audit | 1 | The gate correctly failed at 2.981 of 800 required VOA hours | `reports/expanded_v3_full_scale_readiness.json` |

## Створені артефакти

- Manifests: `data/expanded_v3/manifests/`.
- ESPnet data: `espnet_recipe/data/expanded_v3_*`.
- Configuration: `conf/expanded_v3_sources.yaml`.
- Token list: `dump_expanded_v3/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `exp_expanded_v3/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Checkpoint: `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k/milestones/25k.pth`.
- Listening WAV files: `eval/generated/five_voice_expanded_v3_25k/`.
- Evaluation report: `reports/five_voice_expanded_v3_25k.json`.
- Runtime reports: `reports/expanded_v3_full_*.json` and
  `reports/training_status_expanded_v3.jsonl`.

The checkpoint SHA-256 is
`3c2882f692a0873f89b31a643db36748c12de7e1f8157cfe9e87ac4f27c70c2c`.

## Відомі проблеми

- The current manifest has 2.981 VOA hours. The minimum is 800 hours.
- The old process rejected 757,829 duration-eligible candidates at confidence
  0.80.
- The earlier model had a metallic timbre in the user listening check.
- The new 25K five-voice set does not have a human listening result.
- GPU 0 reached 86 degrees C. Check cooling before a longer run.
- Flash Attention is not installed. ESPnet used its standard attention code.

## Наступна одна дія

Run the versioned VOA pseudo-label pipeline with the 0.50 confidence threshold,
the 20-second maximum, and the 0.5-second same-speaker merge gap. Do not start
another long training run until the manifest has at least 800 VOA hours.
