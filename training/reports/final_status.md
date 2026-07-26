# Final status

## Виконано

The expanded-v3 source policy, clean-audio process, hybrid embedding process,
NeMo pseudo-label process, ESPnet recipe, readiness gate, monitor, and 25K
launcher exist under `training/`.

The data-preparation cleanup trigger is 60 GiB. The process does not remove a
completed source batch above this trigger. The training stop threshold is 60
GiB. The warning threshold is 80 GiB. The monitor records GPU power and
calculates the ETA in Kyiv time. The maximum status interval is five minutes.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS for a real 360-record source sample.
- Data Preparation: PASS for 318 retained clean records.
- Modeling: PASS for 100 fresh dual-GPU iterations.
- Evaluation: PASS for 31 of 31 WAV files.
- Deployment: PASS for the existing local inference entry point.

## MVP-gate

| Gate | Status |
|---|---|
| Source policy | PASS |
| Frontend regression | PASS |
| Non-empty phonemes | PASS |
| Clean smoke data | PASS |
| Exact 50/50 hybrid embeddings | PASS |
| ESPnet token list | PASS |
| Pitch and energy statistics | PASS |
| JETS construction | PASS |
| Dual-GPU smoke training | PASS |
| Checkpoint | PASS |
| Valid 24 kHz mono WAV | PASS |
| Full enabled-source coverage | NOT RUN |
| Full NeMo pass | NOT RUN |
| Fresh 25K training | NOT RUN |
| Five-voice 25K evaluation | NOT RUN |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `prepare_expanded_v3.sh smoke` | 0 | 318 clean records; token list and statistics exist | `reports/expanded_v3_smoke_validation.json` |
| `run_expanded_v3_smoke.sh` | 0 | 100 iterations; checkpoint exists | `exp_expanded_v3_smoke/tts_jets_uk_24k_expanded_v3_smoke/1epoch.pth` |
| Expanded smoke inference | 0 | 31 of 31 WAV files passed | `reports/expanded_v3_smoke_inference.json` |
| Full training test suite | 0 | 63 tests passed | `tests/` |

## Створені артефакти

- Manifests: `data/expanded_v3_smoke/manifests/`.
- ESPnet data: `espnet_recipe/data/expanded_v3_smoke_*`.
- Configuration: `conf/expanded_v3_sources.yaml`.
- Token list: `dump_expanded_v3_smoke/token_list/`.
- Statistics: `exp_expanded_v3_smoke/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Checkpoint: `exp_expanded_v3_smoke/tts_jets_uk_24k_expanded_v3_smoke/1epoch.pth`.
- WAV files: `exp_expanded_v3_smoke/tts_jets_uk_24k_expanded_v3_smoke/decode_jets_train.total_count.ave/`.
- Reports: `reports/expanded_v3*.json` and `reports/expanded_v3.md`.

## Відомі проблеми

- The direct Common Voice 26 directory is not present.
- The `speech-uk/voice-of-america` data files are not accessible.
- Per-file sources do not have a completed license manifest.
- The full NeMo environment and full unlabeled pass did not run.
- Human evaluation of metallic sound did not run for a new 25K checkpoint.

## Наступна одна дія

Download the direct Common Voice archive. Then run:

```sh
MDC_COMMON_VOICE_ROOT=/path/to/extracted/uk \
  training/scripts/prepare_expanded_v3.sh full
```

Do not start `launch_expanded_v3_training.sh` until the full readiness report
has `PASS`.
