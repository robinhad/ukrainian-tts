# Scale readiness

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Host resource preflight | PASS | PyTorch 2.9.1+cu128 sees GPU 0; 2x RTX 3090 idle, 122.25 GiB RAM and 329.17 GiB disk available | `reports/resource_usage.jsonl` (runtime artifact) | Recheck immediately before training |
| Frontend tests | PASS | `18 passed in 0.80s` on 2026-07-22 | `tests/`, `tests/frontend/expected_phonemes.json` | Preserve snapshot in later stages |
| eSpeak version/data hash pinned | PASS | eSpeak-ng 1.52.0; data hash `924ed10e...aa80d6d` | `vendor/ESPEAK_NG_VERSION`, `vendor/ESPEAK_NG_DATA_HASH` | Reject version/hash drift |
| No empty phoneme sequences | PASS | All 50 regression inputs returned deterministic non-empty tokens | `tests/frontend/expected_phonemes.json` | Repeat after any frontend change |
| ESPnet data directories valid | NOT RUN | Builder implemented | `espnet_recipe/local/` | Prepare smoke data |
| Token list created | NOT RUN | Recipe configured | `dump/` | Run ESPnet stage 5 |
| Pitch/energy statistics created | NOT RUN | Recipe configured | `exp/` | Run ESPnet stage 6 |
| JETS model initializes | NOT RUN | Config implemented | `conf/tuning/` | Run model construction |
| Training iterations without NaN | NOT RUN | — | `exp/` | Run 100-iteration smoke |
| Checkpoint saved | NOT RUN | — | `exp/` | Complete smoke epoch |
| Valid 24 kHz mono WAV | NOT RUN | Validator implemented | `eval/generated/` | Run stage 8 inference |
| Reproduction commands documented | PASS | Bootstrap, stage and inference commands | `README.md` | Keep actual commands in log |

Scaling is forbidden until every critical row is `PASS`.
