# Scale readiness

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Host resource preflight | PASS | 2x RTX 3090 idle; RAM/disk thresholds pass | console preflight | Run bootstrapped PyTorch probe |
| Frontend tests | NOT RUN | Dependencies not bootstrapped | `tests/frontend/` | Bootstrap environment |
| eSpeak version/data hash pinned | NOT RUN | Source pins committed | `vendor/` | Build and record data hash |
| No empty phoneme sequences | NOT RUN | Test implemented | `tests/frontend/` | Run regression suite |
| ESPnet data directories valid | NOT RUN | Builder implemented | `espnet_recipe/local/` | Prepare smoke data |
| Token list created | NOT RUN | Recipe configured | `dump/` | Run ESPnet stage 5 |
| Pitch/energy statistics created | NOT RUN | Recipe configured | `exp/` | Run ESPnet stage 6 |
| JETS model initializes | NOT RUN | Config implemented | `conf/tuning/` | Run model construction |
| Training iterations without NaN | NOT RUN | — | `exp/` | Run 100-iteration smoke |
| Checkpoint saved | NOT RUN | — | `exp/` | Complete smoke epoch |
| Valid 24 kHz mono WAV | NOT RUN | Validator implemented | `eval/generated/` | Run stage 8 inference |
| Reproduction commands documented | PASS | Bootstrap, stage and inference commands | `README.md` | Keep actual commands in log |

Scaling is forbidden until every critical row is `PASS`.
