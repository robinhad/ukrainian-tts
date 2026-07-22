# Scale readiness

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Host resource preflight | PASS | PyTorch 2.9.1+cu128 sees GPU 0; 2x RTX 3090 idle, 122.25 GiB RAM and 329.17 GiB disk available | `reports/resource_usage.jsonl` (runtime artifact) | Recheck immediately before training |
| Frontend tests | PASS | `18 passed in 0.80s` on 2026-07-22 | `tests/`, `tests/frontend/expected_phonemes.json` | Preserve snapshot in later stages |
| eSpeak version/data hash pinned | PASS | eSpeak-ng 1.52.0; data hash `924ed10e...aa80d6d` | `vendor/ESPEAK_NG_VERSION`, `vendor/ESPEAK_NG_DATA_HASH` | Reject version/hash drift |
| No empty phoneme sequences | PASS | All 50 regression inputs returned deterministic non-empty tokens | `tests/frontend/expected_phonemes.json` | Repeat after any frontend change |
| ESPnet data directories valid | PASS | ESPnet validators kept 256 train, 32 dev and 32 eval utterances | `espnet_recipe/data/smoke_{train,dev,eval}/` | Regenerate from manifests on a new host |
| Token list created | PASS | 153 tokens; ESPnet reported OOV rate 0.0% | `dump/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve with checkpoint |
| Pitch/energy statistics created | PASS | Stage 6 completed and produced train/valid global-MVN NPZ files | `exp/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve with checkpoint |
| JETS model initializes | PASS | Dry run created 83.31M-parameter FP32 JETS plus two AdamW optimizers | `exp/tts_train_jets_uk_24k_raw_phn_espeak_ng_ukrainian_dry_runtrue/train.log` | Start 100-iteration smoke training |
| Training iterations without NaN | NOT RUN | — | `exp/` | Run 100-iteration smoke |
| Checkpoint saved | NOT RUN | — | `exp/` | Complete smoke epoch |
| Valid 24 kHz mono WAV | NOT RUN | Validator implemented | `eval/generated/` | Run stage 8 inference |
| Reproduction commands documented | PASS | Bootstrap, stage and inference commands | `README.md` | Keep actual commands in log |

Scaling is forbidden until every critical row is `PASS`.
