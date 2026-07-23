# Scale readiness

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Host resource preflight | PASS | PyTorch 2.9.1+cu128 passed a CUDA check on both RTX 3090 GPUs; the post-run check found 122.06 GiB RAM and 314.79 GiB disk available | `reports/resource_usage.jsonl` (runtime artifact) | Recheck immediately before the next training run |
| Frontend tests | PASS | `21 passed in 5.18s` on 2026-07-23, including pinned-version mismatch rejection | `tests/`, `tests/frontend/expected_phonemes.json` | Preserve the snapshot in later stages |
| eSpeak version/data hash pinned | PASS | eSpeak-ng 1.52.0; data hash `924ed10e...aa80d6d` | `vendor/ESPEAK_NG_VERSION`, `vendor/ESPEAK_NG_DATA_HASH` | Reject version/hash drift |
| No empty phoneme sequences | PASS | All 50 regression inputs returned deterministic non-empty tokens | `tests/frontend/expected_phonemes.json` | Repeat after any frontend change |
| ESPnet data directories valid | PASS | ESPnet validators kept 256 train, 32 dev and 32 eval utterances | `espnet_recipe/data/smoke_{train,dev,eval}/` | Regenerate from manifests on a new host |
| Token list created | PASS | 153 tokens; ESPnet reported OOV rate 0.0% | `dump/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve with checkpoint |
| Pitch/energy statistics created | PASS | Stage 6 completed and produced train/valid global-MVN NPZ files | `exp/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve with checkpoint |
| JETS model initializes | PASS | Dry run created 83.31M-parameter FP32 JETS plus two AdamW optimizers | `exp/tts_train_jets_uk_24k_raw_phn_espeak_ng_ukrainian_dry_runtrue/train.log` | Start 100-iteration smoke training |
| Training iterations without NaN | PASS | 100 train iterations plus 5 validation batches; generator, discriminator and alignment losses finite; peak cache 5.938 GiB | `exp/tts_train_jets_uk_24k_raw_phn_espeak_ng_ukrainian_max_epoch1/train.log` | Preserve log and calibrate full run separately |
| Checkpoint saved | PASS | 1-epoch model SHA-256 `6092bf24...c02685c` | `exp/tts_train_jets_uk_24k_raw_phn_espeak_ng_ukrainian_max_epoch1/1epoch.pth` | Keep runtime artifact off Git |
| Valid 24 kHz mono WAV | PASS | 32/32 eval WAVs passed finite/nonzero/mono/24 kHz checks; duration 1.65--4.10 s; median RTF 0.0185 | `reports/smoke_inference.json`, `eval/generated/example.wav` | Listening quality is intentionally outside the MVP gate |
| Reproduction commands documented | PASS | Bootstrap, stage and inference commands | `README.md` | Keep actual commands in log |

Every critical MVP row is `PASS`. Cycle 2 preparation and batch calibration are
permitted.
