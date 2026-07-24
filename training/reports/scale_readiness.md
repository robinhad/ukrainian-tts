# Scale readiness

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Host resource preflight | PASS | PyTorch 2.9.1+cu128 passed a CUDA check on both RTX 3090 GPUs; the post-run check found 122.06 GiB RAM and 314.79 GiB disk available | `reports/resource_usage.jsonl` (runtime artifact) | Recheck immediately before the next training run |
| Frontend tests | PASS | Final suite: `28 passed in 5.55s`, including pinned-version mismatch rejection | `tests/`, `tests/frontend/expected_phonemes.json` | Preserve the snapshot in later stages |
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

## Silence-trimmed full-run gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Silence trim | PASS | 6,787 of 6,787 files processed; 4.778 h removed; raw source unchanged | `reports/full_trimmed_dataset.json` | Keep the trim configuration fixed |
| Full split | PASS | 6,461 train, 146 development, and 180 evaluation files; no exact leakage | `data/full_trimmed/manifests/` | Preserve manifests with the model |
| Tokenization and statistics | PASS | 0.0% OOV; speech, pitch, and energy statistics exist | `dump_full_trimmed/`, `exp_full_trimmed/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve with the checkpoint |
| Dual-GPU training | PASS | 25,000 iterations; exit 0; no NaN, OOM, or critical runtime error | `exp_full_trimmed/tts_jets_uk_24k_trimmed/train.log` | Use listening results for checkpoint selection |
| Loss trend | PASS | Validation mel loss fell from 56.198 at 1k to 33.549 at 25k | TensorBoard validation events | Review speech quality |
| Memory reserve | PASS | Final peak cache was 22.178 GiB per process after the 3.8M restart | Training log | Keep 3.8M unless the host changes |
| Power monitor | PASS | 140 samples; maximum power 264.08 W and 261.39 W | `reports/power_summary_trimmed.json` | Recheck cooling before another long run |
| Final checkpoint | PASS | SHA-256 `58f46736...f9d75` | `exp_full_trimmed/milestones/25k.pth` | Preserve off Git |
| Fixed-set inference | PASS | 180 of 180 WAV files; 24 kHz mono; no clipping warning | `reports/full_trimmed_inference_25k.json` | Complete listening review |
| Local inference | PASS | 3.189 s WAV; RTF 0.10438; metadata written | `eval/generated/example_trimmed_25k.wav` | Use this entry point for local tests |

All available gates for the silence-trimmed 25k run are `PASS`.

## Common Voice and Lada full-run gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Full source manifest | PASS | 76,762 Common Voice and 6,787 Lada records; six embedded-metadata transcriptions and 33 cross-source duplicate texts removed | `data/multispeaker_full/source_records.jsonl` | Keep the source revisions and text limits fixed |
| Silence trim | PASS | 83,549 retained model copies; 84.871 h after trim; raw files unchanged | `reports/multispeaker_full_dataset.json` | Keep the trim configuration fixed |
| Split validation | PASS | 80,041 train, 1,831 development, and 1,677 evaluation records; no validation errors; maximum phoneme length 140 | `data/multispeaker_full/manifests/` | Preserve the manifest hashes |
| Frontend regression | PASS | 50 training tests; 50-case and 500-case snapshots; no joined punctuation token; extreme text guard | `tests/frontend/`, `tests/test_validate_dataset.py` | Reject frontend or source-text drift |
| Token list | PASS | 87 lines and 0.0 percent OOV | `dump_multispeaker_full/token_list/phn_espeak_ng_ukrainian/tokens.txt` | Preserve with the model |
| Speaker embeddings | PASS | 83,549 retained finite, nonzero 192-D ECAPA vectors | `dump_multispeaker_full/xvector/` | Preserve the pinned ECAPA revision |
| Pitch and energy statistics | PASS | 80,041 train and 1,831 development shapes; six finite NPZ files | `exp_multispeaker_full/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve with the model |
| Rejected first long run | FAIL | Batch 941--950 caused text-attention OOM; six source rows had 10,763--244,054 phonemes; no checkpoint existed | `exp_multispeaker_full/tts_jets_uk_24k_multispeaker_rejected_corrupt_text_4500000/` | Use only the corrected data |
| Corrected dual-GPU 1k gate | PASS | Exit 0; 1,000 train iterations and 45 validation batches; finite validation generator loss 77.467 and mel loss 59.788 | `exp_multispeaker_full/milestones/1k.pth` | Resume from the verified checkpoint |
| 4.5M memory reserve | FAIL | Peak cached VRAM was 22.684 GiB, which leaves less than 10 percent free | `reports/training_status_multispeaker.jsonl` | Use a smaller setting |
| 3.8M, 3.4M, and 3.0M reserve | FAIL | Late dynamic batches left less than 10 percent CUDA memory free | `exp_multispeaker_full/tts_jets_uk_24k_multispeaker/train_rejected_reserve_*.log` | Use 2.5M |
| 2.5M memory reserve | PASS | Epoch 2 completed; peak cached VRAM was 19.416 GiB; validation generator loss was 70.997 | `exp_multispeaker_full/tts_jets_uk_24k_multispeaker/2epoch.pth` | Keep 2.5M |
| Long training | RUNNING | Epoch 2 completed without NaN or OOM; epoch 3 started | `scripts/run_multispeaker_training.sh` | Continue to 25k |
| Full fixed-set inference | NOT RUN | A long-run checkpoint does not exist yet | `scripts/run_multispeaker_milestone_inference.sh` | Run after a milestone checkpoint |

All corrected data, one-epoch training, and 2.5M memory gates are `PASS`.
Larger batch settings failed the memory-reserve gate. The 25k continuation uses
2.5M. Dmytro remains an inference-only zero-shot target because no raw Dmytro
corpus is available.
