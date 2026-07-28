# Scale readiness

## Expanded-v3 gate

The historical 25K training and the five-voice automatic evaluation are
complete. The revised expanded-v3 full gate has `FAIL` because the manifest
contains 2.981 VOA hours. The minimum is 800 hours.

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Disk reserve | PASS | The minimum monitored reserve was 75.49 GiB; the stop limit was 60 GiB | `reports/training_status_expanded_v3.jsonl` | Keep the 60 GiB limit |
| Source policy | PASS | The default-deny registry passed | `conf/expanded_v3_sources.yaml` | Keep each revision and license fixed |
| Expanded smoke data | PASS | 318 clean files from five sources | `reports/expanded_v3_smoke_validation.json` | Preserve the smoke evidence |
| Full data | FAIL | 76,578 records passed file validation, but the corpus does not meet the VOA duration requirement | `reports/expanded_v3_full_validation.json` | Rebuild the VOA pseudo-label data |
| Enabled source coverage | PASS | All nine permitted source groups are present | `reports/expanded_v3_full_sources.json` | Keep the source revisions fixed |
| VOA retained duration | FAIL | The manifest has 2.981 hours; the required minimum is 800 hours | `data/expanded_v3/manifests/all.parquet` | Run pseudo-label pipeline version `v3-c050-d20-g050-duration-split` |
| Hybrid embeddings | PASS | 38,289 raw and 38,289 clean vectors are finite 192-value vectors | `reports/expanded_v3_full_hybrid_embeddings.json` | Preserve the ECAPA revision |
| Token list and statistics | PASS | The token list has 110 lines; pitch and energy statistics exist | `dump_expanded_v3/`, `exp_expanded_v3/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve these files with the model |
| Dual-GPU smoke training | PASS | 100 finite iterations made a checkpoint | `exp_expanded_v3_smoke/` | Keep runtime files outside Git |
| Full readiness | FAIL | The new source-duration gate rejects the current manifest | `reports/expanded_v3_full_scale_readiness.json` | Do not start another long training run |
| Fresh expanded 25K run | PASS | Training ended at 25,000 iterations with exit 0; no NaN or OOM occurred | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k/train.log` | Use listening results for the next decision |
| GPU use | PASS | Both RTX 3090 GPUs reached 100 percent compute use; peak power was 241.20 W and 234.98 W | `reports/training_status_expanded_v3.jsonl` | Check GPU 0 cooling before a longer run |
| Checkpoint | PASS | The epoch and milestone files are identical; SHA-256 starts with `3c2882f6` | `exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k/milestones/25k.pth` | Preserve the file outside Git |
| Five-voice evaluation | PASS | Five WAV files are non-empty, finite, mono, and 24 kHz | `reports/five_voice_expanded_v3_25k.json` | Do the human listening check |

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
| 3.8M, 3.4M, 3.0M, and 2.5M reserve | FAIL | Late or later-epoch dynamic batches left less than 10 percent CUDA memory free | `exp_multispeaker_full/tts_jets_uk_24k_multispeaker/train_rejected_reserve_*.log` | Use 2.0M and expandable segments |
| 2.0M memory reserve | PASS | Epoch 3 completed; peak cached VRAM was 12.562 GiB; validation generator loss was 65.770 | `exp_multispeaker_full/tts_jets_uk_24k_multispeaker/3epoch.pth` | Keep 2.0M and expandable segments |
| Long training | PASS | Exit 0 after 25,000 iterations; no NaN, OOM, or critical error; final validation generator loss 58.242 and mel loss 41.292 | `exp_multispeaker_full/tts_jets_uk_24k_multispeaker/train.log` | Use listening results for checkpoint selection |
| Power and memory | PASS | Peak cached memory 15.500 GiB; sampled maximum power 244.29 W and 254.45 W; both GPUs reached 100 percent compute use | `reports/power_summary_multispeaker.json` | Keep 2.0M batch bins on this host |
| Final checkpoint | PASS | The 25k checkpoint and milestone copy are byte-identical; SHA-256 `395ccaba...f445` | `exp_multispeaker_full/milestones/25k.pth` | Preserve outside Git |
| Full fixed-set inference | PASS | The 1k, 5k, and 25k checkpoints each made 1,677 valid 24 kHz mono WAV files; no clipping warning | `reports/multispeaker_full_inference_{1k,5k,25k}.json` | Complete listening review |
| Local speaker inference | PASS | Lada and Dmytro zero-shot WAV files are finite, non-empty, mono, and 24 kHz | `eval/generated/multispeaker_25k_{lada,dmytro_zero_shot}.wav` | Do not describe Dmytro as a trained speaker |
| Release candidate | PASS | The release has 34 files, cards, reports, examples, licenses, and checksums | `releases/uk-tts-jets-multispeaker-25k-rc/` | Promote only after listening review |

All available gates for the corrected Common Voice and Lada run are `PASS`.
Larger batch settings failed the memory-reserve gate. The completed 25k run
used 2.0M batch bins and expandable allocator segments. Dmytro remains an
inference-only zero-shot target because no raw Dmytro corpus is available.

## Enhanced audio v2 gate

| Gate | Status | Evidence | Artifact | Next action |
|---|---|---|---|---|
| Resource check | PASS | Two RTX 3090 GPUs and more than 122 GiB available RAM | `reports/resource_usage.jsonl` | Keep both GPUs available |
| DeepFilterNet3 | PASS | DeepFilterNet 0.5.6 processed the source audio with the pinned DeepFilterNet3 model | `reports/multispeaker_enhanced_v2_dataset.json` | Keep the model hashes fixed |
| Conservative mastering | PASS | The process used a 70 Hz high-pass filter, light de-essing, and 1.5 ratio compression | `reports/multispeaker_enhanced_v2.md` | Do not use these operations on a different run without a new data version |
| Two-pass EBU R128 | PASS | 82,662 files are within 1 LU of -23 LUFS; maximum true peak is -1 dBTP | `reports/multispeaker_enhanced_v2_validation.json` | Keep rejected files out of training |
| Split validation | PASS | 79,188 train, 1,815 development, and 1,659 evaluation files; no leakage | `data/multispeaker_enhanced_v2/manifests/` | Preserve manifest hashes |
| Token list and statistics | PASS | 87 tokens, 0.0 percent OOV, and finite speech, pitch, and energy statistics | `exp_multispeaker_enhanced_v2/tts_stats_raw_phn_espeak_ng_ukrainian/` | Preserve with the model |
| Smoke training | PASS | 100 dual-GPU iterations, checkpoint, and 1,659 valid WAV files | `reports/multispeaker_enhanced_v2_smoke_eval.json` | Full training was permitted |
| Full training | PASS | 100,000 total FP32 iterations; resumed from 25k; exit 0; no NaN, OOM, or critical error | `exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/train.log` | Use listening results for model selection |
| Loss trend | PASS | Best validation generator loss is 56.459 and best mel loss is 38.897 at 97k | `reports/tensorboard_metrics_multispeaker_enhanced_v2_100k.json` | Compare audio, not loss alone |
| Power and memory | PASS | Maximum cached memory was 20.670 GiB; sampled peak power was 231.31 W and 241.60 W | `reports/power_summary_multispeaker_enhanced_v2_100k.json` | Improve GPU0 cooling before another long run |
| 100k checkpoint | PASS | Epoch 100 and averaged five-best checkpoints exist; averaged SHA-256 starts with `7ad91344` | `exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/` | Preserve the files outside Git |
| Full inference | PASS | 1,659 of 1,659 WAV files passed; median RTF is 0.00799; no clipping warning | `reports/multispeaker_enhanced_v2_inference_100k.json` | Complete the listening check |
| Five-voice set | PASS | Five of five 100k WAV files passed automatic checks | `reports/five_voice_enhanced_v2_100k.json` | Listen for metallic timbre |
| Release package | NOT RUN | The model and evaluation artifacts exist, but no enhanced-v2 release package was made | `exp_multispeaker_enhanced_v2/` | Package only after the listening check |

All training and automatic evaluation gates are `PASS`. Release promotion is
not permitted until a human listening check is complete.
