# Final Status

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

## Виконано

The expanded-v8 pipeline completed at 15:45:47 Kyiv time on 2026-08-18. It
processed 74,156 utterances. The total duration is 92.266 hours. The training
split has 71,334 utterances and 89.223 hours. The development split has 1,404
utterances and 1.514 hours. The evaluation split has 1,418 utterances and
1.529 hours. The corpus has no VOA records.

The source breakdown is:

| Source | Records | Hours |
|---|---:|---:|
| Common Voice 22 | 39,905 | 42.899 |
| Ukrainian dialects | 24,941 | 38.610 |
| Lada | 5,238 | 4.880 |
| FLEURS | 1,046 | 2.518 |
| Mykyta | 1,704 | 2.008 |
| Tetiana | 534 | 0.610 |
| Telegram voices | 323 | 0.396 |
| UA-SER | 465 | 0.344 |

The pipeline used the clean boundary-trimmed audio as its source. It applied
ClearerVoice, Sidon, light de-essing, de-clicking, peak limiting,
DeepFilterNet3, RNNoise85, and a peak-safe linear loudness match. It did not
use compression, dynamic loudness normalization, VAD removal, or a second
boundary trim. It kept the exact source sample count. It wrote each output as
a physical mono 24 kHz PCM 24-bit WAV file. It wrote each file atomically and
did not change a source file.

All 74,156 files passed validation. Nine ClearerVoice outputs were too quiet.
For these files, the recorded fallback started from the clean source at Sidon.
It then applied all later stages. No final file had digital clipping.

The pipeline kept the exact 50/50 speaker-embedding assignment. It reused
37,078 vectors from audio before processing. It calculated 37,078 vectors
from final expanded-v8 audio. Each vector has 192 values.

JETS loaded model weights from the expanded-v7 epoch 93 checkpoint. It made
new optimizer and scheduler states. It used two RTX 3090 GPUs in FP32 mode. It
completed 100,000 new optimizer steps and returned exit status 0. The full
training time was 88,217 seconds.

The evaluation made 1,418 WAV files for each of the 25K, 50K, 75K, and 100K
milestones. All 5,672 WAV files passed. The pipeline also made five physical
listening WAV files for the fixed Kamianets-Podilskyi sentence. All five files
passed.

## CRISP-DM cycle 1

- Business Understanding: PASS. The v8 objective, fixed process, source
  checkpoint, limits, and gates are in the run document.
- Data Understanding: PASS. The corpus report has source, split, duration,
  format, and quality data for all 74,156 records.
- Data Preparation: PASS. Audio, text, phonemes, ESPnet data, embeddings,
  token list, and statistics passed.
- Modeling: PASS. The smoke run completed 100 steps. The full run completed
  100,000 new steps on two GPUs.
- Evaluation: PASS. Smoke inference, four milestone evaluations, and the
  five-voice export passed.
- Deployment: PASS. The local inference command made physical 24 kHz mono WAV
  files and JSON metadata.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Dataset policy | PASS | The corpus has 74,156 records and no VOA records. |
| Processing profile | PASS | The profile hash is `09471a87613a680615753d75d6fe36479903dffd052ba7fcd834f13cdeb1ddd3`. |
| PCM24 and physical output | PASS | All 74,156 training WAV files are physical mono 24 kHz PCM24 files. |
| Dataset validation | PASS | All records passed. No final file has clipping or corruption. |
| Exact 50/50 embeddings | PASS | 37,078 raw and 37,078 clean vectors exist. |
| Frontend and phonemes | PASS | eSpeak-ng 1.52.0 made non-empty phonemes. |
| Token list | PASS | Its SHA-256 is `dc4ea2634513b87e2e70b27d9545f5dd14cea91dd99391f615a24ded5ee964f2`. |
| Pitch and energy statistics | PASS | ESPnet made 252 statistics files. |
| Dual-GPU smoke run | PASS | The run completed 100 new steps. |
| Smoke inference | PASS | All 1,418 smoke WAV files passed. |
| Dual-GPU full run | PASS | JETS completed 100,000 new steps. |
| Runtime health | PASS | The log has no NaN, OOM, or critical runtime error. |
| Finite model values | PASS | The audit found 549 finite model tensors. |
| 100K checkpoint | PASS | The optimizer and report counters are 100,000. |
| Milestone match | PASS | The 100K model matches the checkpoint model. |
| Fixed-set inference | PASS | 5,672 of 5,672 WAV files passed. |
| Five-voice inference | PASS | Five of five physical WAV files passed. |
| Monitor interval | PASS | The largest training-monitor interval was 15.011 minutes. |
| Disk reserve | PASS | The minimum training reserve was 42.19 GiB. |
| Test suite | PASS | All 159 tests passed. |
| Human listening | NOT RUN | A person must check voice quality. |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/run_expanded_v8_training_cascade_pipeline.sh` | 0 | The complete processing, smoke, training, and evaluation chain passed. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_pipeline.log` |
| `training/scripts/run_expanded_v8_training_cascade_preprocessing.sh` | 0 | The command made 74,156 atomic PCM24 outputs. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_full_dataset.json` |
| `training/scripts/prepare_expanded_v8_training_cascade.sh` | 0 | Manifests, embeddings, tokens, and statistics passed. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_full_scale_readiness.json` |
| `training/scripts/run_expanded_v8_training_cascade_smoke.sh` | 0 | The run completed 100 steps and inference passed. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_smoke_checkpoint.json` |
| `training/scripts/launch_expanded_v8_training_cascade.sh 100000` | 0 | JETS completed 100,000 new steps. | `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/train.log` |
| `training/scripts/evaluate_expanded_v8_training_cascade_milestone.sh 25k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00655. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_inference_25k.json` |
| `training/scripts/evaluate_expanded_v8_training_cascade_milestone.sh 50k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00633. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_inference_50k.json` |
| `training/scripts/evaluate_expanded_v8_training_cascade_milestone.sh 75k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00629. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_inference_75k.json` |
| `training/scripts/evaluate_expanded_v8_training_cascade_milestone.sh 100k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00654. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_inference_100k.json` |
| `training/scripts/generate_five_voice_expanded_v8_eval.sh 100k` | 0 | Five physical listening WAV files passed. | `training/eval/generated/five_voice_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k/` |
| `training/scripts/audit_finetune_checkpoint.py` | 0 | Step, tensor, and model-match checks passed. | `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k_checkpoint.json` |
| `training/.venv/bin/python -m pytest -q training/tests/test_expanded_v8_training_cascade.py` | 0 | All 8 v8 tests passed. | `training/tests/test_expanded_v8_training_cascade.py` |
| `training/.venv/bin/python -m pytest -q training/tests` | 0 | All 159 tests passed in 11.10 seconds. | `training/tests/` |

## Training metrics

The final validation generator loss is 58.034. The final validation mel loss
is 36.408. The final validation alignment loss is 3.961. The final validation
discriminator loss is 1.909.

The minimum validation mel loss is 35.994 at epoch 97. The minimum validation
alignment loss is 3.959 at epoch 88. The minimum validation generator loss is
56.409 at epoch 17. The pipeline preserved all three checkpoints. A small mel
improvement continued through epoch 97, but the curve is not monotonic.

GPU 0 had a mean sampled power of 197.66 W and a maximum of 241.95 W. GPU 1
had a mean sampled power of 204.86 W and a maximum of 236.21 W. Both GPUs
reached 100 percent sampled compute use. Maximum sampled VRAM was 20,908 MiB
and 18,622 MiB. Maximum sampled temperature was 85 degrees C and 76 degrees C.
No sample had a thermal slowdown condition.

The minimum measured free disk space was 42.19 GiB. This value stayed above
the 30 GiB limit. The training monitor wrote 98 samples. It found no critical
condition and no error match.

## Створені артефакти

- Manifest: `training/data/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/manifests/all.parquet`.
- PCM24 audio: `training/data/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/audio_24k/`.
- ESPnet data: `training/espnet_recipe/data/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_*`.
- Token list: `training/dump_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Resume checkpoint: `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/checkpoint.pth`.
- Final model: `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/milestones/100k.pth`.
- Validation-best models: `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/best_checkpoints/`.
- Fixed evaluation reports: `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_inference_*k.json`.
- Fixed evaluation WAV files: `training/exp_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa/tts_jets_uk_24k_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_from_v7e93_100k/decode_jets_milestone_*/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_eval/wav/`.
- Five-voice listening WAV files: `training/eval/generated/five_voice_expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k/`.
- TensorBoard summary: `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_tensorboard_metrics.json`.
- Runtime summary: `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_training_monitor_summary.json`.
- Checkpoint audit: `training/reports/expanded_v8_clearervoice_sidon_deess_declick_limit_deepfilternet3_rnnoise85_novoa_100k_checkpoint.json`.

The final 100K model SHA-256 is
`190bc5ca7fa907d10d7f236dbb5e26def921ec52b5b55f8717ed453c9615d409`.
The validation-mel best model SHA-256 is
`a2a6af526e6135ebd161c5e37c577be4d4333631efe5b338e32770295c453804`.

## Відомі проблеми

- Human listening is not complete. Automatic checks cannot measure
  naturalness.
- A person must check for rasp, metallic sound, and robotic sound.
- Nine files used the recorded fallback because ClearerVoice made an output
  that was too quiet.
- The best validation mel result improves the prior in-run best by only
  0.004. Listening must select the final checkpoint.
- Flash Attention is not installed. ESPnet used its standard attention code.
- The file system has approximately 42 GiB free. Keep the 30 GiB limit.

## Наступна одна дія

Listen to the five 100K WAV files and the fixed 25K, 50K, 75K, and 100K
evaluation outputs. Select the checkpoint that has the least rasp, metallic
sound, and robotic sound.
