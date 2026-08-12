# Final Status

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this report.

## Виконано

The expanded-v7 pipeline completed at 01:28:48 Kyiv time on 2026-08-13. It
processed 74,156 utterances. The total duration is 92.266 hours. The training
split has 71,334 utterances and 89.223 hours. The development split has 1,404
utterances and 1.514 hours. The evaluation split has 1,418 utterances and
1.529 hours. The corpus has no VOA records.

The pipeline used the completed expanded-v6 Sidon and light de-essing audio as
its source. It applied this final FFmpeg filter graph:

```text
aformat=sample_fmts=fltp,adeclick=w=55:o=75:a=2:t=4:b=2,alimiter=limit=0.891251:attack=5:release=80:level=false:latency=true
```

The filters used floating-point audio. The pipeline wrote 24 kHz mono PCM24
WAV files. It wrote each output atomically. It did not change a source file.
All 74,156 files passed. No file failed.

The pipeline kept the exact 50/50 speaker-embedding assignment. It reused
37,078 vectors from audio before processing. It calculated 37,078 vectors
from final expanded-v7 audio. Each vector has 192 values. Training used the
final clean PCM24 audio.

ESPnet made the token list and 252 statistics files. JETS loaded model weights
from the expanded-v6 epoch 94 checkpoint. It made new optimizer and scheduler
states. It used two RTX 3090 GPUs in FP32 mode. It completed 100,000 new
optimizer steps and ended with exit code 0.

The evaluation made 1,418 WAV files for each of the 25K, 50K, 75K, and 100K
milestones. All 5,672 WAV files passed the automatic checks. Each WAV file is
mono at 24 kHz. No evaluation reported an error or a clipping warning.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS.
- Data Preparation: PASS.
- Modeling: PASS.
- Evaluation: PASS.
- Deployment: PASS.

The smoke run completed 100 optimizer steps on two GPUs. The checkpoint audit
passed. Smoke inference made 1,418 valid WAV files.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Dataset policy | PASS | The corpus has 74,156 records and no VOA records. |
| Final audio process | PASS | All 74,156 files used de-clicking and peak limiting. |
| PCM24 output | PASS | All training paths refer to 24 kHz mono PCM24 WAV files. |
| Atomic output | PASS | The batch report confirms atomic writes and preserved sources. |
| Dataset validation | PASS | All records passed. No file has clipping or corruption. |
| Exact 50/50 embeddings | PASS | 37,078 raw and 37,078 clean vectors exist. |
| Frontend and phonemes | PASS | eSpeak-ng 1.52.0 made non-empty phonemes. |
| Token list | PASS | Its SHA-256 is `dc4ea2634513b87e2e70b27d9545f5dd14cea91dd99391f615a24ded5ee964f2`. |
| Pitch and energy statistics | PASS | ESPnet made all required statistics. |
| Dual-GPU smoke run | PASS | The run completed 100 steps. |
| Smoke inference | PASS | All 1,418 smoke WAV files passed. |
| Dual-GPU full run | PASS | JETS completed 100,000 new steps. |
| Finite model values | PASS | The audit found 549 finite model tensors. |
| 100K checkpoint | PASS | The optimizer and report counters are 100,000. |
| Milestone match | PASS | The 100K model matches the checkpoint model. |
| Fixed-set inference | PASS | 5,672 of 5,672 WAV files passed. |
| Monitor interval | PASS | The largest interval was 15.010 minutes. |
| Disk reserve | PASS | The minimum measured reserve was 72.30 GiB. |
| Test suite | PASS | All 137 tests passed. |
| Human listening | NOT RUN | A person must check voice quality. |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/run_expanded_v7_declick_limited_pipeline.sh` | 0 | The complete processing, smoke, training, and evaluation chain passed. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_pipeline.log` |
| `training/scripts/run_expanded_v7_audio_postprocessing.sh` | 0 | The command wrote 74,156 atomic PCM24 outputs. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_postprocessing.json` |
| `training/scripts/prepare_expanded_v7_declick_limited.sh` | 0 | Manifests, embeddings, tokens, and statistics passed. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_full_scale_readiness.json` |
| `training/scripts/run_expanded_v7_declick_limited_smoke.sh` | 0 | The run completed 100 steps and inference passed. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_smoke_checkpoint.json` |
| `training/scripts/launch_expanded_v7_declick_limited_training.sh 100000` | 0 | JETS completed 100,000 new steps. | `training/exp_expanded_v7_sidon_deess_declick_limit_novoa/tts_jets_uk_24k_expanded_v7_sidon_deess_declick_limit_novoa_from_v6e94_100k/train.log` |
| `training/scripts/evaluate_expanded_v7_declick_limited_milestone.sh 25k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00667. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_inference_25k.json` |
| `training/scripts/evaluate_expanded_v7_declick_limited_milestone.sh 50k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00633. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_inference_50k.json` |
| `training/scripts/evaluate_expanded_v7_declick_limited_milestone.sh 75k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00648. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_inference_75k.json` |
| `training/scripts/evaluate_expanded_v7_declick_limited_milestone.sh 100k` | 0 | All 1,418 WAV files passed. Median RTF was 0.00651. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_inference_100k.json` |
| `training/scripts/audit_finetune_checkpoint.py` | 0 | Step, tensor, and model-match checks passed. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_100k_checkpoint.json` |
| `training/scripts/summarize_tensorboard.py` | 0 | Two TensorBoard runs and all scalar tags passed. | `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_tensorboard_metrics.json` |
| `training/.venv/bin/python -m pytest -q training/tests` | 0 | All 137 tests passed in 10.95 seconds. | `training/tests/` |

## Training metrics

The final validation generator loss is 55.655. The final validation mel loss
is 33.722. The final validation alignment loss is 3.972. The final validation
discriminator loss is 2.021.

The minimum validation mel loss is 33.045 at epoch 93. The minimum validation
alignment loss is 3.969 at epoch 82. The minimum validation generator loss is
54.160 at epoch 17. The pipeline preserved all three checkpoints.

GPU 0 had a mean sampled power of 201.99 W and a maximum of 243.24 W. GPU 1
had a mean sampled power of 206.07 W and a maximum of 247.17 W. Maximum
sampled compute use was 99 percent and 100 percent. Maximum sampled VRAM was
21,772 MiB and 18,622 MiB. Maximum sampled temperature was 87 degrees C and
78 degrees C. No sample reached the 90 degrees C critical limit.

The minimum measured free disk space was 72.30 GiB. This value stayed above
the 30 GiB limit. The monitor wrote 104 status samples. It found no critical
condition and no error match.

## Створені артефакти

- Manifest: `training/data/expanded_v7_sidon_deess_declick_limit_novoa/manifests/all.parquet`.
- PCM24 audio: `training/data/expanded_v7_sidon_deess_declick_limit_novoa/audio_24k/`.
- ESPnet data: `training/espnet_recipe/data/expanded_v7_sidon_deess_declick_limit_novoa_*`.
- Token list: `training/dump_expanded_v7_sidon_deess_declick_limit_novoa/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics: `training/exp_expanded_v7_sidon_deess_declick_limit_novoa/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Resume checkpoint: `training/exp_expanded_v7_sidon_deess_declick_limit_novoa/tts_jets_uk_24k_expanded_v7_sidon_deess_declick_limit_novoa_from_v6e94_100k/checkpoint.pth`.
- Final 100K model: `training/exp_expanded_v7_sidon_deess_declick_limit_novoa/tts_jets_uk_24k_expanded_v7_sidon_deess_declick_limit_novoa_from_v6e94_100k/milestones/100k.pth`.
- Validation-best models: `training/exp_expanded_v7_sidon_deess_declick_limit_novoa/tts_jets_uk_24k_expanded_v7_sidon_deess_declick_limit_novoa_from_v6e94_100k/best_checkpoints/`.
- Fixed evaluation reports: `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_inference_*k.json`.
- Fixed evaluation WAV files: `training/exp_expanded_v7_sidon_deess_declick_limit_novoa/tts_jets_uk_24k_expanded_v7_sidon_deess_declick_limit_novoa_from_v6e94_100k/decode_jets_milestone_*/expanded_v7_sidon_deess_declick_limit_novoa_eval/wav/`.
- TensorBoard summary: `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_tensorboard_metrics.json`.
- Runtime summary: `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_training_monitor_summary.json`.
- Checkpoint audit: `training/reports/expanded_v7_sidon_deess_declick_limit_novoa_100k_checkpoint.json`.

The final 100K model SHA-256 is
`9217fc116df3511dccfb4dc1f7cfecca095216aa662e0c2cf299e4581990e318`.
The validation-mel best model SHA-256 is
`bec6bb4587a16250e39ce554cb6f2a6563f4d67fc825fc711a2c07436560129a`.

## Відомі проблеми

- Human listening is not complete. Automatic checks cannot measure naturalness.
- A person must check for rasp, metallic sound, and robotic sound.
- GPU 0 reached 87 degrees C. The critical limit was 90 degrees C.
- Flash Attention is not installed. ESPnet used its standard attention code.
- The final model choice needs a listening comparison. Validation mel loss is not sufficient.

## Наступна одна дія

Listen to the fixed evaluation outputs from the 25K, 50K, 75K, and 100K
milestones. Select the checkpoint that has the least rasp, metallic sound, and
robotic sound.
