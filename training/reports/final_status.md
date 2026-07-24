# Final status

## Виконано

The enhanced-v2 iteration is complete on branch `autotrain`. All training code
is under `training/`. The process made new audio copies and did not change raw
source files.

The audio process used DeepFilterNet3, a 70 Hz high-pass filter, light
de-essing, gentle 1.5 ratio compression, and two-pass EBU R128 normalization.
It wrote mono PCM WAV files at 24 kHz. The loudness target was -23 LUFS.

The model is ESPnet2 JETS with Ukrainian eSpeak-ng phonemes and ECAPA speaker
embeddings. It completed 25,000 FP32 iterations on two RTX 3090 GPUs.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS for the enhanced smoke set.
- Data Preparation: PASS.
- Modeling: PASS for 100 dual-GPU iterations.
- Evaluation: PASS for 1,659 smoke-eval WAV files.
- Deployment: PASS for the local inference entry point.

## CRISP-DM cycle 2

- Business Understanding: PASS.
- Data Understanding: PASS for 83,549 input files.
- Data Preparation: PASS for 82,662 retained files and 83.896 hours.
- Modeling: PASS for 25,000 dual-GPU iterations.
- Evaluation: PASS for 1,659 fixed-eval WAV files and five listening WAV files.
- Deployment: PASS for local inference. The release package is NOT RUN.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Frontend tests | PASS | 20 tests passed |
| eSpeak-ng pin | PASS | Version 1.52.0 and the language-data hash are pinned |
| Non-empty phonemes | PASS | Regression and corpus checks found no empty sequence |
| Enhanced dataset | PASS | 82,662 files; 83.896 hours; no split leakage |
| EBU R128 | PASS | Retained files are within 1 LU of -23 LUFS |
| Token list | PASS | 87 tokens and 0.0 percent OOV |
| Statistics | PASS | Speech, pitch, and energy statistics exist |
| JETS construction | PASS | Two DDP model processes started |
| Smoke training | PASS | 100 iterations; no NaN |
| Full training | PASS | 25,000 iterations; no NaN, OOM, or critical error |
| Checkpoint | PASS | Epoch 25 and averaged five-best checkpoints exist |
| Fixed-set inference | PASS | 1,659 of 1,659 WAV files passed |
| Five-voice evaluation | PASS | Five of five WAV files passed automatic checks |
| Human timbre check | NOT RUN | Automatic checks cannot measure metallic timbre |
| Release package | NOT RUN | Wait for the human timbre check |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/prepare_multispeaker_enhanced_v2.sh` | 0 | 82,662 retained files and statistics | `reports/multispeaker_enhanced_v2_dataset.json` |
| Enhanced smoke training | 0 | 100 dual-GPU iterations; no NaN | `exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2_smoke/1epoch.pth` |
| `training/scripts/run_multispeaker_enhanced_v2_training.sh 25000` | 0 | 25,000 iterations in 27,295 seconds | `exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/train.log` |
| ESPnet stage 8 with `train.total_count.ave.pth` | 0 | 1,659 WAV files in 76 seconds | `reports/multispeaker_enhanced_v2_inference_25k.json` |
| `training/scripts/generate_five_voice_enhanced_v2_eval.sh` | 0 | Five valid listening WAV files | `reports/five_voice_enhanced_v2_25k.json` |
| `pytest -q training/tests training/tests/frontend` | 0 | 20 tests passed in 5.21 seconds | `tests/` |

The final validation generator loss is 60.965. The final mel loss is 43.070.
Epoch 24 has the best values: 59.654 and 42.327. The log has no NaN marker.

The highest observed power samples were 240.76 W for GPU0 and 237.30 W for
GPU1. GPU0 reached 85 C. GPU1 reached 77 C. Short software thermal slowdown
occurred, but training continued without an error.

TensorBoard has 29 train scalar tags and 16 validation scalar tags.

## Створені артефакти

- Manifests: `training/data/multispeaker_enhanced_v2/manifests/`.
- Enhanced audio: `training/data/multispeaker_enhanced_v2/processed_24k/`.
- ESPnet data directories:
  `training/espnet_recipe/data/multispeaker_{train,dev,eval}/`.
- JETS config:
  `training/espnet_recipe/conf/tuning/train_jets_uk_24k_multispeaker.yaml`.
- Token list:
  `training/dump_multispeaker_enhanced_v2/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics:
  `training/exp_multispeaker_enhanced_v2/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Epoch 25 checkpoint:
  `training/exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/25epoch.pth`.
- Averaged checkpoint:
  `training/exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/train.total_count.ave_5best.pth`.
- Fixed-eval WAV files:
  `training/exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/decode_jets_train.total_count.ave/multispeaker_eval/wav/`.
- Five-voice listening set:
  `training/eval/generated/five_voice_enhanced_v2_25k/`.
- Evaluation reports:
  `training/reports/multispeaker_enhanced_v2_inference_25k.json` and
  `training/reports/five_voice_enhanced_v2_25k.json`.
- TensorBoard report:
  `training/reports/tensorboard_metrics_multispeaker_enhanced_v2.json`.

The epoch 25 SHA-256 is
`adaafa329850b34ee4c727de16a602b4c60d349e7591eddf7058cb06edd5a4dd`.
The averaged checkpoint SHA-256 is
`b09151b998dfefa2547b6cc7920092448417b214f94b3203483ce74642e3c134`.

## Відомі проблеми

- Dmytro has no raw training corpus. The Dmytro sample is zero-shot.
- Automatic checks do not measure naturalness, pronunciation, or speaker
  similarity.
- The earlier model had a user-reported metallic timbre. The new five-voice
  set needs a human check. This report does not claim that the issue is fixed.
- GPU0 had short software thermal slowdown at up to 85 C.
- No enhanced-v2 release package exists yet.

## Наступна одна дія

Listen to all five files in
`training/eval/generated/five_voice_enhanced_v2_25k/`. Record if the metallic
artifact is present in each voice.
