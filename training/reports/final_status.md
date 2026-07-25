# Final status

## Виконано

The enhanced-v2 model reached 100,000 total iterations on branch `autotrain`.
All training code is under `training/`. The run used both RTX 3090 GPUs.

The audio data uses DeepFilterNet3, a 70 Hz high-pass filter, light de-essing,
gentle 1.5 ratio compression, and two-pass EBU R128 normalization. The process
wrote new mono PCM WAV files at 24 kHz. It did not change the source audio.

The model is ESPnet2 JETS. It uses Ukrainian eSpeak-ng phonemes and ECAPA
speaker embeddings.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS.
- Data Preparation: PASS.
- Modeling: PASS for 100 dual-GPU iterations.
- Evaluation: PASS for 1,659 smoke-eval WAV files.
- Deployment: PASS for the local inference entry point.

## CRISP-DM cycle 2

- Business Understanding: PASS.
- Data Understanding: PASS for 83,549 input files.
- Data Preparation: PASS for 82,662 retained files and 83.896 hours.
- Modeling: PASS for 100,000 total dual-GPU iterations.
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
| Full training | PASS | 100,000 total iterations; no NaN, OOM, or critical error |
| Checkpoint | PASS | Epoch 100 and averaged five-best checkpoints exist |
| Fixed-set inference | PASS | 1,659 of 1,659 WAV files passed |
| Five-voice evaluation | PASS | Five of five WAV files passed automatic checks |
| Human timbre check | NOT RUN | Automatic checks cannot measure metallic timbre |
| Release package | NOT RUN | Wait for the human timbre check |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/prepare_multispeaker_enhanced_v2.sh` | 0 | 82,662 retained files and statistics | `reports/multispeaker_enhanced_v2_dataset.json` |
| Enhanced smoke training | 0 | 100 dual-GPU iterations; no NaN | `exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2_smoke/1epoch.pth` |
| `training/scripts/run_multispeaker_enhanced_v2_training.sh 25000` | 0 | First 25,000 iterations | `exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/25epoch.pth` |
| `training/scripts/run_multispeaker_enhanced_v2_training.sh 100000` | 0 | Resumed to 100,000 total iterations in 81,938 seconds | `exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/100epoch.pth` |
| ESPnet stage 8 with `train.total_count.ave.pth` | 0 | 1,659 valid WAV files in 77 seconds | `reports/multispeaker_enhanced_v2_inference_100k.json` |
| 100k five-voice command | 0 | Five valid listening WAV files | `reports/five_voice_enhanced_v2_100k.json` |
| `pytest -q training/tests training/tests/frontend` | 0 | 20 tests passed in 5.16 seconds | `tests/` |

The final validation generator loss is 57.538. The final mel loss is 39.769.
The best values are 56.459 and 38.897 at 97k. The best alignment loss is 4.204
at 99k. The final discriminator loss is 1.833.

The highest sampled power was 231.31 W for GPU0 and 241.60 W for GPU1. GPU0
reached 86 C. GPU1 reached 78 C. GPU0 had software thermal slowdown. The
training process completed without a hardware thermal shutdown.

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
- Epoch 100 checkpoint:
  `training/exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/100epoch.pth`.
- Averaged checkpoint:
  `training/exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/train.total_count.ave_5best.pth`.
- Fixed-eval WAV files:
  `training/exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/decode_jets_train.total_count.ave/multispeaker_eval/wav/`.
- Five-voice listening set:
  `training/eval/generated/five_voice_enhanced_v2_100k/`.
- Evaluation reports:
  `training/reports/multispeaker_enhanced_v2_inference_100k.json` and
  `training/reports/five_voice_enhanced_v2_100k.json`.
- Power report:
  `training/reports/power_summary_multispeaker_enhanced_v2_100k.json`.
- TensorBoard report:
  `training/reports/tensorboard_metrics_multispeaker_enhanced_v2_100k.json`.

The epoch 100 SHA-256 is
`9a5cfbde6c1e8133dff95278b1920d1ed596836bcbd4c5c1ef5ac35f611c4384`.
The averaged checkpoint SHA-256 is
`7ad913443283cab76db31c9ad058dcdec43f87dce148dd29a6cd660920208897`.

## Відомі проблеми

- Dmytro has no raw training corpus. The Dmytro sample is zero-shot.
- Automatic checks do not measure naturalness, pronunciation, or speaker
  similarity.
- The earlier model had a user-reported metallic timbre. The 100k five-voice
  set needs a human check. This report does not claim that the issue is fixed.
- GPU0 had software thermal slowdown at up to 86 C.
- No enhanced-v2 release package exists.

## Наступна одна дія

Listen to all five files in
`training/eval/generated/five_voice_enhanced_v2_100k/`. Record if the metallic
artifact is present in each voice.
