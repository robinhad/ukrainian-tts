# Final status

## Виконано

The reproducible Ukrainian JETS pipeline is on branch `autotrain`. All new
training code is under `training/`. The Common Voice and Lada run completed
25,000 FP32 iterations on two RTX 3090 GPUs. The command returned exit status
0. The code uses author `Codex <codex@openai.com>`.

The pipeline made silence-trimmed model copies and kept raw source files
unchanged. It uses ESPnet2 JETS, GAN-TTS, 24 kHz mono audio, phoneme tokens,
eSpeak-ng 1.52.0, and 192-value ECAPA speaker embeddings. Dmytro is not in the
train set. The Dmytro example uses the pinned legacy embedding for zero-shot
inference.

## CRISP-DM cycle 1

- Business Understanding: PASS.
- Data Understanding: PASS for the Common Voice and Lada smoke subset.
- Data Preparation: PASS for 320 smoke records.
- Modeling: PASS for 100 dual-GPU iterations.
- Evaluation: PASS for 32 of 32 smoke-eval WAV files.
- Deployment: PASS for Lada and Dmytro zero-shot local entry points.

## CRISP-DM cycle 2

- Business Understanding: PASS.
- Data Understanding: PASS for 83,549 retained utterances.
- Data Preparation: PASS for trimmed audio, split, token list, statistics, and
  speaker embeddings.
- Modeling: PASS for 25,000 FP32 iterations on two GPUs.
- Evaluation: PASS for 5,031 fixed-set WAV files across 1k, 5k, and 25k.
- Deployment: PASS for local inference, listening set, and release candidate.

## MVP-gate

| Gate | Status | Evidence |
|---|---|---|
| Frontend tests | PASS | Final training suite has 50 passing tests |
| eSpeak-ng pin | PASS | Version 1.52.0 and the language-data hash are pinned |
| Non-empty phonemes | PASS | Regression and corpus checks found no empty sequence |
| Dataset validation | PASS | 80,041 train, 1,831 development, and 1,677 evaluation records; zero errors |
| Token list | PASS | 87 lines and 0.0 percent OOV |
| Statistics | PASS | Finite speech, pitch, and energy statistics exist |
| Speaker embeddings | PASS | 83,549 finite, nonzero 192-value vectors exist |
| JETS construction | PASS | JETS and both AdamW optimizers initialize |
| Stable training | PASS | 25,000 iterations; no NaN, OOM, or critical error |
| Checkpoint | PASS | The 25k checkpoint and milestone copy are byte-identical |
| Fixed-set inference | PASS | 5,031 of 5,031 WAV files pass |
| Release candidate | PASS | The package has 34 files and verified checksums |
| Reproduction commands | PASS | `training/README.md` and scripts contain the commands |

## Фактичні запуски

| Command | Exit | Key result | Artifact |
|---|---:|---|---|
| `training/scripts/run_multispeaker_smoke_test.sh` | 0 | 100 dual-GPU iterations and 32 valid eval WAV files | `reports/multispeaker_smoke_inference.json` |
| `training/scripts/prepare_multispeaker_full.sh` | 0 | 83,549 records; 84.871 h; 0.0 percent OOV; statistics complete | `reports/multispeaker_full_dataset.json` |
| `BATCH_BINS=2000000 training/scripts/run_multispeaker_training.sh 25000` | 0 | 25,000 iterations on two GPUs in 23,461 s | `exp_multispeaker_full/tts_jets_uk_24k_multispeaker/train.log` |
| `training/scripts/finalize_multispeaker_training.sh 25epoch.pth 25k` | 0 | Three 1,677-file evaluations, local examples, listening set, and release | `reports/multispeaker_full_inference_25k.json` |
| `source training/activate.sh && pytest -q training/tests` | 0 | 50 tests passed | `tests/` |
| `sha256sum -c checksums.txt` in the release directory | 0 | All 33 payload checksums passed | `releases/uk-tts-jets-multispeaker-25k-rc/checksums.txt` |
| `pytest -q` from the repository root | 2 | Legacy test collection stopped because optional `stanza` is not installed | `tests/` |

The 25k validation generator, mel, and alignment losses are 58.242, 41.292,
and 4.498. The lowest validation mel loss is 40.226 at 24k. This value is 32.7
percent below the 1k value of 59.788. Peak cached VRAM is 15.500 GiB.

The GPU monitor collected 104 samples. GPU0 mean and maximum power were 193.16 W
and 244.29 W. GPU1 mean and maximum power were 215.55 W and 254.45 W. Both
cards reached 100 percent sampled compute use.

The TensorBoard audit found 29 train scalar tags and 16 validation scalar tags.
The 25k fixed-set run accepted 1,677 of 1,677 mono 24 kHz files. It found no
clipping warning. Duration is 0.832 to 7.371 seconds. Median RTF is 0.00800.

## Створені артефакти

- Manifests: `training/data/multispeaker_full/manifests/`.
- ESPnet data directories:
  `training/espnet_recipe/data/multispeaker_{train,dev,eval}/`.
- JETS config:
  `training/espnet_recipe/conf/tuning/train_jets_uk_24k_multispeaker.yaml`.
- ESPnet patch: `training/patches/espnet-espeak-ng-ukrainian.patch`.
- Token list:
  `training/dump_multispeaker_full/token_list/phn_espeak_ng_ukrainian/tokens.txt`.
- Statistics:
  `training/exp_multispeaker_full/tts_stats_raw_phn_espeak_ng_ukrainian/`.
- Checkpoint: `training/exp_multispeaker_full/milestones/25k.pth`.
- Checkpoint SHA-256:
  `395ccaba7e6837a60257a622b8d9ce0352246e41f728273e92d09c41b444f445`.
- TensorBoard events:
  `training/exp_multispeaker_full/tts_jets_uk_24k_multispeaker/tensorboard/`.
- Fixed-eval WAV files:
  `training/exp_multispeaker_full/tts_jets_uk_24k_multispeaker/decode_jets_milestone_25k/multispeaker_eval/wav/`.
- Evaluation reports:
  `training/reports/multispeaker_full_inference_{1k,5k,25k}.json`.
- Power report: `training/reports/power_summary_multispeaker.json`.
- Local examples: `training/eval/generated/multispeaker_25k_{lada,dmytro_zero_shot}.wav`.
- Listening set: `training/eval/generated/listening_multispeaker_25k/`.
- Release candidate:
  `training/releases/uk-tts-jets-multispeaker-25k-rc/`.

## Відомі проблеми

- Common Voice client IDs are not present in the pinned public source.
- Dmytro has no raw training corpus. Dmytro is a zero-shot inference target.
- Automatic WAV checks do not measure naturalness, pronunciation, or speaker
  similarity.
- The user reported a metallic timbre in the 25k output. The perceptual status
  is `FAIL` until the five-voice review identifies whether the artifact is
  shared or speaker-dependent.
- GPU0 reached 85 C and had brief software thermal slowdown. Hardware thermal
  slowdown did not occur.
- NCCL cannot use direct P2P on this host. It uses shared-memory transport.
- The repository root test collection needs the legacy optional `stanza`
  dependency. The isolated `training/tests` suite does not need it.

## Наступна одна дія

Listen to the five files in
`training/eval/generated/five_voice_metallic_review/`. Record whether the
metallic artifact is present in all voices before you select the next training
change.
