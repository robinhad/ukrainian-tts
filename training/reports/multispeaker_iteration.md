# Common Voice and Lada iteration

## Objective

Train one Ukrainian JETS model with Common Voice and Lada audio. Use ECAPA
speaker embeddings. Keep Dmytro as a zero-shot inference target until raw
Dmytro audio and text are available.

The model uses 24 kHz mono PCM WAV files. It uses eSpeak-ng 1.52.0 with Ukrainian
phonemes and stress marks. It does not use a separate text verbalizer, stress
model, G2P model, vocoder, VAD, or forced aligner.

## Smoke gate

| Gate | Status | Evidence |
|---|---|---|
| Frontend tests | PASS | The full test suite had 33 passing tests |
| Source records | PASS | 160 Common Voice and 160 Lada records |
| Audio preparation | PASS | 320 valid 24 kHz mono WAV files |
| ESPnet statistics | PASS | Token, speech, pitch, and energy artifacts exist |
| Dual-GPU training | PASS | 100 iterations and one checkpoint |
| Loss values | PASS | Generator loss fell from 290.070 to 98.775 |
| Eval inference | PASS | 32 of 32 WAV files passed validation |
| Lada inference | PASS | A finite, non-empty 24 kHz mono WAV exists |
| Dmytro inference | PASS | A finite, non-empty 24 kHz mono WAV exists |

The smoke checkpoint proves pipeline operation. It does not prove speech
quality.

## Full data

| Item | Value |
|---|---:|
| Common Voice records | 76,768 |
| Lada records | 6,787 |
| Dmytro training records | 0 |
| Total records | 83,555 |
| Total duration after trim | 84.880 h |
| Train records | 80,047 |
| Development records | 1,831 |
| Evaluation records | 1,677 |
| Cross-source duplicate texts removed | 33 |
| Dataset validation errors | 0 |
| Clipping flags | 633 |

The trim uses a relative frame-RMS threshold of 40 dB and 100 ms boundary
padding. It changes model copies only. It does not change raw source files. It
does not apply denoise, normalization, compression, de-essing, or mastering.

## Current modeling status

ESPnet data preparation is complete. ECAPA extraction made 80,047 train, 1,831
development, and 1,677 evaluation vectors.

The first full token-list attempt exposed punctuation that was joined to
phonemes. The list had 873 tokens. The pipeline rejected this output before
calibration. Frontend version `uk_espeak_v2` separates each punctuation mark
from phonemes. The 50-case and 500-case snapshots pass. The corrected full list
has 103 tokens, no joined punctuation token, and 0.0 percent OOV.

The statistics stage processed all 80,047 train and 1,831 development records.
Speech, pitch, and energy statistics are finite.

The full training plan is:

1. Run 200 FP32 iterations with two GPUs at each bounded batch setting.
2. Check peak memory, GPU power, all losses, and the checkpoint.
3. Keep 10 to 15 percent free VRAM.
4. Start the 25,000-iteration run only after calibration passes.
5. Generate the fixed evaluation set at the 1k, 5k, and 25k milestones.

Calibration results at this time are:

| Batch bins | Result | Peak cached VRAM | Validation generator loss |
|---:|---|---:|---:|
| 3,800,000 | PASS | 17.350 GiB | 99.021 |
| 4,500,000 | PASS | 17.350 GiB | 104.686 |
| 4,650,000 | FAIL reserve | 22.58 GiB observed | Not completed |
| 4,750,000 | FAIL reserve | 21.87 GiB observed | Not completed |
| 5,000,000 | FAIL reserve | 22.213 GiB | 97.804 |
| 5,500,000 | FAIL reserve | More than 23 GiB observed | Not completed |

The 4,650,000 and larger settings did not keep the required memory reserve.
Discrete batch composition causes a memory jump above 4,500,000. The selected
long-run value is 4,500,000. It is the largest setting that completed 200
iterations and kept the memory gate.

ESPnet writes TensorBoard event files through PyTorch. TensorFlow is not a
training runtime dependency.

## Known limit

The public cv22-opus data does not include Common Voice client IDs. The pipeline
uses one utterance-level ECAPA embedding for each Common Voice recording. It
does not claim that two Common Voice recordings have the same speaker.

No raw Dmytro corpus is available in the repository or configured storage. The
model can use the pinned Dmytro ECAPA vector for inference. This vector does not
make Dmytro part of the training set.
