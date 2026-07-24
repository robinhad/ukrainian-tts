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
| Common Voice records | 76,762 |
| Lada records | 6,787 |
| Dmytro training records | 0 |
| Total records | 83,549 |
| Total duration after trim | 84.871 h |
| Train records | 80,041 |
| Development records | 1,831 |
| Evaluation records | 1,677 |
| Cross-source duplicate texts removed | 33 |
| Dataset validation errors | 0 |
| Clipping flags | 633 |

The trim uses a relative frame-RMS threshold of 40 dB and 100 ms boundary
padding. It changes model copies only. It does not change raw source files. It
does not apply denoise, normalization, compression, de-essing, or mastering.

## Current modeling status

ESPnet data preparation is complete. The retained ECAPA store contains 80,041
train, 1,831 development, and 1,677 evaluation vectors.

The first full token-list attempt exposed punctuation that was joined to
phonemes. The list had 873 tokens. The pipeline rejected this output before
calibration. Frontend version `uk_espeak_v2` separates each punctuation mark
from phonemes. The 50-case and 500-case snapshots pass.

The first long run exposed six corrupted Common Voice transcriptions. Each
transcription contained many source metadata rows. Their phoneme lengths were
10,763 to 244,054 tokens. Normal corpus text has at most 140 tokens. Transformer
self-attention caused a CUDA out-of-memory error in batch 941--950. The run
stopped before the first checkpoint. The rejected run and its data artifacts
remain in paths that contain `rejected_corrupt_text`.

The ingestion step now rejects embedded tab or newline metadata and text that
has more than 500 characters. Dataset validation also rejects text or phoneme
sequences above 500 items. The corrected source has six fewer Common Voice
rows. The training test suite has 50 passing tests.

The corrected full list has 87 lines, no joined punctuation token, and 0.0
percent OOV. It contains 84 corpus tokens and three ESPnet special tokens. The
statistics stage processed all 80,041 train and 1,831 development records.
Speech, pitch, and energy statistics are finite. Maximum train text shape is
140 tokens.

The full training plan is:

1. Run 200 FP32 iterations with two GPUs at each bounded batch setting.
2. Check peak memory, GPU power, all losses, and the checkpoint.
3. Keep 10 to 15 percent free VRAM.
4. Start the 25,000-iteration run only after calibration passes.
5. Generate the fixed evaluation set at the 1k, 5k, and 25k milestones.

The initial calibration results were:

| Batch bins | Result | Peak cached VRAM | Validation generator loss |
|---:|---|---:|---:|
| 3,800,000 | PASS | 17.350 GiB | 99.021 |
| 4,500,000 | PASS | 17.350 GiB | 104.686 |
| 4,650,000 | FAIL reserve | 22.58 GiB observed | Not completed |
| 4,750,000 | FAIL reserve | 21.87 GiB observed | Not completed |
| 5,000,000 | FAIL reserve | 22.213 GiB | 97.804 |
| 5,500,000 | FAIL reserve | More than 23 GiB observed | Not completed |

The 4,650,000 and larger settings did not keep the required memory reserve.
Discrete batch composition causes a memory jump above 4,500,000. The
200-iteration test at 4,500,000 did not contain a corrupted row. Therefore, it
did not expose the source defect.

The corrected 1,000-iteration gate completed on both GPUs. The command returned
exit status 0. It made 45 finite validation batches and a checkpoint. The
validation generator loss was 77.467. The validation mel loss was 59.788. The
checkpoint SHA-256 is
`0ae3911e797e035513341b7b4a2b03c7305b3566aa17b430e398ebc1580cd4c8`.

The gate reported 22.684 GiB peak cached VRAM. This value leaves less than 10
percent free VRAM. The 4,500,000 setting fails the memory-reserve gate. The
25,000-iteration continuation uses the verified 3,800,000 setting.

ESPnet writes TensorBoard event files through PyTorch. TensorFlow is not a
training runtime dependency.

## Known limit

The public cv22-opus data does not include Common Voice client IDs. The pipeline
uses one utterance-level ECAPA embedding for each Common Voice recording. It
does not claim that two Common Voice recordings have the same speaker.

No raw Dmytro corpus is available in the repository or configured storage. The
model can use the pinned Dmytro ECAPA vector for inference. This vector does not
make Dmytro part of the training set.
