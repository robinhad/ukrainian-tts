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

ESPnet data preparation is complete. ECAPA extraction is in progress. Full
tokenization and statistics must pass before batch calibration starts.

The full training plan is:

1. Run 200 FP32 iterations with two GPUs and `batch_bins=3800000`.
2. Check peak memory, GPU power, all losses, and the checkpoint.
3. Keep at least 10 percent free VRAM.
4. Start the 25,000-iteration run only after calibration passes.
5. Generate the fixed evaluation set at the 1k, 5k, and 25k milestones.

ESPnet writes TensorBoard event files through PyTorch. TensorFlow is not a
training runtime dependency.

## Known limit

The public cv22-opus data does not include Common Voice client IDs. The pipeline
uses one utterance-level ECAPA embedding for each Common Voice recording. It
does not claim that two Common Voice recordings have the same speaker.

No raw Dmytro corpus is available in the repository or configured storage. The
model can use the pinned Dmytro ECAPA vector for inference. This vector does not
make Dmytro part of the training set.
