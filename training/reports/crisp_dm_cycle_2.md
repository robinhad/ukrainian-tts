# CRISP-DM cycle 2

## Business understanding

The objective is stable Ukrainian speech in the Lada voice. The pipeline must
keep one speaker and must make reproducible local inference. JETS, ESPnet2,
eSpeak-ng for Ukrainian, phoneme tokens, and 24 kHz audio remain fixed. The
restart also removes leading and trailing silence from model audio copies.

## Data understanding

The pinned source contains 6,962 rows. Data preparation selected 6,787
utterances. It removed 174 duplicate sanitized texts and one duration outlier.
The untrimmed selected audio contains 10.343 hours.

The silence trim processed all 6,787 model copies. It kept 5.565 hours and
removed 4.778 hours. The split contains 6,461 train, 146 development, and 180
evaluation utterances. Exact ID, audio-hash, and text-hash leakage is zero.

The source has no document ID. The split uses each contiguous block of 50 numeric
source files as one related group. This proxy can leave related content in
different splits.

## Data preparation

The preparation stage made mono PCM WAV files at 24 kHz. A relative frame-RMS
detector removed leading and trailing silence at 40 dB below each file maximum.
It kept 100 ms of boundary padding. It did not use VAD or a forced aligner. It
did not apply denoise, normalization, compression, de-essing, or mastering.

ESPnet accepted the train, development, and evaluation directories. Tokenization
reported 0.0 percent OOV. The statistics stage made speech, pitch, and energy
statistics. The same pinned eSpeak-ng 1.52.0 frontend serves training and
inference.

## Modeling

The trimmed smoke run completed 100 iterations and made a checkpoint. It used
both generator and discriminator paths. It also made 32 of 32 valid eval WAV
files.

Dual-GPU calibration tested 4,000,000 and 4,200,000 batch bins. The 4,200,000
setting left less than 10 percent device-memory reserve. The long run first used
4,000,000. One card later had only 935 MiB free. The run stopped after the 3k
checkpoint and resumed from that state with `batch_bins: 3800000`.

The resumed FP32 run completed 25,000 iterations on both RTX 3090 cards. It
returned exit status 0 after 32,318 seconds. Peak cached memory was 22.178 GiB.
All reported generator, discriminator, alignment, pitch, and energy losses were
finite.

| Iteration | Validation generator loss | Validation mel loss |
|---:|---:|---:|
| 1,000 | 74.101 | 56.198 |
| 5,000 | 53.736 | 40.951 |
| 10,000 | 50.890 | 37.879 |
| 15,000 | 48.865 | 35.251 |
| 20,000 | 48.014 | 34.565 |
| 25,000 | 47.112 | 33.549 |

The validation mel loss fell by 40.3 percent. The validation generator loss fell
by 36.4 percent. The last train and validation mel losses were 33.677 and 33.549.
This is a meaningful loss decrease. A listening test must still measure
naturalness and pronunciation.

The external monitor collected 140 samples. GPU0 mean and maximum power were
197.11 W and 264.08 W. GPU1 mean and maximum power were 221.82 W and 261.39 W.
Both cards reached 100 percent sampled compute use. Their maximum temperatures
were 85 C and 79 C.

TensorBoard contains 29 train scalar tags and 16 validation scalar tags through
step 25,000. PyTorch writes the event files. TensorFlow is not a runtime
dependency.

## Evaluation

The 25k checkpoint made all 180 fixed evaluation utterances. Independent
validation accepted 180 of 180 files. Each file is finite, non-empty, mono, and
24 kHz. The duration range is 1.013 to 5.376 seconds. Median RTF is 0.01338.
There are no clipping warnings.

The local entry point made a 3.189-second raw WAV. Its RTF is 0.10438 and its peak
absolute sample is 0.509. It also wrote the frontend, checkpoint, and config
metadata.

## Deployment

The 25k checkpoint, config, token list, statistics, TensorBoard files, inference
report, local example, and 20-item listening set exist. The checkpoint SHA-256 is
`58f4673676cd382d1ae2bc6c5a7a80e809ccce9e2b3dea42edef6cae177f9d75`.
Perceptual review and release packaging remain separate steps.
