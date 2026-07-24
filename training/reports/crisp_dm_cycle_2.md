# CRISP-DM cycle 2

## Business understanding

The objective was one reproducible Ukrainian JETS model for Common Voice and
Lada audio. The model must support speaker embeddings. Lada is the named local
speaker. Dmytro stays an inference-only zero-shot target because no raw Dmytro
training corpus is available.

The fixed architecture uses ESPnet2 GAN-TTS, JETS, 24 kHz audio, phoneme tokens,
and eSpeak-ng 1.52.0 for Ukrainian. It does not use a separate verbalizer,
stress model, G2P model, vocoder, VAD, or forced aligner.

## Data understanding

The pinned and corrected data contains 76,762 Common Voice records and 6,787
Lada records. The ingestion step rejected six Common Voice rows that contained
embedded TSV metadata. It also removed 33 cross-source duplicate texts.

The retained data contains 83,549 utterances and 84.871 hours after boundary
silence removal. The split contains 80,041 train, 1,831 development, and 1,677
evaluation records. Validation found no ID, audio-hash, or text-hash split
leakage. It found 633 clipping flags. The flags stay in the manifest.

The public Common Voice source does not contain stable client IDs. The pipeline
uses one utterance-level ECAPA embedding for each Common Voice recording. It
does not claim a common speaker identity for two Common Voice files.

## Data preparation

The preparation stage made mono PCM WAV model copies at 24 kHz. A relative
frame-RMS detector removed leading and trailing silence at 40 dB below each
file maximum. It kept 100 ms of boundary padding. It did not change raw source
files. It did not apply denoise, normalization, compression, de-essing, or
mastering.

The text stage applies Unicode NFC, apostrophe unification, control-character
removal, standard spaces, repeated-space collapse, and edge trimming. It does
not expand numbers, dates, units, URLs, or email addresses. The same pinned
eSpeak-ng frontend serves training and inference.

The corrected token list has 87 lines and 0.0 percent OOV. The maximum retained
phoneme sequence has 140 tokens. The statistics stage made finite speech,
pitch, and energy statistics. The retained ECAPA store has one finite,
nonzero, 192-value vector for each retained utterance.

## Modeling

The first long-run attempt found six corrupted Common Voice transcriptions.
Their phoneme sequences had 10,763 to 244,054 tokens. The model stopped with a
CUDA out-of-memory error before a checkpoint existed. The corrected ingestion
and validation steps now reject tabs, newlines, texts above 500 characters, and
phoneme sequences above 500 tokens.

Dynamic batch tests rejected 4,500,000, 3,800,000, 3,400,000, 3,000,000, and
2,500,000 batch bins because a later batch did not keep the required memory
reserve. The stable setting is 2,000,000 batch bins with expandable PyTorch
allocator segments. The run used FP32 and both RTX 3090 GPUs.

The corrected run completed 25,000 iterations and returned exit status 0 after
23,461 seconds. It reported no NaN, OOM, or critical runtime error. Peak cached
memory was 15.500 GiB.

| Iteration | Validation generator loss | Validation mel loss | Validation alignment loss |
|---:|---:|---:|---:|
| 1,000 | 77.467 | 59.788 | 5.558 |
| 5,000 | 65.756 | 48.828 | 4.961 |
| 10,000 | 62.793 | 44.599 | 4.742 |
| 15,000 | 59.247 | 42.689 | 4.626 |
| 19,000 | 58.305 | 40.971 | 4.557 |
| 24,000 | 57.872 | 40.226 | 4.506 |
| 25,000 | 58.242 | 41.292 | 4.498 |

The lowest validation mel loss is at 24k. It is 32.7 percent below the 1k
value. The final validation mel loss is 30.9 percent below the 1k value.
Alignment loss continued to improve through 25k. These changes are meaningful.
A listening test must still measure naturalness, pronunciation, and speaker
quality.

The external monitor collected 104 samples. GPU0 mean and maximum power were
193.16 W and 244.29 W. GPU1 mean and maximum power were 215.55 W and 254.45 W.
Both cards reached 100 percent sampled compute use. Their maximum temperatures
were 85 C and 77 C.

TensorBoard contains 29 train scalar tags and 16 validation scalar tags.
PyTorch writes the event files. TensorFlow is not a training runtime
dependency.

## Evaluation

The 1k, 5k, and 25k checkpoints each made all 1,677 fixed evaluation
utterances. Independent validation accepted all 5,031 WAV files. Each file is
finite, non-empty, mono, and 24 kHz. No evaluation set has a clipping warning.

| Milestone | WAV count | Duration range, s | Median RTF |
|---|---:|---:|---:|
| 1k | 1,677 | 0.736--5.760 | 0.00829 |
| 5k | 1,677 | 0.704--6.635 | 0.00833 |
| 25k | 1,677 | 0.832--7.371 | 0.00800 |

The local entry point made one Lada WAV and one Dmytro zero-shot WAV. The Lada
WAV is 2.880 seconds and has RTF 0.13110. The Dmytro WAV is 3.115 seconds and
has RTF 0.11909. Both files have adjacent JSON metadata.

## Deployment

The release candidate is
`training/releases/uk-tts-jets-multispeaker-25k-rc/`. It contains 34 files,
including the checkpoint, config, token list, statistics, frontend snapshot,
automatic evaluation, power report, TensorBoard report, licenses, cards,
examples, and SHA-256 checksums.

The checkpoint SHA-256 is
`395ccaba7e6837a60257a622b8d9ce0352246e41f728273e92d09c41b444f445`.
The 20-item listening set contains balanced Common Voice and Lada items for the
1k, 5k, and 25k candidates. Perceptual review is the next release decision.
