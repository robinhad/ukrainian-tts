# Expanded-v3 500K listening follow-up

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify the report.

## User result

The user hears rasp in the 500K audio. The audio also sounds robotic in some
parts. Human listening has `FAIL`. Release status has `FAIL`.

## Automatic result

The WAV files do not have digital clipping. The maximum measured peak in the
fixed evaluation is 0.607. No sample is at or above 0.99. The files are finite,
mono, and 24 kHz.

The high-frequency energy ratio is similar in the generated audio and in the
five reference files. This result does not support digital clipping as the
cause. It does not exclude a JETS generator artifact or a learned audio-process
artifact.

## A/B candidates

| Candidate | Selection reason | Automatic status | Human status | Directory |
|---|---|---|---|---|
| Epoch 500 | Final checkpoint | PASS | FAIL | `eval/generated/five_voice_expanded_v3_500k/` |
| Epoch 367 | Best validation generator loss, 56.028 | PASS | NOT RUN | `eval/generated/five_voice_expanded_v3_500k_e367_genbest/` |
| Epoch 443 | Best validation mel loss, 39.860 | PASS | NOT RUN | `eval/generated/five_voice_expanded_v3_500k_e443_melbest/` |
| Epoch 488 | Best validation alignment loss, 4.316 | PASS | NOT RUN | `eval/generated/five_voice_expanded_v3_500k_e488_alignbest/` |

Each candidate contains the same five voices and the same sentence. The 20 WAV
files passed the automatic checks.

## Listening order

1. Listen to `voice_01.wav` from all four directories.
2. Rank the four files for rasp.
3. Rank the four files for robotic sound.
4. Repeat the check for voices 02 to 05.
5. Record whether the defect occurs in all voices or only in some voices.

If epoch 367 or 443 is better, use that checkpoint for the next fixed-set
evaluation. If all checkpoints have the same defect, inspect the JETS generator
and the clean training-audio process before more training.
