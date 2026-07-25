# Enhanced audio iteration

## Purpose

This iteration tests a new audio preparation process. The source audio does not
change. The process makes new model copies.

The process has these operations:

1. Remove noise with DeepFilterNet3.
2. Apply a 70 Hz high-pass filter.
3. Apply light de-essing.
4. Apply gentle compression with a ratio of 1.5.
5. Apply two-pass EBU R128 loudness normalization.
6. Write mono PCM WAV at 24 kHz.

The loudness target is -23 LUFS. The true-peak limit is -1 dBTP. A file must be
within 1 LU of the loudness target.

## Software

- DeepFilterNet version: 0.5.6
- DeepFilterNet model: DeepFilterNet3
- PyTorch version: 2.9.1+cu128
- ESPnet commit: see `vendor/ESPNET_COMMIT`
- eSpeak-ng version: see `vendor/ESPEAK_NG_VERSION`

DeepFilterNet 0.5.6 uses an old torchaudio metadata interface. The local
compatibility module supplies only this interface. It does not change PyTorch.

## Data results

- Input files: 83,549
- Retained files: 82,662
- Retained duration: 83.896 hours
- Processing rejects: 7
- Loudness rejects: 880
- Train files: 79,188
- Development files: 1,815
- Evaluation files: 1,659
- Clipped retained files: 0
- Split leakage: 0

The 7 processing rejects were below the absolute EBU R128 gate. The 880
loudness rejects were outside the 1 LU tolerance.

For retained files, the median loudness is -22.97 LUFS. The range is -23.98 to
-22.00 LUFS. The maximum true peak is -1.0 dBTP.

## Problems and corrections

The first full process used long-life DeepFilterNet workers. Native memory
increased during the process. The system stopped some workers.

The corrected process handles a maximum of 300 new files in one worker. It then
starts a new worker. The process uses the saved record files to continue. It
does not process complete files again.

The first smoke training command used a separate experiment root. That root did
not contain the statistics. The corrected command uses the enhanced experiment
root and a separate smoke model directory.

## Smoke gates

| Gate | Status | Evidence |
|---|---|---|
| Resource check | PASS | Two RTX 3090 GPUs and more than 120 GiB available RAM |
| Frontend tests | PASS | 20 tests passed |
| Dataset validation | PASS | `multispeaker_enhanced_v2_validation.json` |
| Token list | PASS | 87 tokens and 0.0% OOV |
| Pitch and energy statistics | PASS | `exp_multispeaker_enhanced_v2/tts_stats_raw_phn_espeak_ng_ukrainian` |
| Model construction | PASS | Two DDP model processes started |
| Smoke training | PASS | 100 iterations, no NaN |
| Smoke checkpoint | PASS | SHA-256 `cc68e70016e6ade390b3eb7128e0b3fc108dcedaec4e4d3f3bc347a8ac566405` |
| Smoke inference | PASS | 1,659 WAV files, 0 errors |
| WAV validation | PASS | 24 kHz, mono, finite, no clipping |

The smoke training used both GPUs. The maximum cached GPU memory was 8.4 GB for
each process. The average validation generator loss was 103.429.

## Full training

The resource check found two RTX 3090 GPUs and more than 122 GiB of available
RAM. The full run used both GPUs, FP32, 2,000,000 batch bins, and 8 data
workers. It completed 25,000 iterations in 27,295 seconds. The command returned
exit status 0. The log has no NaN, OOM, or critical runtime error.

The final validation generator loss is 60.965. The final validation mel loss is
43.070. Epoch 24 has the best values. Its generator loss is 59.654, and its mel
loss is 42.327. The generator loss fell by 31.2 percent from epoch 1 to the best
epoch.

TensorBoard has 29 train scalar tags and 16 validation scalar tags. The report
is `tensorboard_metrics_multispeaker_enhanced_v2.json`.

The 25-epoch checkpoint SHA-256 is
`adaafa329850b34ee4c727de16a602b4c60d349e7591eddf7058cb06edd5a4dd`.
The averaged five-best checkpoint SHA-256 is
`b09151b998dfefa2547b6cc7920092448417b214f94b3203483ce74642e3c134`.

The GPU checks included power use. The highest observed samples were 240.76 W
for GPU0 and 237.30 W for GPU1. GPU0 reached 85 C. GPU1 reached 77 C. Short
software thermal slowdown occurred. Training continued without an error.

## Full evaluation

Inference used the averaged five-best checkpoint. It made 1,659 of 1,659
evaluation WAV files in 76 seconds. All files are mono and 24 kHz. The
validator found no empty file, non-finite sample, or clipping warning. Duration
is 0.811 to 7.723 seconds. Median RTF is 0.00778.

The five-voice listening set also passed the automatic WAV checks. It contains
Lada, zero-shot Dmytro, and three Common Voice embeddings. Each voice uses this
sentence:

> Кам'янець-Подільський - місто в Хмельницькій області України, центр
> Кам'янець-Подільської міської об'єднаної територіальної громади і
> Кам'янець-Подільського району.

Automatic checks do not measure metallic timbre. A human listening check must
confirm if the new audio preparation reduced this issue.

## 100k continuation

The continued run resumed from the 25k checkpoint. The target was 100,000
total iterations. It did not add 100,000 iterations to the first run.

The command used both RTX 3090 GPUs, FP32, 2,000,000 batch bins, and 8 data
workers. It completed at 22:11 Europe/Kyiv on 25 July 2026. The continued
process took 81,938 seconds. It returned exit status 0. The log has no NaN,
OOM, or critical runtime error.

The final validation generator loss is 57.538. The final validation mel loss
is 39.769. The best generator loss is 56.459 at 97k. The best mel loss is
38.897 at 97k. The best alignment loss is 4.204 at 99k. The final
discriminator loss is 1.833.

The maximum cached GPU memory was 20.670 GiB. The highest sampled power was
231.31 W for GPU0 and 241.60 W for GPU1. GPU0 reached 86 C. GPU1 reached
78 C. GPU0 had software thermal slowdown. The process did not report a
hardware thermal shutdown.

The epoch 100 checkpoint SHA-256 is
`9a5cfbde6c1e8133dff95278b1920d1ed596836bcbd4c5c1ef5ac35f611c4384`.
The new averaged five-best checkpoint SHA-256 is
`7ad913443283cab76db31c9ad058dcdec43f87dce148dd29a6cd660920208897`.

Inference used the new averaged checkpoint. It made 1,659 of 1,659 evaluation
WAV files in 77 seconds. All files passed the automatic checks. Duration is
0.768 to 8.224 seconds. Median RTF is 0.00799. The validator found no clipping
warning.

The 100k five-voice set also passed all automatic WAV checks. Each voice uses
the required sentence about Kamianets-Podilskyi. The set contains Lada,
zero-shot Dmytro, and three Common Voice embeddings.

## Next action

Listen to the five files in
`eval/generated/five_voice_enhanced_v2_100k/`. Record if the metallic artifact
is present in each voice.
