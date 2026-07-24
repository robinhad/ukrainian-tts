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

## Next action

Run:

```bash
training/scripts/run_multispeaker_enhanced_v2_training.sh 25000
```

