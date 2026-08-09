# Audio Enhancement Backend Review

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Purpose

This review compares source audio from all nine approved training datasets. It
does not compare synthesized audio. It includes VOA audio.

The review uses 10 matched utterances from each dataset. It makes eight files
for each utterance. Thus, the complete set has 90 utterances and 720 WAV files.
All WAV files are real files. The process does not make symbolic links.

## Profiles

1. `trim_only` is the boundary-trimmed training input.
2. `historical_dfn3_compressed` is the prior enhanced v3 audio. This profile
   contains compression.
3. `dfn3_no_compression` uses DeepFilterNet3 and no compression.
4. `sidon_no_compression` uses Sidon and no compression.
5. `sidon_deess_only` uses boundary trimming, Sidon, and light de-essing. It
   does not use a post-processing high-pass filter, loudness normalization,
   compression, or limiting. Sidon has its fixed internal 50 Hz input filter.
6. `resemble_denoise_no_compression` uses the Resemble denoiser and no
   compression.
7. `resemble_full_no_compression` uses the complete Resemble Enhance model and
   no compression.
8. `mossformer2_no_compression` uses MossFormer2_SE_48K and no compression.

Five normalized enhanced profiles use the same final process. This process
applies a 70 Hz high-pass filter and light de-essing. It then applies accurate
two-pass EBU R128 normalization. The target values are -23 LUFS, LRA 7, and
-1 dBTP. The `sidon_deess_only` profile does not use this final process. All
profiles use 24 kHz mono PCM WAV files.

## Fixed Software

- Sidon code commit: `c8cde2b24e4c77c599ad43a9871140cdc9beeffa`
- Sidon model revision: `b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52`
- W2V-BERT feature extractor revision: `da985ba0987f70aaeb84a80f2851cfac8c697a7b`
- Resemble Enhance code commit: `8e978149bfe8abab3eb77d965d579a111afdb0ff`
- Resemble model revision: `4e3510ce4a8391159f665903544c5150bee7b2cb`
- DeepSpeed inference dependency: `0.18.7`
- ClearVoice code commit: `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`
- MossFormer2 model revision: `eff8c97925c8bec812af707814b3e5d777fd4503`

Each run records the SHA-256 value of each downloaded inference weight in its
backend summary and WAV metadata.

## Commands

Run the tests:

```bash
PYTHONPATH=. training/.venv/bin/pytest -q \
  training/tests/test_enhancement_backend_review.py
```

Make the isolated environment:

```bash
training/scripts/setup_enhancement_review_env.sh
```

Run the complete review:

```bash
training/scripts/run_enhancement_backend_review.sh
```

Add or refresh only the Sidon and de-essing profile:

```bash
training/scripts/run_sidon_deess_listening_review.sh
```

The output directory is
`training/eval/generated/per_dataset_train_audio_enhancement_review_v2`.
The directory contains `manifest.tsv`, `feedback_template.tsv`, metadata, and
the copied WAV files. The resource log records GPU use, GPU memory, GPU power,
and temperature every 30 seconds.

The run must stop if free disk space is less than 30 GiB. The operator can
delete source caches only after this threshold condition occurs.

## Completed Run

The run completed on 2026-08-10. The final validation status is PASS.

- The set has 720 WAV files and 720 metadata files.
- Each profile has 90 WAV files.
- The set has no symbolic links.
- The files use 185,551,584 bytes.
- The compression policy has no violations.
- The complete training test suite has 118 passed tests.
- The minimum measured free disk space was 124 GiB.
- The Sidon and de-essing run used two physical GPUs.
- The maximum measured value was 221.53 W on GPU 0.
- The maximum measured value was 212.97 W on GPU 1.

See `per_dataset_train_audio_enhancement_review_v2.json` for the validation
result. See `enhancement_backend_review_resources.csv` for the resource data.
See `sidon_deess_listening_resources.csv` for the new profile resource data.
