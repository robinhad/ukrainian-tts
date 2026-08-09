# Sidon and De-essing Fine-Tune

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Objective

Train JETS for 100,000 new optimizer steps. Start from the expanded-v5 best mel
checkpoint at epoch 81. Load model weights only. Make new optimizer, scheduler,
epoch, and step states.

## Fixed data

Use the expanded-v5 non-VOA IDs and splits. The expected corpus has 74,156
utterances. The training split has 71,334 utterances. The development split has
1,404 utterances. The evaluation split has 1,418 utterances.

Use the existing 24 kHz mono boundary-trimmed audio as the Sidon input. Apply
Sidon. Then apply light de-essing. Write 24 kHz mono PCM16 WAV files.

Do not apply these operations:

- DeepFilterNet
- an additional high-pass filter
- compression
- EBU R128 loudness normalization
- limiting

Sidon has a fixed internal 50 Hz input filter. This filter is part of Sidon.

## Speaker embeddings

Use an exact 50/50 input mix. Use audio before Sidon for 37,078 vectors. Use
Sidon and de-essed audio for 37,078 vectors. Keep the expanded-v5 assignment.
Reuse the 37,078 compatible raw vectors. Calculate the 37,078 clean vectors
again.

## Model

Use this initial checkpoint:

```text
training/exp_expanded_v5_enhanced_novoa/tts_jets_uk_24k_expanded_v5_enhanced_novoa_ft_v4e28_100k/best_checkpoints/81epoch.pth
```

Its SHA-256 is
`ec8f6ec4e6ac330b796d91cb1c55239d81b1f4d8bb8aefd399b6c44b4206ff90`.
Use two GPUs, FP32, `batch_bins=2000000`, and `accum_grad=1`.

## Gates

The full training cannot start until all these conditions have PASS status:

- The free disk space is more than 30 GiB.
- All 74,156 audio files pass processing and dataset validation.
- The corpus has no VOA records.
- The exact 50/50 embedding archive is valid.
- The token list and all statistics exist.
- The two-GPU 100-step smoke run has no NaN or critical runtime error.
- Smoke inference makes a valid 24 kHz mono WAV.

## Monitoring

Write a status record at least once every 15 minutes. Include the iteration,
losses, GPU use, VRAM, power, temperature, free disk space, and Kyiv ETA. Stop
preprocessing if free disk space is less than 30 GiB. Do not remove data before
this threshold. Start four Sidon workers on each GPU by default. Reduce this
value only if a resource check finds an OOM error or instability.

Save the 25K, 50K, 75K, and 100K checkpoints. Run the fixed evaluation set for
each checkpoint. Make five listening voices at 100K. Use the fixed
Kamianets-Podilskyi sentence for all five voices.
