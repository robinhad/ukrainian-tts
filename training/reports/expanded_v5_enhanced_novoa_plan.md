# Enhanced Non-VOA Fine-Tune

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Objective

Train JETS for 100,000 new optimizer steps. Start from the best expanded-v4
checkpoint at epoch 28. Use the enhanced expanded-v3 audio. Do not use VOA
audio.

The run loads model weights only. It makes new optimizer, scheduler, epoch, and
step states. It does not load the old normalization modules. It calculates new
speech, pitch, and energy statistics for the filtered corpus.

## Data

The expected corpus has 74,156 utterances and 92.266 hours. The training split
has 71,334 utterances and 89.223 hours. The development split has 1,404
utterances and 1.514 hours. The evaluation split has 1,418 utterances and 1.529
hours.

The training audio uses the expanded-v3 enhanced copy. This copy has boundary
silence trim, DeepFilterNet3, a high-pass filter, light de-essing, gentle
compression, and two-pass EBU R128 normalization.

The speaker-conditioning set has 37,078 vectors from audio before enhancement.
It has 37,078 vectors from audio after enhancement. The pipeline reuses
compatible vectors from expanded-v3. It reuses one changed FLEURS vector from
expanded-v4.

## Commands

Prepare the filtered corpus and its statistics:

```sh
training/scripts/prepare_expanded_v5_enhanced_novoa.sh
```

Run the 100-step dual-GPU smoke gate:

```sh
training/scripts/run_expanded_v5_enhanced_novoa_smoke.sh
```

Start 100,000 new iterations on both GPUs:

```sh
training/scripts/launch_expanded_v5_enhanced_novoa_training.sh 100000
```

The monitor writes GPU use, power, VRAM, temperature, losses, free disk space,
and the Kyiv ETA at intervals of not more than 15 minutes. The disk cleanup can
start only when free space is 30 GiB or less.

## Gates

The full run cannot start until the readiness report and smoke inference report
have `PASS` status. The process must stop for NaN, a repeated OOM error, a GPU
temperature of 90 degrees C or more, or an unresolved disk limit.
