# Expanded-v3 500K run

This report uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify the report.

## Goal

The pipeline trained JETS for 500,000 iterations. It used two RTX 3090 GPUs.
It used the accepted expanded-v3 dataset.

## Audio scope

The active manifest contains audio from 2 to 20 seconds. It does not contain
the deferred segments that are longer than 20 seconds. The user selected this
scope before the training run.

The training audio uses the clean processing path. The path trims boundary
silence. It applies DeepFilterNet3, a high-pass filter, light de-essing, gentle
compression, and two-pass EBU R128 normalization.

## Training result

The experiment directory is:

`exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k`

The process saved milestone copies at 1K, 5K, 15K, 25K, 50K, 100K, 200K,
300K, 400K, and 500K. It completed 500,000 iterations with exit code 0.

## Monitoring result

The monitor wrote GPU utilization, VRAM, power, temperature, disk space,
metrics, and Kyiv ETA. The visible interval was 30 minutes. The safety monitor
used a shorter interval. The minimum reported disk reserve was 48.93 GiB. The
stop threshold was 30 GiB.

## Evaluation result

The finalizer used GPU 0 for the 1,418-item fixed set. It used GPU 1 for the
five-voice listening set. All 1,423 WAV files passed the automatic checks.

## Readiness exception

The old VOA target is 800 hours. The accepted VOA subset has 377.489 hours.
The readiness report keeps this result as `FAIL`. The user authorized this
training run on the accepted dataset.
