# Expanded-v3 500K run

## Goal

The pipeline will train JETS for 500,000 iterations. It will use two GPUs.
It will use the complete accepted expanded-v3 dataset.

## Long audio

The VOA v4 process saves speech parts that are longer than 20 seconds. A
second process will split these files at low-energy points. Each new part
will be from 2 to 20 seconds long.

The process will use the pinned Parakeet model to create text for each part.
It will apply the same confidence and Ukrainian-script checks as the main
VOA process. It will add accepted parts to the training dataset.

## Audio preparation

Training audio will use the clean processing path. The path trims boundary
silence. It applies DeepFilterNet3, a high-pass filter, light de-essing,
gentle compression, and two-pass EBU R128 normalization.

## Training

The training target is 500,000 iterations. One epoch has 1,000 iterations.
The maximum epoch is 500. The process uses a new experiment directory:

`exp_expanded_v3/tts_jets_uk_24k_expanded_v3_500k`

The process will save milestone copies at 1K, 5K, 15K, 25K, 50K, 100K,
200K, 300K, 400K, and 500K.

## Monitoring

The chain writes GPU utilization, VRAM, power, temperature, disk space,
stage, metrics, and Kyiv ETA. The maximum status interval is 15 minutes.
The readable live status is in
`reports/expanded_v3_500k_visible_status.log`. The tmux session name is
`expanded_v3_visible_monitor`.

## Readiness exception

The readiness report will keep the actual PASS or FAIL result. The user
authorized training on the available recovered dataset even if the old
800-hour VOA gate remains FAIL.
