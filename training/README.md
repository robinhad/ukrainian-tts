# Ukrainian JETS training

See [Audio Enhancement Backend Review](reports/audio_enhancement_backend_review.md)
for the matched source-audio comparison of DeepFilterNet3, Sidon, Resemble
Enhance, and MossFormer2_SE_48K.

This directory contains the reproducible ESPnet2 GAN-TTS pipeline. The pipeline
is separate from the legacy multi-speaker inference code. It does not use a
verbalizer or a separate stress model. Each versioned iteration fixes its audio
processing profile.

The project documents use ASD-STE100 Simplified Technical English style. No
approved STE checker has certified these documents.

## Final de-clicking and peak limiting

Each new speech-enhancement run uses one final FFmpeg stage. The stage uses
32-bit planar floating-point audio for the filters. It does not set a sample
rate or a channel count. Thus, it keeps these properties from its enhanced
input. An earlier model-specific stage can still make a 24 kHz mono signal.

The final filter graph is:

```text
aformat=sample_fmts=fltp,adeclick=w=55:o=75:a=2:t=4:b=2,alimiter=limit=0.891251:attack=5:release=80:level=false:latency=true
```

The stage writes a PCM 24-bit WAV. It writes a temporary file in the output
directory. It validates the temporary file and then renames it atomically. It
does not permit the source path as the output path.

The default configuration is
`training/conf/audio_postprocess.yaml`. A pipeline command accepts
`--postprocess-config` and `--ffmpeg`. The value `auto` uses the static FFmpeg
7.0.2 binary from the pinned `imageio-ffmpeg` package. You can also set
`UKTTS_FFMPEG` or give an explicit FFmpeg path. FFmpeg must have the
`alimiter` `latency` option.

Process one enhanced file:

```bash
training/.venv/bin/python training/scripts/declick_and_limit_audio.py file \
  --config training/conf/audio_postprocess.yaml \
  --input enhanced.wav --output final.wav --report final.json
```

Process a directory and keep its directory structure:

```bash
training/.venv/bin/python training/scripts/declick_and_limit_audio.py batch \
  --config training/conf/audio_postprocess.yaml \
  --input-dir enhanced --output-dir final --recursive \
  --report final/batch-report.json
```

The CLI also has one option for each filter value. A CLI value has priority
over the configuration file. Use `--overwrite` only when the command can
replace an existing output. The command never changes an input file.

## Expanded-v6 Sidon and de-essing iteration

This iteration trains for 100,000 new optimizer steps. It loads model weights
from the expanded-v5 best mel checkpoint at epoch 81. It makes new optimizer,
scheduler, epoch, and step states.

The completed 2026-08-11 corpus has 74,156 non-VOA utterances. That corpus uses
the boundary-trimmed input, Sidon, and light de-essing. It predates the final
de-click and limiter stage. A new preprocessing run adds the final stage and
uses a new configuration hash. It does not change the completed corpus.
Sidon has a fixed internal 50 Hz input filter.

Run these commands in sequence:

```sh
training/scripts/run_expanded_v6_sidon_preprocessing.sh
training/scripts/prepare_expanded_v6_sidon_deess_novoa.sh
training/scripts/run_expanded_v6_sidon_deess_novoa_smoke.sh
training/scripts/launch_expanded_v6_sidon_deess_novoa_training.sh 100000
training/scripts/finalize_expanded_v6_sidon_deess_novoa_100k.sh
```

For an unattended run, start the continuation after the preprocessing process.
Pass the preprocessing PID to this command:

```sh
training/scripts/continue_expanded_v6_pipeline.sh PREPROCESS_PID
```

The full-pipeline watchdog writes a phase and resource record every 30 minutes.
The training monitor supplies the iteration, loss metrics, and Kyiv ETA after
training starts.

The preprocessing command starts four workers on each RTX 3090 by default.
The training command uses both RTX 3090 cards. The monitors
write GPU use, power, VRAM, temperature, disk space, progress, and a Kyiv ETA
at intervals of not more than 15 minutes. See
`training/reports/expanded_v6_sidon_deess_novoa_plan.md` for the fixed gates.

The completed run used both RTX 3090 GPUs and finished 100,000 new steps with
exit code 0. The 25K, 50K, 75K, and 100K checkpoints each made 1,418 valid
fixed-set WAV files. The five-voice listening set also passed the automatic
checks. The final model SHA-256 is
`d4ed14c8bfb828c97fffb7ca33065a882c6ba342829eb5a00f3dda8b4205d9d5`.

Listen to the five files here:

```text
training/eval/generated/five_voice_expanded_v6_sidon_deess_novoa_100k/
```

## Expanded-v5 enhanced non-VOA iteration

This iteration trains for 100,000 new optimizer steps. It loads model weights
from the best expanded-v4 checkpoint at epoch 28. It resets the optimizer,
scheduler, epoch, and step states.

The corpus uses the expanded-v3 enhanced audio. It excludes all VOA records.
It keeps an exact 50/50 speaker-conditioning mix: 37,078 vectors use audio
before enhancement, and 37,078 vectors use audio after enhancement.

Run these commands in sequence:

```sh
training/scripts/prepare_expanded_v5_enhanced_novoa.sh
training/scripts/run_expanded_v5_enhanced_novoa_smoke.sh
training/scripts/launch_expanded_v5_enhanced_novoa_training.sh 100000
```

The full launcher uses both RTX 3090 cards. It writes loss, power, GPU use,
VRAM, temperature, disk space, and Kyiv ETA data at intervals of not more than
15 minutes. See `training/reports/expanded_v5_enhanced_novoa_plan.md` for the
fixed inputs and gates.

## Expanded-v4 trim-only iteration

This iteration tests the effect of simple audio preparation. It uses one
boundary-silence trim. It does not use DeepFilterNet, a high-pass filter,
de-essing, compression, or R128 normalization. The training files are 24 kHz
mono PCM WAV files.

The accepted corpus has 207,505 utterances and 467.127 hours. The run excludes
1,357 clipped utterances and 5 utterances that are shorter than 2 seconds. It
does not include audio that is longer than 20 seconds.

The speaker embedding set has 103,752 vectors from audio before the trim. It
has 103,753 vectors from audio after the trim. Each vector has 192 values.

Run the smoke preparation and the dual-GPU smoke test:

```sh
training/scripts/prepare_expanded_v4_trim_only.sh smoke
training/scripts/run_expanded_v4_trim_only_smoke.sh
```

Run the full preparation and the 100K fine-tune:

```sh
training/scripts/prepare_expanded_v4_trim_only.sh full
training/scripts/launch_expanded_v4_trim_only_training.sh 100000
```

The fine-tune loads model weights from expanded-v3 epoch 367. It makes new
optimizer, scheduler, epoch, and step states. Thus, `100000` means 100,000 new
optimizer steps. The run uses two RTX 3090 GPUs and FP32.

The monitor writes loss, GPU power, temperature, VRAM, free disk space, and a
Kyiv ETA every 15 minutes. The cleanup monitor does not remove data while free
disk space is more than 30 GiB.

After training, the launcher validates the 25K, 50K, 75K, and 100K milestones.
It then makes the fixed evaluation set and the five-voice 100K listening set.
See these documents for the current state:

```text
training/reports/expanded_v4_trim_only_plan.md
training/reports/expanded_v4_trim_only_completion_checklist.md
```

Make the per-dataset listening set after the 100K checkpoint exists:

```sh
CUDA_VISIBLE_DEVICES=0 training/.venv/bin/python \
  training/scripts/generate_per_dataset_listening_eval.py \
  --manifest training/data/expanded_v4_trim_only/hybrid_manifest/all.parquet \
  --xvector-root training/dump_expanded_v4_trim_only/xvector \
  --config training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/config.yaml \
  --checkpoint training/exp_expanded_v4_trim_only/tts_jets_uk_24k_expanded_v4_trim_only_ft367_100k/milestones/100k.pth \
  --output training/eval/generated/per_dataset_expanded_v4_trim_only_100k \
  --report training/reports/per_dataset_expanded_v4_trim_only_100k.json \
  --count-per-dataset 10
```

The command makes 10 synthesized WAV files for each source dataset. It also
makes 10 independent copies of the paired trim-only reference files. Five items in each
dataset use pre-trim speaker embeddings. Five items use post-trim speaker
embeddings. Use `feedback_template.tsv` in the output directory to record the
listening results.

Make a source-audio-only A/B review of the two processing revisions:

```sh
training/.venv/bin/python \
  training/scripts/build_per_dataset_source_audio_review.py \
  --previous-manifest training/data/expanded_v3/manifests/all.parquet \
  --current-manifest training/data/expanded_v4_trim_only/hybrid_manifest/all.parquet \
  --output training/eval/generated/per_dataset_source_audio_ab_v3_enhanced_vs_v4_trim_only \
  --report training/reports/per_dataset_source_audio_ab_v3_enhanced_vs_v4_trim_only.json \
  --count-per-dataset 10
```

This source review does not contain synthesized audio. For each dataset, it
contains 10 matched files from the previous enhanced revision and 10 matched
files from the current trim-only revision. The WAV files are independent
copies. They are not symlinks. Use its `feedback_template.tsv` to record the
preferred processing revision and all audible problems.

## Expanded-v3 iteration

The expanded-v3 pipeline uses all permitted Ukrainian sources. The source
registry uses a default-deny policy. A record must have a permitted audio
license. A code license does not prove the audio license.

The pipeline applies these audio operations to each training copy:

1. Remove leading and trailing silence.
2. Apply DeepFilterNet3.
3. Apply a 70 Hz high-pass filter.
4. Apply light de-essing.
5. Apply gentle compression.
6. Apply two-pass EBU R128 normalization.
7. Apply the final de-click and peak-limit graph.
8. Encode the final WAV as PCM 24-bit.

See `training/reports/audio_enhancement_steps.md` for the exact sequence,
filter values, output format, and validation rules.

The source audio does not change. The training manifest points only to the clean
copy. ECAPA uses raw audio for 50 percent of the records. It uses clean audio for
the other 50 percent.

The unlabeled-audio stage limits each diarization input to 60 seconds. This
limit controls GPU memory for long source files. The stage writes a progress
record after each source file. A restart uses the progress records and a
complete existing collection.

The pipeline reads the Hugging Face token from
`/home/ballvan/Projects/hf_token.txt`. It does not export or copy the token.
Set `MDC_COMMON_VOICE_ROOT` to the extracted direct Common Voice directory.
The directory must contain `validated.tsv` and `clips/`.

Run the expanded smoke stages:

```sh
training/scripts/prepare_expanded_v3.sh smoke
training/scripts/run_expanded_v3_smoke.sh
```

Run the full preparation:

```sh
MDC_COMMON_VOICE_ROOT=/path/to/uk \
  training/scripts/prepare_expanded_v3.sh full
```

The full preparation makes
`training/reports/expanded_v3_full_scale_readiness.json`. The old 800-hour VOA
gate has `FAIL`. The user authorized the 500K run on the accepted dataset.

Start or resume the 500,000-iteration run:

```sh
training/scripts/launch_expanded_v3_training.sh 500000
```

The launcher uses both RTX 3090 cards. It writes TensorBoard event files. It
reports GPU load, VRAM, temperature, power, disk space, and Kyiv ETA. The report
interval is 30 minutes for the current long run. The safety monitor checks more
often. The process stops when free disk space is less than 30 GiB.

The completed run used 208,867 utterances and 469.755 hours. The active
manifest contains audio from 2 to 20 seconds. It does not contain deferred
audio that is longer than 20 seconds. Training completed with exit code 0.

The data preparation keeps each processed source batch while free disk space is
more than 30 GiB. At 30 GiB, it removes the oldest processed batch. It stops the
removal when free disk space is more than 30 GiB. It does not remove an
unprocessed batch.

### Evaluate the 500K model

Do not run the finalizer while training is active. The finalizer stops if the
training chain does not have the `COMPLETE` state. The automatic finalizer ran
after the 500K checkpoint became available. Use this command to repeat it:

```sh
training/scripts/finalize_expanded_v3_500k.sh
```

The finalizer uses GPU 0 for the fixed set of 1,418 evaluation utterances. It
uses GPU 1 for the five-voice listening set. All five voices use the required
sentence about Kamianets-Podilskyi. The finalizer validates the WAV files and
writes these reports:

```text
training/reports/expanded_v3_inference_500k.json
training/reports/five_voice_expanded_v3_500k.json
training/reports/expanded_v3_500k_artifacts.sha256
```

The finalizer completed with exit code 0. It validated 1,418 fixed-set files
and five listening files. The user hears rasp and intermittent robotic sound.
Release status is `FAIL`.

The A/B review has five voices from epochs 367, 443, 488, and 500. Use this
guide:

```text
training/reports/expanded_v3_500k_listening_followup.md
```

## Fixed versions

- ESPnet `v.202604-patch1`, commit `cff0a07`
- eSpeak-ng `1.52.0`, commit `4870adf`
- PyTorch/torchaudio `2.9.1`, CUDA 12.8 wheels
- Dataset `speech-uk/opentts-lada`, revision
  `729289b58251da4a21ce85f8808dd908f28b0d7f`

## Prepare the environment

```bash
cd training
./scripts/bootstrap_env.sh
source ./activate.sh
python scripts/validate_espeak.py --write-hash
python -m training.frontend.snapshot \
  --cases tests/frontend/regression.tsv \
  --snapshot tests/frontend/expected_phonemes.json \
  --cache data/phonemes.sqlite3 --update
```

The bootstrap installs files only in `training/.venv` and `training/vendor`. It
applies `patches/espnet-espeak-ng-ukrainian.patch` to the pinned ESPnet source.

## Audio trimming

Audio preparation removes leading and trailing silence from each model copy. It
does not change the raw OGG file. The detector uses relative frame RMS with these
fixed values:

- Threshold: 40 dB below the maximum frame RMS.
- Frame length: 1024 samples.
- Hop length: 256 samples.
- Safety padding: 100 ms at each active boundary.

The manifest records the original duration, the removed duration, both trim
boundaries, and the trim configuration hash. Trimming does not use VAD,
normalization, denoise, compression, or mastering.

## Smoke cycle

The smoke cycle uses GPU0. The code selects GPU0 by its UUID. `run_logged.py`
records the exit status for each long command. It prints a heartbeat at intervals
of five minutes or less. The heartbeat contains the Europe/Kyiv time and the ETA.

```bash
cd training
source ./activate.sh
./scripts/run_smoke_test.sh
```

Use these commands to resume an ESPnet stage:

```bash
./espnet_recipe/run.sh --stage 1 --stop_stage 6
./espnet_recipe/run.sh --stage 7 --stop_stage 7 --train_args "--max_epoch 1"
./espnet_recipe/run.sh --stage 8 --stop_stage 8 \
  --train_args "--max_epoch 1" \
  --inference_model train.total_count.ave.pth
```

Do not run full-corpus preparation or long training unless every critical gate in
`reports/scale_readiness.md` is `PASS`.

## Local inference

The entry point finds the pinned eSpeak-ng installation in this repository. You
can run it from a new shell after the bootstrap. Source `activate.sh` before you
run a recipe command.

```bash
training/.venv/bin/python -m training.inference.synthesize \
  --text "Український синтез мовлення працює офлайн." \
  --output training/eval/generated/example.wav \
  --config training/exp_full/tts_jets_uk_24k_full/config.yaml \
  --checkpoint training/exp_full/milestones/25k.pth
```

Keep the raw WAV separate from the publication WAV.

## Full-corpus training after MVP PASS

Calibration selected `batch_bins=3000000`. This value kept approximately 12% of
the VRAM free for each process. The 200-iteration AMP test increased the validation
loss. Thus, full training uses FP32. Smoke tests use one RTX 3090. Full training
uses both RTX 3090 cards. Use these commands to resume the same experiment:

```bash
cd training
./scripts/run_full_training.sh 1000
./scripts/run_full_training.sh 25000
./scripts/run_milestone_inference.sh latest.pth
./scripts/run_full_training.sh 50000
./scripts/run_full_training.sh 100000
```

Use the same command for the 200k and 400k targets. Continue only if the fixed-set
listening results and the inference diagnostics improve.

## Restart with trimmed audio

These commands keep the first full-corpus experiment. They make new data,
statistics, and checkpoints with the `trimmed` suffix.

```bash
training/scripts/run_trimmed_smoke_test.sh
training/scripts/prepare_trimmed_full.sh
BATCH_BINS=3800000 training/scripts/run_trimmed_training.sh 25000
training/scripts/finalize_trimmed_training.sh 25epoch.pth 25k
```

Calibrate `BATCH_BINS` before a long run. Use both GPUs. Increase the value until
each GPU uses approximately 85--90% of its VRAM. Keep at least 10% free VRAM. Do
not increase the GPU power limit and do not overclock the GPUs. Keep
`cudnn_benchmark=false` for this variable-length workload. The benchmark mode
made the calibration slower.

The finalization command keeps the 25k checkpoint. It runs inference on all 180
fixed evaluation items. It validates each WAV and makes one local inference
example and one 20-item listening set. It also summarizes sampled GPU power,
utilization, temperature, and device memory. Run this command only after the
25k training command exits with status 0.

The short calibration selected 4,000,000 batch bins. The complete data sequence
later used 23.19 GiB on one GPU and left only 935 MiB free. The run resumed from
the 3k checkpoint with 3,800,000 batch bins. This value keeps a safer device
memory reserve for the long run.

## Make the listening set

This command makes links to 20 raw references and the 15k, 23k, and 25k outputs.
It does not copy the large audio files. Listen to the same utterance in each
directory before you select a release checkpoint.

```bash
training/.venv/bin/python training/scripts/build_listening_set.py \
  --manifest training/data/full/manifests/eval.parquet \
  --raw-dir training/data/full/raw \
  --candidate jets_15k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_15k/eval/wav \
  --candidate jets_23k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_23k/eval/wav \
  --candidate jets_25k=training/exp_full/tts_jets_uk_24k_full/decode_jets_milestone_25k/eval/wav \
  --count 20 \
  --output training/eval/generated/listening_25k
```

## Get the training status

Use this command to get the progress, the Europe/Kyiv ETA, the checkpoint list,
the error count, and the status of each GPU. The GPU status includes power draw,
power limit, temperature, memory use, and compute use.

```bash
python scripts/training_status.py \
  --log exp_full/tts_jets_uk_24k_full/train.log \
  --output reports/training_status.jsonl
```

## View the training metrics

ESPnet writes TensorBoard event files through PyTorch. It does not use the
TensorFlow training runtime. Use this command from the repository root:

```bash
training/.venv/bin/tensorboard \
  --logdir training/exp_full_trimmed/tts_jets_uk_24k_trimmed/tensorboard
```

The completed trimmed run has 29 train scalar tags and 16 validation scalar
tags through step 25,000. The validation mel loss fell from 56.198 at 1k to
33.549 at 25k.

## Train with Common Voice and Lada

This iteration uses Common Voice and Lada audio. It uses one 192-value ECAPA
embedding for each utterance. It uses both RTX 3090 GPUs for training.

The public Common Voice source does not contain stable client IDs. Therefore,
the data pipeline does not claim a common speaker identity for two Common Voice
files. The pipeline uses `lada` as the speaker ID for Lada data.

Run these commands from the repository root:

```bash
training/scripts/run_multispeaker_smoke_test.sh
training/scripts/prepare_multispeaker_full.sh
training/scripts/calibrate_multispeaker_batch.sh 4500000 200
training/scripts/run_multispeaker_training.sh 25000
training/scripts/finalize_multispeaker_training.sh 25epoch.pth 25k
```

The preparation command makes 24 kHz mono PCM WAV model copies. It removes only
leading and trailing silence. It uses a 40 dB relative frame-RMS threshold and
100 ms boundary padding. It does not use MFA, VAD, denoise, compression,
de-essing, loudness normalization, or mastering.

The preparation command uses the pinned Common Voice and Lada revisions. Set
`DMYTRO_MANIFEST` only when a valid Dmytro manifest and its audio files are
available. If this variable is not set, Dmytro is not in the train set.

The selected long-run value is `batch_bins=2000000` with expandable PyTorch
allocator segments. Tests at 4,500,000, 3,800,000, 3,400,000, 3,000,000, and
2,500,000 did not keep the required VRAM reserve on later dynamic batches. The
training command uses FP32 and writes TensorBoard metrics through PyTorch.

Use this command to run the external five-minute monitor:

```bash
training/scripts/monitor_multispeaker_training.sh \
  TRAIN_PID 25000 300
```

The monitor writes current progress, power, memory, temperature, and the Kyiv
ETA to `training/reports/training_status_multispeaker.jsonl`.

The finalization command starts only after the 25k checkpoint exists. It
validates all 1,677 fixed evaluation files for the 1k, 5k, and 25k
checkpoints. It also makes a listening set with 10 Common Voice references and
10 Lada references. This set contains all three model candidates. The command
makes one Lada example and one zero-shot Dmytro example. The zero-shot example
does not make Dmytro a trained speaker. It makes a release candidate with
model, frontend, statistics, reports, licenses, cards, and SHA-256 checksums.

Use this command to view the current model metrics:

```bash
training/.venv/bin/tensorboard \
  --logdir training/exp_multispeaker_full/tts_jets_uk_24k_multispeaker/tensorboard
```

The corrected run completed 25,000 iterations on both RTX 3090 GPUs. It
returned exit status 0. The 25k checkpoint SHA-256 is
`395ccaba7e6837a60257a622b8d9ce0352246e41f728273e92d09c41b444f445`.
The lowest validation mel loss is 40.226 at 24k. This value is 32.7 percent
below the 1k value.

The 1k, 5k, and 25k checkpoints each made all 1,677 fixed evaluation WAV files.
All files passed the automatic 24 kHz mono, finite, nonzero, duration, and
clipping checks. The release candidate is:

```text
training/releases/uk-tts-jets-multispeaker-25k-rc/
```

Listen to the balanced set before you promote this candidate:

```text
training/eval/generated/listening_multispeaker_25k/
```

## Generate the five-voice metallic-timbre review

The user reported a metallic timbre in the 25k model. This command makes the
same long sentence with Lada, Dmytro zero-shot, and three representative Common
Voice embedding clusters:

```bash
training/scripts/generate_five_voice_listening_eval.sh
```

The script selects Common Voice cluster medoids. It does not select extreme
embedding outliers. It validates all five WAV files and writes:

```text
training/eval/generated/five_voice_metallic_review/
training/reports/five_voice_listening_eval.json
```

If all five files have the same artifact, review the shared JETS generator and
the mixed acoustic data. If the artifact changes by voice, review the speaker
embedding and its source audio.

## Train with enhanced audio v2

This data version uses DeepFilterNet3. It then applies a 70 Hz high-pass
filter, light de-essing, and gentle compression. It applies two-pass EBU R128
normalization with a -23 LUFS target and a -1 dBTP true-peak limit. It writes
new 24 kHz mono PCM WAV files. It does not change the source audio.

Run these commands from the repository root:

```bash
training/scripts/prepare_multispeaker_enhanced_v2.sh
training/scripts/run_multispeaker_enhanced_v2_training.sh 25000
training/scripts/run_multispeaker_enhanced_v2_training.sh 100000
MILESTONE_TAG=100k MODEL_FILE=train.total_count.ave.pth \
  training/scripts/generate_five_voice_enhanced_v2_eval.sh
```

The completed run used both RTX 3090 GPUs and FP32. It resumed at 25k and
stopped at 100,000 total iterations. It did not add 100,000 iterations to the
first run. The log has no NaN, OOM, or critical runtime error. The fixed
evaluation made 1,659 valid WAV files. The 100k five-voice script made five
valid WAV files. It uses the required sentence about Kamianets-Podilskyi.

The best validation generator loss is 56.459 at 97k. The best validation mel
loss is 38.897 at 97k. The averaged five-best checkpoint SHA-256 is
`7ad913443283cab76db31c9ad058dcdec43f87dce148dd29a6cd660920208897`.

Listen to the five 100k files here:

```text
training/eval/generated/five_voice_enhanced_v2_100k/
```

Use this command to view the new metrics:

```bash
training/.venv/bin/tensorboard \
  --logdir training/exp_multispeaker_enhanced_v2/tts_jets_uk_24k_multispeaker_enhanced_v2/tensorboard
```

TensorBoard reads PyTorch event files. The training process does not use the
TensorFlow runtime.
