# Expanded v9 scratch training plan

## Objective

This run will make a new JETS model. It will use all retained audio, including
VOA. It will start from random model weights. It will not load a model,
optimizer, scheduler, token list, or statistics from an earlier training run.

## Data

The retained manifest has 208,867 utterances and approximately 469.76 hours.
It has 134,711 VOA utterances. The input files are the preserved canonical
boundary-trimmed files. The process does not use the deleted derived audio
copies.

The fixed audio sequence is:

1. Use the existing boundary trim.
2. Apply ClearerVoice MossFormer2_SE_48K.
3. Apply Sidon.
4. Apply light de-essing.
5. Apply the FFmpeg de-click filter and peak limiter.
6. Apply DeepFilterNet3.
7. Apply RNNoise with an 85 percent wet mix.
8. Apply a peak-safe linear loudness match to the source.
9. Keep the exact sample count and write 24 kHz mono PCM 24-bit WAV.

The process does not apply compression. It does not apply a second silence
trim. It does not apply VAD removal.

## Speaker embeddings

The process calculates all speaker embeddings again. It uses the established
stable hash and speaker-stratified assignment. It uses pre-enhancement audio
for 104,433 files. It uses final processed audio for 104,434 files. The total
difference is one file because the corpus size is odd.

## ESPnet preparation

The process makes new ESPnet data directories. It makes a new phoneme token
list and new speech, pitch, and energy statistics. It then runs 100 scratch
smoke iterations and inference. Full training can start only if these gates
have PASS status.

## Training

Full training uses two RTX 3090 cards and FP32. It uses 1,000 iterations in
each epoch and 500 epochs. Thus, the target is 500,000 optimizer iterations.
The initial `batch_bins` value is 2,000,000. The run preserves checkpoints at
25K, 50K, 75K, 100K, 200K, 300K, 400K, and 500K.

The training monitor writes iteration, metrics, GPU power, GPU memory, disk
space, and a Kyiv ETA every 15 minutes. The full-pipeline monitor writes a
record every 30 minutes. A second preparation monitor calculates its ETA from
processed audio hours. This method accounts for differences in file duration
between data sources. The end-to-end estimate uses this duration-aware value.
It adds 114 hours for the remaining ESPnet preparation, scratch training, and
evaluation until direct training measurements become available. The disk
threshold is 30 GiB.

## Start command

```bash
setsid training/scripts/run_expanded_v9_with_voa_pipeline.sh \
  > training/reports/expanded_v9_cascade_with_voa_pipeline.log 2>&1 &
```

The scripts write results atomically where the audio processor supports this
operation. They preserve source audio and manifests. The deleted derived audio
copies can be made again from these preserved inputs.

After all gates pass, the finalizer runs the full test suite. It writes the
measured final status to `training/reports/final_status.md`. It commits only
the selected small reports. It uses `codex <codex@openai.com>` as the commit
author. It then pushes the commit to the `autotrain` branch.
The default GitHub token file is `/home/ballvan/Projects/tts-token.txt`. Set
`TTS_GITHUB_TOKEN_FILE` to use a different token file. The helper does not
write the token to a Git remote or a report.
