# Non-VOA quality evaluation

Run all commands from the repository root. The evaluator measures quality. It
does not reject recordings or select checkpoints. All threshold flags are
report-only. HNR is not a metric or a rejection condition in this evaluator.

## Fixed data

Download the eight sources at the revisions already recorded in this branch:

```bash
python training/scripts/download_non_voa.py
```

This restores the original CV22 source used by the earlier local copy. It does
not download VOA. Download state is in
`training/data/non_voa_downloads/downloads.json`. Reconstructed counts can
differ from the previous host. Record measured counts before training.

Freeze two panels from a canonical, unenhanced manifest with existing splits:

```bash
python -m training.quality select --manifest /path/to/all.parquet \
  --output training/quality_runs/v10/panels --per-source 100
```

The processing panel uses the training split. The checkpoint panel uses only
the evaluation/test split. Each panel takes the same number from each available
source, capped at 100 and at the smallest eligible source. Sorting uses a fixed
hash seed. Held-out items sharing normalized text or audio bytes with train/dev
are excluded. The files, transcripts, split, original audio hashes, source
counts, and panel hashes are frozen. Existing panels cannot be overwritten.
Keep the complete held-out split out of training; a panel is a measurement
subset, not a new training split. This is utterance/text holdout, not a claim
that speakers are unseen.

## Metrics

The same evaluator handles processing outputs and synthesized checkpoint WAVs:

- [Microsoft SigMOS](https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos):
  overall, speech, noise, coloration, discontinuity, plus loudness and reverb.
- [Meta Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics):
  production quality (PQ).
- [Whisper](https://github.com/openai/whisper): Ukrainian transcription with
  fixed language and decoding settings. CER/WER compare against the supplied
  human transcript. Normalization uses NFC, lowercase, standardized apostrophes,
  punctuation/whitespace cleanup, and removal of stress marks. Digits are kept;
  there is no number verbalizer. CER excludes spaces. Corpus error rates use
  total edits divided by total reference characters/words.
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3):
  an additional multilingual ASR check supporting Ukrainian, with automatic
  language detection and the pinned checkpoint's default decoding. Audio is
  resampled to 16 kHz mono. `parakeet_text`, `parakeet_cer`, `parakeet_wer`, and
  edit/reference counts are saved separately; `cer`/`wer` remain Whisper scores.
  Both ASR systems use identical transcript normalization. Per-source,
  source-macro, corpus-weighted rates, and checkpoint deltas include Parakeet.
  Missing reference text has no CER/WER and contributes no corpus edits.
  ASR errors measure intelligibility and recognizer behavior, not acoustic MOS.
- [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb):
  cosine similarity to the original recording. This is speaker preservation,
  not proof of speaker identity.
- Signal diagnostics: clipping, relative duration changes, and local bursts
  above 6 kHz. The burst diagnostic combines spectral energy fraction, excess
  above a local median, and a signal floor. It is a heuristic; fricatives can
  trigger it. It is unavailable when the sample rate has no band above 6 kHz.

All learned quality scores and ECAPA similarity are also calculated on
overlapping 3-second windows, including the file tail. Each file lists its
worst three windows for each metric. Local times are relative to the candidate
audio. Windows from synthesized and original speech are not time-aligned.
Segment CER/WER is deliberately unavailable without time-aligned human text;
the whole transcript must not be compared to a short crop. Short windows and
all uncalibrated thresholds need human review.

## Run evaluation and comparisons

Set local model paths in `training/conf/quality.yaml`. Use the same configuration
and model versions for every result in a comparison. First score originals:

```bash
python -m training.quality evaluate \
  --panel training/quality_runs/v10/panels/heldout.jsonl \
  --label original --output training/quality_runs/v10/original --resume
```

Score a checkpoint's decoded WAVs (`<utterance_id>.wav`):

```bash
python -m training.quality evaluate \
  --panel training/quality_runs/v10/panels/heldout.jsonl \
  --wav-dir /path/to/decoded/wav --allow-extra-wavs \
  --checkpoint /path/to/25k.pth --training-config /path/to/config.yaml \
  --label 25k --output training/quality_runs/v10/25k --resume
python -m training.quality compare \
  --candidate training/quality_runs/v10/25k \
  --original training/quality_runs/v10/original \
  --previous /path/to/previous_quality --best /path/to/reviewed_best_quality \
  --output training/quality_runs/v10/25k
```

For the first checkpoint omit `--previous` and `--best` if no prior model exists.
The best baseline is supplied explicitly after review. No metric automatically
promotes a checkpoint. Comparison requires identical panel/config/model
provenance, complete coverage, and unchanged reference hashes. Missing model
dependencies or inference failures produce an incomplete result, not a proxy
score. Resume caches include audio, model, configuration, and checkpoint hashes.

Each result directory contains `per_file`, `segments`, `aggregate`, and `errors`
in both JSONL and CSV, plus `run.json` and resumable per-file caches. Comparisons
produce `comparison_per_file` and `comparison_aggregate` in JSONL/CSV.
Aggregates include per-source statistics and equally weighted source means,
alongside pooled statistics. All deltas are candidate minus baseline. Higher
quality/ECAPA is better; lower CER/WER/clipping/burst count is better. Duration
ratio is diagnostic and should be near one.

## Existing milestone integration

Every existing milestone script that invokes `synthesize_eval.py` can enable
quality evaluation through these environment variables (use absolute paths):

```bash
export UKTTS_QUALITY_PANEL="$PWD/training/quality_runs/v10/panels/heldout.jsonl"
export UKTTS_QUALITY_CONFIG="$PWD/training/conf/quality.yaml"
export UKTTS_QUALITY_ORIGINAL="$PWD/training/quality_runs/v10/original"
export UKTTS_QUALITY_PREVIOUS=/absolute/path/to/previous_quality
export UKTTS_QUALITY_BEST=/absolute/path/to/reviewed_best_quality
```

The same values can be passed as `--quality-*` arguments to
`training/scripts/synthesize_eval.py`. Quality output is saved beside the
existing structural evaluation report, with `_quality` appended. Structural
errors stop scoring. Threshold flags do not fail the checkpoint.

## Validation

```bash
python -m pytest -c training/pytest.ini training/tests/test_quality.py -q
```

These tests use synthetic signals and explicit test doubles. They validate
data isolation, diagnostics, output formats, coverage, cache invalidation, and
paired comparisons. They are not evidence of learned-model accuracy or of a
completed real audio experiment.

## Environment and SLURM execution

Install PyTorch/torchaudio 2.9.1 from the CUDA wheel index compatible with the
allocated GPU. The training bootstrap accepts `TORCH_INDEX_URL`; it preserves
SLURM's `CUDA_VISIBLE_DEVICES`. ARM builds of DeepFilterNet need Cargo/Rust.
Then run `bash training/scripts/bootstrap_quality.sh`. SigMOS code is pinned
to a commit, its ONNX model is checked against its Git LFS SHA-256, and the
Audiobox checkpoint hash is recorded. Whisper and ECAPA identities are saved
in each evaluation. The FFmpeg executable is local to the environment.

Install the isolated, pinned NeMo environment for Parakeet as well:

```bash
bash training/scripts/bootstrap_nemo_env.sh
```

It accepts the same `TORCH_INDEX_URL` override. The quality configuration pins
Parakeet's model revision and names its Python executable and cache. Evaluation
downloads that revision if absent, records the checkpoint hash and NeMo source
revision, and invokes one batched NeMo worker per panel. Per-file transcription
caches bind model identity to audio bytes. A failed worker fails evaluation;
there is no silent fallback to Whisper. Existing reports without Parakeet must
be regenerated in a new result directory before comparison with enabled runs.

The enhancement adapters additionally need ClearerVoice, Transformers,
DeepFilterNet, and the pinned RNNoise executable. With Cargo/Rust on `PATH`, run
`bash training/scripts/bootstrap_quality_processing.sh` after the quality
bootstrap. It accepts `TORCH_INDEX_URL` and records processor identities per
processed file. Run a small processing sweep to verify actual inference before
the full experiment; successful imports alone do not validate model execution.

Reconstruct the downloaded corpus without a denoiser:

```bash
python -m training.scripts.materialize_non_voa --workers 8
python -m training.quality select --manifest training/data/quality_v10_raw/all.parquet \
  --output training/quality_runs/v10/panels --per-source 100
```

The reconstruction preserves native decoded audio until the 24 kHz conversion,
uses the existing boundary-trim algorithm, and writes PCM24. It removes exact
audio and normalized-text duplicates and uses fixed global text groups for
96/2/2 train/dev/eval splits. These are newly reconstructed splits, not a claim
to reproduce the old host's manifest. Downloaded originals remain preserved.

Submit GPU work from the repository root (the default SLURM partition is used):

```bash
mkdir -p training/quality_runs
sbatch training/slurm/quality.sbatch python -m training.quality.search \
  --panel training/quality_runs/v10/panels/processing.jsonl \
  --output training/quality_runs/v10/search
```

The search compares an unprocessed control, individual processors, and cascades,
including the previous training cascade. It also tests a 50 percent dry mix.
`combinations.jsonl/.csv` records the measured metric vector for each profile.
`search_report.json` lists non-dominated profiles. It does not select one.
Extend the sweep around the best reviewed combinations (for example with
`--wet 0.25 0.75`) and repeat until the tested refinements show no meaningful
improvement. Review the per-source and worst-window results as well as the mean.
Record the tested search space and the stopping decision. Finishing one finite
sweep is not proof of a global optimum. Keep checkpoint holdout out of this
processing search.

After reviewing and freezing a profile:

```bash
export PROFILE="$PWD/training/quality_runs/v10/search/PROFILE_NAME/profile.yaml"
sbatch training/slurm/quality.sbatch bash training/scripts/prepare_quality_v10.sh
# After preparation succeeds:
sbatch training/slurm/quality.sbatch python -m training.scripts.calibrate_quality_v10
# Read calibration/recommended.json; set the measured values:
export BATCH_BINS=4000000 WORKERS=8
sbatch training/slurm/train_quality_v10.sbatch
```

The numerical batch example is a starting point, not a measured recommendation.
Calibration tries real JETS training batches and records runtime, power, and
available unified memory. It uses batch-bin throughput, with power as a tie
breaker, and a 100 W reference. It does not change the hardware power limit.
Unified CPU/GPU memory is one budget; do not add nominal host RAM and VRAM.

The default run starts from random weights and trains 100,000 steps. The
preparation uses new statistics/tokens and recomputes the established 50/50
raw/processed speaker embeddings. `INIT_CHECKPOINT` is optional and must be
compatible with the new tokens/architecture. Every 1,000-step epoch is a
checkpoint boundary. Milestones at 25K/50K/75K/100K are preserved, audited, and
evaluated after training. Set `UKTTS_QUALITY_BEST` to an explicitly reviewed
quality result directory to include the best baseline; previous checkpoints
are chained automatically. No quality threshold can promote or reject them.

Each submitted command has a durable command log, 5-second quality-job telemetry
(60 seconds for training), a current
status JSON, and a completion summary under the ignored run directory. Check
`squeue`, the command log, status, and progress at least every 30 minutes while
the pipeline is running. A failed command exits nonzero. Resume only the same
run with unchanged provenance; correct the failure before resubmission.

For a rebuilt eSpeak runtime, `verify_local_frontend.py` checks all 550 committed
regression cases before accepting a local data hash. V10 scripts use that
ignored, verified pin. The global historical pin is not overwritten. A new
hash never permits different phonemes through this verification step.
