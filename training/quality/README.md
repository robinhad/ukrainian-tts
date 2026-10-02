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
  --output training/quality_runs/v10/panels --per-source 100 \
  --downloads training/data/non_voa_downloads/downloads.json
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

For this downloaded corpus, `--downloads` recovers the selected original
recordings at their native sample rate, channel count, and untrimmed boundaries
as float WAV references. The panel separately binds canonical 24 kHz processing
inputs by hash. Thus original/checkpoint comparisons include effects of boundary
trimming and resampling, and the identity processor measures that base conversion.
For an external manifest without the download registry, omit `--downloads`;
its existing audio files become the references, as recorded in `selection.json`.

## Metrics

The same evaluator handles processing outputs and synthesized checkpoint WAVs:

- [Microsoft SigMOS](https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos):
  overall, speech, noise, coloration, discontinuity, plus loudness and reverb.
- [Meta Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics):
  production quality (PQ).
- [Whisper](https://github.com/openai/whisper): Ukrainian transcription with
  fixed language and decoding settings. CER/WER compare against the supplied
  source transcript. Normalization uses NFC, lowercase, standardized apostrophes,
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
Segment CER/WER is deliberately unavailable without time-aligned reference text;
the whole transcript must not be compared to a short crop. Short windows and
all uncalibrated thresholds need human review.
CER/WER also depend on the accuracy of the supplied source transcripts.

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
errors stop scoring. Threshold flags do not fail the checkpoint. Silent but
nonempty WAVs and unusual durations are warnings and still reach the quality
evaluator. The synthesis report's absolute duration warning bounds default to
0.1–30 seconds and can be changed with `--duration-warning-min` and
`--duration-warning-max`. Relative duration-change flags use the quality YAML.
Empty files, nonfinite samples, wrong audio format, and missing IDs remain
structural errors.

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

SigMOS uses the ONNX Runtime GPU distribution. The bootstrap removes the CPU
distribution because both share the same Python namespace. Runtime profiling
checks that neural operations run on CUDA; integer tensor-shape bookkeeping may
run on CPU. Its placement counts are recorded in model provenance. Whisper,
Audiobox, ECAPA, and Parakeet also verify GPU tensor placement. CUDA installation
or model-placement failures stop the job rather than producing CPU model scores.

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

The enhancement adapters additionally need ClearerVoice, Transformers, and
DeepFilterNet. With Cargo/Rust on `PATH`, run
`bash training/scripts/bootstrap_quality_processing.sh` after the quality
bootstrap. It accepts `TORCH_INDEX_URL` and records processor identities per
processed file. Run a small processing sweep to verify actual inference before
the full experiment; successful imports alone do not validate model execution.

Reconstruct the downloaded corpus without a denoiser:

```bash
python3 -m venv training/.venv-data
training/.venv-data/bin/pip install -r training/requirements-data.in
DATA_PYTHON="$PWD/training/.venv-data/bin/python"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$DATA_PYTHON" -m training.scripts.materialize_non_voa --workers 8
"$DATA_PYTHON" -m training.quality select --manifest training/data/quality_v10_raw/all.parquet \
  --output training/quality_runs/v10/panels --per-source 100 \
  --downloads training/data/non_voa_downloads/downloads.json
```

The reconstruction preserves native decoded audio until the 24 kHz conversion,
uses the existing boundary-trim algorithm, and writes PCM24. It removes exact
audio and normalized-text duplicates and uses fixed global text groups for
96/2/2 train/dev/eval splits. These are newly reconstructed splits, not a claim
to reproduce the old host's manifest. Downloaded originals remain preserved.
Conversion uses separate worker processes, with one decoder thread per file.
Verified per-file caches survive interruption; rerunning the command reconstructs
the complete manifest with the same deduplication and split policy.
The recorded corpus uses PyAV 19.0.0 and SoundFile 0.14.0 in the isolated CPU
decoder environment; the scoring environment has separate dependency pins.
See [decoder environment](../reports/quality_v10_decoder_environment.json) for
library versions and implementation hashes. Preserve the frozen audio hashes
for comparisons. If changing decoder versions, reconstruct into a fresh output
directory instead of mixing new outputs with existing per-file caches.

Submit GPU work from the repository root (the default SLURM partition is used):

```bash
mkdir -p training/quality_runs
sbatch training/slurm/quality.sbatch python -m training.quality.search \
  --panel training/quality_runs/v10/panels/processing.jsonl \
  --output training/quality_runs/v10/search
```

The search compares a base-conversion control, individual GPU processors, and
GPU model cascades. It also tests a 50 percent dry mix. Loaded tensor devices,
including frozen TorchScript weight constants, are checked and recorded.
GPU-capable neural stages must stay on GPU. The bundled RNNoise executable
runs on CPU and is permitted, as are CPU decoding, resampling, loudness
matching, and non-model signal filters.
CPU loudness matching and WAV writes overlap GPU inference through a bounded
worker queue (`process --cpu-workers`, default 4). Model calls stay sequential
to preserve recurrent state, and worker failures propagate to the pipeline.
`--model-workers` can run independent model processes on disjoint input shards.
Each process owns its model state and exits after a bounded chunk (default
64 files, `--chunk-size`) to release retained native allocations. Unused host
allocations are also released every eight completed files. Chunks have separate
progress and result files; the parent audits complete coverage before merging
the report. `--minimum-available-gib` stops processing before the configured
available-memory reserve is exhausted. Benchmark worker
counts and verify output identity before increasing concurrency. Preparation
accepts `PROCESS_MODEL_WORKERS`, `PROCESS_CPU_WORKERS`, and
`PROCESS_CPU_THREADS` (defaults 1, 2, and 1), plus `PROCESS_CHUNK_SIZE`
and `PROCESS_MIN_AVAILABLE_GIB` (defaults 64 and 4).
If an input or enhanced signal falls below the EBU loudness measurement gate,
normalization applies only the peak cap and records `below_measurement_gate`.
The file remains in the evaluation, exposing suppression in the content and
quality metrics; an undefined loudness measurement never causes an unlimited gain.
An optional profile field, `input_peak_dbfs: -3`, peak-normalizes the model input.
The processor reverses that input gain before wet/dry blending and final loudness
matching. This is an experimental processing setting and must be compared on the
fixed panel before use; it is disabled by default.
Profiles can also set `output_lufs: -23` to test a common final loudness target
instead of matching each input recording. This remains linear gain with a peak
cap; the report records when the peak cap prevents reaching the target. The
option also supports `backend: identity` to isolate level normalization from
model effects. Without this field, final loudness matches the original input.

For faster local refinements, freeze a smaller source-balanced subset of the
processing panel and derive its reference scores from completed measurements:

```bash
python -m training.quality.refine_panel \
  --parent training/quality_runs/v10/panels/processing.jsonl \
  --output training/quality_runs/v10/refinement --per-source 20 \
  --reports training/quality_runs/v10/search/original \
    training/quality_runs/v10/search/identity_wet1/quality
```

This selects training items only, verifies measured audio hashes, and recomputes
aggregates from the selected per-file and segment scores. Derived reports record
their parent provenance and do not pretend the models were rerun. Evaluate new
profiles with `refinement/processing.jsonl`, compare with `refinement/references`,
then confirm promising settings on the full 800-item processing panel before
selection. The held-out checkpoint panel remains unchanged.
To evaluate the complete historical v8/v9 training recipe, including its DSP,
RNNoise85, and production fallbacks for outputs below the loudness gate:

```bash
sbatch training/slurm/quality.sbatch bash training/scripts/evaluate_legacy_quality.sh
```

The default uses the fixed 160-item refinement panel and matching references.
ClearVoice, Sidon, and DeepFilterNet run on GPU; RNNoise runs on CPU. Two
processing workers share the allocation. The launcher checks frozen audio hashes
and the RNNoise archive checksum, records code/binary hashes and model devices,
and preserves the previous recipe's fallback behavior. This reproduces the
historical processing recipe in the current environment, without claiming
byte identity with files produced by an earlier environment.

For the complete 800-item comparison, set all panel and reference paths together:

```bash
PANEL=training/quality_runs/v10/panels/processing.jsonl \
OUTPUT_DIR=training/quality_runs/v10/search/legacy_v8_exact \
ORIGINAL_REPORT=training/quality_runs/v10/search/original \
CONTROL_REPORT=training/quality_runs/v10/search/identity_wet1/quality \
sbatch training/slurm/quality.sbatch bash training/scripts/evaluate_legacy_quality.sh
```

`PROCESS_WORKERS` sets concurrency (default 2); `QUALITY_CONFIG` selects the
frozen metric configuration. Historical reports use the same per-file, segment,
aggregate, and comparison formats as the other profiles. The generic three-model
cascade does not include all these historical stages and is a separate trial.

The sweep also accepts `--profile-workers 2` to run independent processing and
scoring trials concurrently within one GPU allocation. Each trial writes to its
own directory and `command.log`. Compare throughput, power, and free unified
memory before increasing concurrency; the default remains one trial.
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
export PROFILE="$PWD/training/conf/quality_v10_processing.yaml"
export PROCESS_MODEL_WORKERS=8 PROCESS_CPU_WORKERS=2 PROCESS_CPU_THREADS=1
export PROCESS_CHUNK_SIZE=64
sbatch training/slurm/quality.sbatch bash training/scripts/prepare_quality_v10.sh
# After preparation succeeds:
sbatch training/slurm/quality.sbatch python -m training.scripts.calibrate_quality_v10
# After calibration succeeds, consume its measured recommendation:
CALIBRATION_FILE="$PWD/training/quality_runs/v10/calibration/recommended.json" \
sbatch training/slurm/train_quality_v10.sbatch
```

If audio processing completed but a later preparation stage failed, resubmit
the preparation command with `SKIP_AUDIO_PROCESSING=true`. This skips loading
the enhancement models again; the manifest builder still verifies every input
and output hash, PCM24 format, and frozen processing profile before proceeding.
It preserves Parquet array fields in JSONL and publishes the manifest atomically.
Reattach pending SLURM dependencies to the replacement preparation job.

`CALIBRATION_FILE` overrides `BATCH_BINS`, `WORKERS`, and `USE_TF32` with
validated values from the completed calibration. Without it, those environment
variables or the script defaults apply. Calibration trials ignore an inherited
recommendation so each declared setting is actually measured. SLURM
`afterok` dependencies can chain these jobs after the profile review.
Calibration tries real JETS training batches, worker counts, and TF32 settings,
and records runtime, power, and available unified memory. Preparation caches
pitch and energy so training can avoid repeated CPU extraction. A trial qualifies
only if both GAN optimizers completed the requested updates and model weights
are finite. Training time comes from the checkpoint reporter, excluding setup and validation from the
throughput comparison. It uses batch-bin throughput, with power as a tie
breaker, and a 100 W reference. It does not change the hardware power limit.
Unified CPU/GPU memory is one budget; do not add nominal host RAM and VRAM.
The training runner applies the same `OMP_NUM_THREADS` limit to OpenMP, MKL,
and OpenBLAS in calibration and full training; the default is four.

Calibration defaults to a 24 GiB available-memory reserve, configurable with
`--minimum-available-gib`. Its supervisor checks memory every second independently
of GPU telemetry and terminates the trial's process group if the reserve is
breached, escalating after two seconds to clean up descendants. A stopped trial
cannot become the recommendation. Full training uses the same guard, configured
by `TRAIN_MIN_AVAILABLE_GIB` (default 24). The actual training log also counts as
activity, avoiding false stall alerts when the launcher itself is quiet.

To refine a completed calibration without repeating successful trials, use a
new `--output` directory and pass `--prior-trials` with an earlier `trials.jsonl`.
Only trials with verified updates in both optimizers and sufficient memory
qualify. `combined_trials.jsonl/.csv` retains the old and new measurements used
for the recommendation. Point `CALIBRATION_FILE` at that new recommendation.

Calibration now defaults to 1,000 updates per trial and releases unused CUDA
cache every 10 updates. Use `--steps` and `--cuda-cache-interval` to override
those measurements explicitly. Prior trials must match both settings; a short
trial or an earlier cache policy cannot silently become the new recommendation.

Short calibration trials do not establish long-run memory stability. The first
full run exhausted its reserve after the last logged 330 updates despite passing
100-update calibration. `UKTTS_CUDA_CACHE_INTERVAL` releases unused PyTorch
allocator blocks between complete GAN updates (default 10 in the V10 launcher;
zero disables it). Each
release logs live, reserved, and peak GPU memory before and after; it does not
discard live tensors. Calibration records this setting and carries it into the
recommendation so full training uses the measured configuration. Validate it
over at least a full training epoch before treating it as a sustained-memory
result. The independent available-memory guard remains enabled. The 50-update
policy subsequently crossed the reserve during epoch six. The active recovery
uses `UKTTS_CUDA_CACHE_INTERVAL=10` with `CALIBRATION_FILE` unset, preserving
four million batch bins and eight workers. It completed the recovery epoch
from 5K to a verified 6K checkpoint; monitoring continues because a successful
epoch does not guarantee later memory stability.

On resume, the bootstrap also restores CPU scalar step counters for ordinary
Adam/AdamW. ESPnet loads checkpoint tensors onto CUDA, but PyTorch deliberately
keeps those counters on CPU when neither `capturable` nor `fused` is enabled.
Keeping them on CUDA adds synchronization during each optimizer update. The
patch changes counter placement only; moment tensors remain on the parameter
device, and capturable/fused modes retain their CUDA counters. A GPU integration
test verifies resumed updates against uninterrupted updates. Run it inside a
SLURM allocation with `UKTTS_TEST_GPU=1 python -m pytest -q
training/tests/test_optimizer_state.py`. Normal CPU test runs skip that explicit
GPU integration case.

The environment bootstrap disables cuDNN benchmarking in the parallel speaker
extractor: variable audio lengths otherwise trigger repeated kernel searches.
The batch size remains eight. An isolated 160-file comparison verified the speed
change and near-identical vectors; see
[embedding measurements](../reports/quality_v10_embedding_benchmark.json).

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

To summarize the original-audio score distributions without rerunning models:

```bash
python training/scripts/report_source_quality_distribution.py \
  --input training/quality_runs/v10/search/original/per_file.jsonl \
  --output-prefix training/reports/quality_v10_source_mos_distribution
```

This writes whole-file percentiles by source and across the sample to CSV/JSONL,
histogram bins to separate CSV/JSONL files, and cumulative distribution plots to
PNG/PDF. The V10 baseline contains 800 recordings, exactly 100 per source. Its
pooled distribution weights sources equally, rather than by their full-corpus
proportions. SigMOS scores use 1–5 and Audiobox PQ uses 1–10; higher is better.
These are model estimates and descriptive summaries, not calibrated rejection
thresholds.

To plot median overall SigMOS across the 16 full-panel preprocessing variants:

```bash
python training/scripts/plot_processing_sigmos.py \
  --input training/reports/quality_v10_processing.csv \
  --output-prefix training/reports/quality_v10_processing_median_sigmos
```

This writes a ranked dot plot to PNG/PDF and its values to CSV. Each variant
covers the same 800 recordings. The original baseline and selected training
recipe are marked; the smaller refinement panels are excluded. The chart uses
a zoomed score axis and reports medians, rather than means or paired changes.

To compare the original and highest-median method as matched histograms:

```bash
python training/scripts/plot_original_vs_cascade.py \
  --original training/quality_runs/v10/search/original/per_file.jsonl \
  --processed training/quality_runs/v10/search/clearervoice_sidon_deepfilternet3_wet1/quality/per_file.jsonl \
  --output-prefix training/reports/quality_v10_original_vs_cascade
```

This checks identical 800-item coverage and writes PNG/PDF with shared axes and
0.2-point bins, plus summary and histogram CSV/JSONL. Threshold counts use exact
per-file scores, independently of histogram bins. Only aggregate values are
exported; private paths and transcripts are omitted.

For a hypothetical per-item selection that uses the cascade only when its
overall SigMOS exceeds the original, add `--keep-original-if-worse` and use
`--output-prefix training/reports/quality_v10_best_of_original_and_cascade`.
This plots `max(original, processed)` after matching recording identifiers;
ties retain original audio. Additional `_selection.csv`/`_selection.jsonl`
files contain opaque sample identifiers, both scores, the chosen score, and
the chosen variant. This report does not modify audio, manifests, or training.
The selection is evaluated using the same score that drives the choice, so its
non-decreasing SigMOS is guaranteed by construction, not independent validation.

To visualize individual original-to-cascade changes:

```bash
python training/scripts/plot_individual_sigmos_changes.py \
  --original training/quality_runs/v10/search/original/per_file.jsonl \
  --processed training/quality_runs/v10/search/clearervoice_sidon_deepfilternet3_wet1/quality/per_file.jsonl \
  --output-prefix training/reports/quality_v10_individual_sigmos_changes
```

The PNG/PDF pairs a before/after scatter plot with individual changes by source.
CSV/JSONL exports contain opaque sample identifiers, sources, scores, and signed
changes; separate summaries report per-source and pooled statistics. Matching
uses original utterance/source keys before anonymizing the exported identifiers.
Positive and negative counts use any nonzero difference, not a calibrated
meaningful-change threshold. Median individual change differs from the
difference between the two groups' medians.

## V11: user-selected corpus and 50K training

V11 applies the full ClearVoice → Sidon → DeepFilterNet3 cascade, compares each
output's overall SigMOS with its native original, and chooses the processed
version only when its score is strictly higher. Ties keep the original. The
chosen score must be at least 3.5. This explicit corpus-selection policy does
not change the report-only checkpoint thresholds or introduce HNR rejection.

```bash
sbatch training/slurm/quality_v11.sbatch
```

The pipeline streams the existing pinned non-VOA downloads, processes bounded
chunks on CUDA, scores both versions with the official CUDA SigMOS model, and
stores only retained audio plus all decisions under `training/data/quality_v11`.
It checks input/output hashes and code/profile provenance on resume. Interrupted
chunks can be regenerated; completed, hash-verified decisions are reused.
`PROCESS_MODEL_WORKERS` and `PROCESS_CHUNK_SIZE` control throughput and memory.
Temporary chunk files are removed after their decisions are saved. Existing
V10 artifacts are preserved.

The V11 processing profile pins a per-recording random seed for ClearVoice's
Kaldi feature dither and disables the profiled TorchScript optimization that
changed Sidon numerics after its first call. These settings make the new
outputs repeatable; the earlier unseeded sweep remains historical evidence,
not a cache of scores to reuse for newly generated audio.

Original winners retain native timing and are converted to mono 24 kHz PCM24
for the model. Their rendered audio is scored again and any drop below 3.5 is
reported separately; selection uses the measured native-original/cascade
comparison. The pipeline exports actual retained counts, hours, source and
split summaries, and Ukrainian letter coverage.

Train/dev assignments remain disjoint and are filtered. The complete original
evaluation split remains fixed, including the existing 88-recording quality
panel, so evaluation is not restricted to high-scoring recordings. The
`records.jsonl` file describes all passing recordings; `training_records.jsonl`
contains filtered train/dev plus that fixed evaluation population.

Preparation rebuilds manifests, ECAPA embeddings, pitch/energy and feature
statistics. Training starts from scratch in `exp_quality_v11/tts_jets_quality_v11_50k`
for 50,000 updates with 4M batch bins, 8 loader workers, FP32, and unused-CUDA-cache
release every 10 GAN updates. A separate one-second memory guard enforces a
24 GiB available-memory reserve. Durable telemetry is sampled every minute.
The 25K and 50K weights are preserved, the final optimizer counters/weights are
audited, and both checkpoints receive the existing held-out audio evaluation
(SigMOS, Audiobox PQ, Whisper/Parakeet, ECAPA, signal and worst-segment reports).

Monitor `data/quality_v11/progress.json`, `quality_runs/v11/pipeline/status.json`,
and `quality_runs/v11/training/status.json`. Selection progress includes an ETA
in Europe/Kyiv time once completed chunks provide a throughput estimate. Check
the SLURM job and logs at least every 30 minutes throughout the run.

## W&B training metrics

The V11 launcher uploads numeric training and validation TensorBoard scalars to
the `ukrainian-tts` W&B project every 30 seconds, using optimizer iterations as
`training_step`. The run is named `quality-v11-50k`. Set `WANDB_PROJECT`,
`WANDB_NAME`, or `WANDB_ENTITY` before submission to override these defaults;
credentials come from the existing W&B login or `WANDB_API_KEY` environment.
`WANDB_METRICS=true` enables the same uploader for other training iterations.

The uploader reads only scalar events. It does not upload model checkpoints,
audio, source code, console logs, or training configuration files. Native ESPnet
W&B integration and model logging are explicitly disabled. Automatic W&B code,
machine metadata, and system-stat collection are disabled. Checkpoints and
TensorBoard files remain local. The ignored experiment directory stores the W&B
run ID, URL, and export cursors so a resumed training run continues the same run.
Authentication is checked before training starts; the uploader flushes its final
metrics when training exits.
