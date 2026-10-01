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
