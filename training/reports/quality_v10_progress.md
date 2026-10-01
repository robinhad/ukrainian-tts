# Non-VOA quality iteration

## Scope

Use the eight non-VOA sources. Compare processing on a fixed source-balanced
training panel, at most 100 items per source. Evaluate checkpoints on a separate
fixed held-out Ukrainian panel. Train for 100,000 steps after the processing
experiments and refinement review are complete.

## Implemented

- SigMOS overall, speech, noise, coloration, and discontinuity; Audiobox PQ.
- Whisper Ukrainian CER/WER against supplied transcripts; ECAPA similarity to
  original recordings.
- Clipping, duration change, and local high-frequency burst diagnostics.
- Whole-file scores, overlapping windows, and worst windows per metric.
- Per-file, per-source, aggregate, and paired comparison JSONL/CSV.
- Original, previous, and explicitly supplied best-checkpoint comparisons.
- Report-only thresholds. No HNR rejection and no automatic quality promotion.
- Pinned-source downloads, canonical PCM24 reconstruction, processing sweep,
  SLURM launchers, real-training calibration, milestone preservation, and
  60-second operational telemetry.

## Execution findings

All eight source downloads completed at their recorded revisions. No VOA source
was requested. The corpus is being reconstructed; its measured size and counts
must be reported after completion rather than copied from the previous run.

The training environment built successfully. An ARM DeepFilterNet dependency
required a local Rust toolchain. The GPU matrix-operation smoke test passed
inside a SLURM allocation. The historical GPU monitor assumed numeric dedicated
VRAM and power-limit readings; it now accepts unavailable values on unified
memory hardware.

The local eSpeak build has a different language-data fingerprint. All 550
committed phoneme cases matched exactly. A separate verified runtime pin keeps
this local fingerprint out of Git and preserves the historical pin.

The training test suite currently reports 166 passed and 8 skipped. The skipped
tests use the historical frontend data fingerprint; the explicit local frontend
verification separately passed all 550 cases. Real scoring-model integration
is under test. No processing winner or audio-quality improvement is claimed yet.

## Remaining work

Finish corpus reconstruction and real scoring validation. Run the processing
sweep and refinements, record measured comparisons and the stopping decision,
then process the full corpus. Prepare embeddings/statistics, calibrate training,
complete the 100K run, and compare its checkpoints. See
[run instructions](../quality/README.md).
