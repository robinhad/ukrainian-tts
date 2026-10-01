# Non-VOA quality iteration

## Scope

Use the eight non-VOA sources. Compare processing on a fixed source-balanced
training panel, at most 100 items per source. Evaluate checkpoints on a separate
fixed held-out Ukrainian panel. Train for 100,000 steps after the processing
experiments and refinement review are complete.

## Implemented

- SigMOS overall, speech, noise, coloration, and discontinuity; Audiobox PQ.
- Whisper and Parakeet Ukrainian CER/WER against supplied transcripts; ECAPA similarity to
  original recordings.
- Clipping, duration change, and local high-frequency burst diagnostics.
- Whole-file scores, overlapping windows, and worst windows per metric.
- Per-file, per-source, aggregate, and paired comparison JSONL/CSV.
- Original, previous, and explicitly supplied best-checkpoint comparisons.
- Report-only thresholds. No HNR rejection and no automatic quality promotion.
- Pinned-source downloads, canonical PCM24 reconstruction, processing sweep,
  SLURM launchers, real-training calibration, milestone preservation, and
  operational telemetry every 5 seconds for quality jobs and 60 for training.

## Execution findings

All eight source downloads and reconstruction completed. No VOA source was
requested. The reconstructed corpus contains **80,751 recordings / 110.272 hours**.
Downloads total **14.845 GB**, and retained mono 24 kHz PCM24 files total
**28.586 GB** (decimal units). These counts use the new reconstruction and
deduplication policy; they do not reproduce the previous iteration's filters.

| Source | Recordings | Hours |
|---|---:|---:|
| Common Voice | 41,180 | 44.023 |
| Ukrainian dialects | 28,335 | 51.747 |
| OpenTTS Lada | 5,266 | 4.904 |
| FLEURS Ukrainian | 1,897 | 5.608 |
| OpenTTS Tetiana | 1,568 | 1.617 |
| OpenTTS Mykyta | 1,417 | 1.410 |
| Telegram voices | 618 | 0.614 |
| UA-SER | 470 | 0.348 |

There are 77,468 training, 1,595 development, and 1,688 evaluation recordings.
The fixed processing panel contains 800 recordings (100 per source); the fixed
checkpoint panel contains 88 (11 per source, limited by the smallest held-out
source). The entire evaluation split remains excluded from training. Panels
retain native, untrimmed originals as references and separately bind canonical
processing inputs. No train/dev overlap was found among eligible held-out items.
See [corpus CSV](quality_v10_corpus.csv) and [JSONL](quality_v10_corpus.jsonl).

The training environment built successfully. An ARM DeepFilterNet dependency
required a local Rust toolchain. The GPU matrix-operation smoke test passed
inside a SLURM allocation. The historical GPU monitor assumed numeric dedicated
VRAM and power-limit readings; it now accepts unavailable values on unified
memory hardware.

The local eSpeak build has a different language-data fingerprint. All 550
committed phoneme cases matched exactly. A separate verified runtime pin keeps
this local fingerprint out of Git and preserves the historical pin.

The complete training test suite passes with the verified local frontend pin.
The explicit frontend verification also passed all 550 reference cases.

All scoring models, including pinned Parakeet TDT 0.6B v3 in a separate NeMo
environment, completed a two-file Ukrainian smoke evaluation with no errors.
All eight processing paths also completed a two-file smoke sweep. These checks
verify model execution and report generation, not representative corpus quality.
The source-balanced experiment remains necessary. No processing winner or
audio-quality improvement is claimed yet.

Preparation profiling compared eight threads with eight worker processes on
512 source records: 10.98 seconds versus 1.04 seconds, with identical output
audio hashes and rejection results. Reconstruction now uses spawned worker
processes and one decoder thread per file. Existing verified caches are reused.
The training encoder was separately loaded from pinned local ECAPA files and
produced finite 192-dimensional embeddings. ClearVoice now explicitly fetches
the model revision recorded in its processing provenance.

The enhancement adapters verify CUDA tensor placement, including constants in
frozen TorchScript models. DeepFilterNet, Sidon, ClearVoice, and their combined
pipeline passed real-audio GPU checks. CPU-only RNNoise profiles are excluded
from the current sweep following the GPU-processing requirement.

SigMOS also runs its neural operations on CUDA. A runtime profile verified
CUDA convolutions, GRU, and dense layers; CPU work was limited to integer shape
bookkeeping. The largest absolute CPU/GPU score difference on two smoke files
was 0.0000012. The GPU sweep uses a frozen quality configuration with Whisper
and Parakeet content metrics. Processing selection and 100K training are pending.

The original baseline, base-conversion control, and initial DeepFilterNet and
Sidon trials completed on all 800 panel items with zero scoring errors. The following are equal-source mean whole-file
scores; CER/WER are mean per-file rates, not pooled corpus rates.

| Processing | SigMOS overall | Speech | Noise | Coloration | Discontinuity | Audiobox PQ | Whisper WER | Parakeet WER | ECAPA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native original | 2.997 | 3.457 | 3.765 | 3.528 | 3.960 | 6.786 | 16.142% | 13.370% | 1.000 |
| Base mono/24 kHz/trim | 2.923 | 3.478 | 3.722 | 3.455 | 4.085 | 6.705 | 16.108% | 13.438% | 0.954 |
| DeepFilterNet, 50% blend | 2.978 | 3.516 | 3.851 | 3.481 | 4.102 | 6.784 | 15.931% | 13.549% | 0.951 |
| DeepFilterNet, full strength | 3.108 | 3.624 | 4.121 | 3.526 | 4.121 | 6.953 | 20.234% | 15.297% | 0.905 |
| Sidon, 50% blend | 3.043 | 3.635 | 4.031 | 3.438 | 4.164 | 7.026 | 17.590% | 14.075% | 0.851 |
| Sidon, full strength | 3.152 | 3.738 | 4.260 | 3.461 | 4.183 | 7.443 | 16.700% | 14.309% | 0.815 |

Base conversion has mixed effects and is not an enhancement result. Its mean
duration ratio to the untrimmed native original is 0.824. Subsequent model
effects must also be compared against this control. Full per-source aggregates
and pooled error rates are in [processing CSV](quality_v10_processing.csv) and
[JSONL](quality_v10_processing.jsonl). Enhancement trials and refinement remain
in progress; no winner has been selected.

Against the base-conversion control, the 50% blend improved mean SigMOS overall
by 0.0549 (95% interval 0.0457–0.0636) and Audiobox PQ by 0.0797
(0.0736–0.0855). Neither ASR's WER difference excluded zero. Full strength
improved SigMOS overall by 0.1852 but worsened Whisper WER by 4.126 percentage
points (2.850–5.509) and Parakeet WER by 1.858 points (0.787–2.946). The blend is
promising for refinement; full strength shows a content-preservation tradeoff.
Speaker similarity also declined, so quality gains alone do not establish a winner.

Sidon improves mean quality scores more, but reduces speaker similarity markedly.
Mean high-frequency burst counts rise from 1.940 in the control to 2.130 at 50%
and 2.384 at full strength. The full-strength mean Whisper CER is 13.320%; one
very short UA-SER item has CER 55.75 from a long recognition error and remains in
the report. Pooled CER/WER and per-file scores are provided alongside these means
to expose the effect of short-reference outliers. No quality-based rejection is applied.

Intervals use 2,000 paired bootstrap resamples within each of the eight sources,
seed 777, and equal-source means. They describe this fixed panel and are
exploratory, without adjustment for multiple trials. They are not calibrated
selection thresholds. See [paired CSV](quality_v10_paired_processing.csv) and
[JSONL](quality_v10_paired_processing.jsonl).

Sequential short-file scoring drew approximately 37–39 W with 42–56% sampled
GPU utilization during the observed control phase. GPU placement alone does
not saturate these small inference workloads. CPU postprocessing now overlaps
model inference; one-worker and four-worker identity outputs were byte-identical
in tests, and worker failures correctly propagated. A two-trial concurrent
sweep is being measured. This is an execution experiment, not a demonstrated
throughput gain. The updated suite passed 186 tests. All 80,751 records also
passed the frontend and structural dataset preflight.

The initial full-strength DeepFilterNet trial encountered an output below the
EBU loudness measurement gate. On that FLEURS item, input RMS was 0.003880 and
output RMS was 0.00000658 (about 55 dB lower). Undefined integrated loudness now
leaves the output unamplified, subject only to the peak cap, and records an
explicit status. The item stays in the panel for content and quality scoring.
This is a single-file failure mode, not an aggregate verdict on the model.

A refinement now peak-normalizes DeepFilterNet input to -3 dBFS, reverses that
input gain before blending, and retains the same final loudness target. On the
same quiet FLEURS item, this applied 26.19 dB of input gain and produced output
RMS 0.003874, versus input RMS 0.003880. The enhanced signal needed only 0.19 dB
of final loudness correction. The normalization/blending test passed, but the
800-item content and quality evaluation is still running; level preservation
alone does not establish content or perceptual quality improvement.

The separate 88-item held-out original baseline also completed with all requested
metrics and zero errors. Its recordings remain excluded from processing tuning
and model training; checkpoint outputs will be compared with these references.

A fixed 160-item refinement panel contains 20 training recordings per source,
selected independently of measured scores with seed 778. Matching reference
reports are derived from the complete 800-item measurements after audio-hash
verification. Promising refinements must be confirmed on the full processing
panel. The updated test suite passes 190 tests. Preparation now enables the
recipe's existing feature cache so training can reuse pitch and energy instead
of recomputing them on CPU every iteration; full-corpus preparation is pending.

## Remaining work

Finish the processing sweep and refinements, record measured comparisons and the stopping decision,
then process the full corpus. Prepare embeddings/statistics, calibrate training,
complete the 100K run, and compare its checkpoints. See
[run instructions](../quality/README.md).
