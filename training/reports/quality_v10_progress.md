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
The reconstruction and native reference decoder used the recorded isolated CPU
environment: PyAV 19.0.0, SoundFile 0.14.0, and libsndfile 1.2.2. Its dependency
pins are separate from the scoring environment. See
[decoder provenance](quality_v10_decoder_environment.json); cross-version audio
byte identity is not assumed.

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
The smoke checks establish execution, not audio-quality improvement. The
source-balanced findings below support the subsequent processing review.

Preparation profiling compared eight threads with eight worker processes on
512 source records: 10.98 seconds versus 1.04 seconds, with identical output
audio hashes and rejection results. Reconstruction now uses spawned worker
processes and one decoder thread per file. Existing verified caches are reused.
The training encoder was separately loaded from pinned local ECAPA files and
produced finite 192-dimensional embeddings. ClearVoice now explicitly fetches
the model revision recorded in its processing provenance.

The enhancement adapters verify CUDA tensor placement, including constants in
frozen TorchScript models. DeepFilterNet, Sidon, ClearVoice, and their combined
pipeline passed real-audio GPU checks. RNNoise was initially excluded because
the GPU preference was interpreted too strictly. CPU stages are permitted;
RNNoise now runs on CPU while GPU-capable neural stages remain on GPU.

SigMOS also runs its neural operations on CUDA. A runtime profile verified
CUDA convolutions, GRU, and dense layers; CPU work was limited to integer shape
bookkeeping. The largest absolute CPU/GPU score difference on two smoke files
was 0.0000012. The GPU sweep uses a frozen quality configuration with Whisper
and Parakeet content metrics. Processing selection is documented below; 100K training is pending.

The original baseline, base-conversion control, initial DeepFilterNet, Sidon, and ClearVoice
trials, and the first input-normalization refinement completed on all 800 panel
items with zero scoring errors. The following are equal-source mean whole-file
scores; CER/WER are mean per-file rates, not pooled corpus rates.

| Processing | SigMOS overall | Speech | Noise | Coloration | Discontinuity | Audiobox PQ | Whisper WER | Parakeet WER | ECAPA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native original | 2.997 | 3.457 | 3.765 | 3.528 | 3.960 | 6.786 | 16.142% | 13.370% | 1.000 |
| Base mono/24 kHz/trim | 2.923 | 3.478 | 3.722 | 3.455 | 4.085 | 6.705 | 16.108% | 13.438% | 0.954 |
| DeepFilterNet, 50% blend | 2.978 | 3.516 | 3.851 | 3.481 | 4.102 | 6.784 | 15.931% | 13.549% | 0.951 |
| DeepFilterNet, full strength | 3.108 | 3.624 | 4.121 | 3.526 | 4.121 | 6.953 | 20.234% | 15.297% | 0.905 |
| DeepFilterNet, -3 dBFS input, full strength | 3.124 | 3.633 | 4.127 | 3.534 | 4.130 | 6.947 | 18.429% | 13.671% | 0.921 |
| Sidon, 50% blend | 3.043 | 3.635 | 4.031 | 3.438 | 4.164 | 7.026 | 17.590% | 14.075% | 0.851 |
| Sidon, full strength | 3.152 | 3.738 | 4.260 | 3.461 | 4.183 | 7.443 | 16.700% | 14.309% | 0.815 |
| ClearVoice, 50% blend | 2.956 | 3.501 | 3.813 | 3.473 | 4.103 | 6.755 | 15.864% | 13.383% | 0.952 |
| ClearVoice, full strength | 3.024 | 3.553 | 3.982 | 3.506 | 4.130 | 6.851 | 16.229% | 13.189% | 0.939 |
| ClearVoice → Sidon, 50% blend | 3.045 | 3.636 | 4.036 | 3.437 | 4.164 | 7.025 | 17.664% | 14.010% | 0.852 |
| ClearVoice → Sidon, full strength | 3.151 | 3.733 | 4.266 | 3.462 | 4.173 | 7.446 | 16.976% | 14.338% | 0.815 |
| ClearVoice → Sidon → DeepFilterNet, 50% blend | 3.052 | 3.642 | 4.037 | 3.436 | 4.147 | 7.044 | 17.553% | 14.218% | 0.853 |
| ClearVoice → Sidon → DeepFilterNet, full strength | 3.196 | 3.765 | 4.304 | 3.470 | 4.142 | 7.528 | 17.682% | 13.788% | 0.811 |
| DeepFilterNet, -3 dBFS input, 75% blend | 3.024 | 3.551 | 3.946 | 3.498 | 4.121 | 6.850 | 15.983% | 13.409% | 0.945 |
| ClearVoice, 75% blend | 2.982 | 3.521 | 3.878 | 3.486 | 4.121 | 6.790 | 15.663% | 13.323% | 0.949 |
| Historical v8/v9 recipe, complete | 3.164 | 3.700 | 4.265 | 3.457 | 4.056 | 7.495 | 17.796% | 13.732% | 0.799 |

Base conversion has mixed effects and is not an enhancement result. Its mean
duration ratio to the untrimmed native original is 0.824. Subsequent model
effects must also be compared against this control. Full per-source aggregates
and pooled error rates are in [processing CSV](quality_v10_processing.csv) and
[JSONL](quality_v10_processing.jsonl). The initial sweep and two full-panel
confirmations and the full historical baseline are complete. The reviewed
processing choice is documented below.

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
very short UA-SER item has CER ratio 55.75 (5,575%) from a long recognition error and remains in
the report. Pooled CER/WER and per-file scores are provided alongside these means
to expose the effect of short-reference outliers. No quality-based rejection is applied.

ClearVoice full strength improves mean SigMOS overall by 0.102 and Audiobox PQ
by 0.147 relative to the control, with both mean ASR WERs close to the control.
Its speaker similarity is 0.939 and mean burst count is 1.920, versus control
values 0.954 and 1.940. It is a promising tradeoff for further review, not an
automatically promoted winner. The refinement plan also tests a 75% ClearVoice
blend and full strength with a common -23 LUFS final target. Adding ClearVoice
before Sidon yields nearly the same quality and speaker scores as Sidon alone;
the added model does not resolve that tradeoff.

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
sweep completed. That experiment did not isolate its throughput gain. The updated suite passed 186 tests. All 80,751 records also
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
of final loudness correction. The full-panel scores above retain the quality
gain and improve content preservation compared with unnormalized full strength.
However, Whisper WER still exceeds the base control by 2.321 percentage points
(95% interval 1.425–3.287), and speaker similarity remains lower. Input
normalization alone therefore does not resolve the full-strength tradeoff.

The separate 88-item held-out original baseline also completed with all requested
metrics and zero errors. Its recordings remain excluded from processing tuning
and model training; checkpoint outputs will be compared with these references.

A fixed 160-item refinement panel contains 20 training recordings per source,
selected independently of measured scores with seed 778. Matching reference
reports are derived from the complete 800-item measurements after audio-hash
verification. Promising refinements must be confirmed on the full processing
panel. The updated test suite passes 192 tests. Preparation now enables the
recipe's existing feature cache so training can reuse pitch and energy instead
of recomputing them on CPU every iteration; full-corpus preparation is underway.
The declared local refinement grid tests input peaks of -6, -3, and 0 dBFS;
50%, 75%, and full wet strength; and a common final target of -23 LUFS with
and without denoising. Only specified combinations are evaluated, not the full
Cartesian product. The final-loudness option passed a real-meter test for target
level, duration preservation, and peak safety. It remains an experiment until
the panel scores are reviewed.

## Refinement screening (160 recordings)

These means use the separate fixed 20-per-source screening subset. Compare rows
within this table; its composition differs from the full 800-item panel above.
Reference rows reuse verified measurements of the same selected recordings.

| Processing | SigMOS overall | Audiobox PQ | Whisper WER | Parakeet WER | ECAPA |
|---|---:|---:|---:|---:|---:|
| Base control | 2.924 | 6.724 | 14.46% | 13.91% | 0.949 |
| Base control, -23 LUFS | 2.931 | 6.716 | 14.34% | 14.31% | 0.949 |
| Historical v8/v9 recipe, including RNNoise85 and fallbacks | 3.185 | 7.485 | 17.48% | 14.69% | 0.798 |
| DeepFilterNet, -6 dBFS input, full | 3.113 | 6.989 | 15.69% | 14.82% | 0.922 |
| DeepFilterNet, -3 dBFS input, full (reference) | 3.122 | 6.982 | 16.50% | 14.83% | 0.922 |
| DeepFilterNet, 0 dBFS input, full | 3.128 | 6.983 | 17.56% | 14.95% | 0.920 |
| DeepFilterNet, -3 dBFS input, 75% | 3.034 | 6.883 | 14.30% | 13.57% | 0.943 |
| DeepFilterNet, -6 dBFS input, 75% | 3.035 | 6.882 | 14.66% | 13.74% | 0.943 |
| DeepFilterNet, -3 dBFS input, 50% | 2.997 | 6.818 | 14.43% | 13.39% | 0.948 |
| DeepFilterNet, -3 dBFS input, 50%, -23 LUFS | 3.005 | 6.815 | 14.39% | 13.95% | 0.948 |
| ClearVoice, 75% | 2.993 | 6.806 | 14.17% | 13.26% | 0.945 |
| ClearVoice, full (reference) | 3.031 | 6.875 | 15.44% | 13.72% | 0.932 |
| ClearVoice, full, -23 LUFS | 3.038 | 6.877 | 15.47% | 14.00% | 0.932 |

Changing the full-strength input peak does not remove the content-preservation
tradeoff. Gentler blends are more promising: the normalized 75% candidate is
now evaluated on all 800 items, alongside the 75% ClearVoice blend. The
full-panel values are in the earlier table. Level normalization alone shows
no clear benefit: quality scores are mixed and Parakeet WER rises slightly. Mean burst flags increase from 2.075 to 2.600;
uniform gain can cross the detector's fixed energy floor, so this does not
establish that normalization created new relative high-frequency bursts.
The fixed-level 50% DeepFilterNet blend likewise does not show a clear benefit
over its matched-level counterpart. The full-strength ClearVoice final-level
trial likewise yields small mixed changes. All eight initially declared
160-item refinements completed without scoring errors. A ninth nearby
DeepFilterNet setting (-6 dBFS input, 75% blend) also completed: quality scores
were essentially unchanged from -3 dBFS, with slightly higher WERs for both
recognizers. Thus the lower input level's earlier benefit at full strength did
not carry over to the gentler blend. The tested local refinements now support
proceeding to full-panel confirmation; they do not establish a global optimum.
These screening means do not trigger automatic selection.
All component scores and source aggregates are in
[refinement CSV](quality_v10_refinement.csv) and
[JSONL](quality_v10_refinement.jsonl). The same within-source paired bootstrap
method is applied to the 160-item subset in
[paired refinement CSV](quality_v10_paired_refinement.csv) and
[JSONL](quality_v10_paired_refinement.jsonl); these exploratory intervals do not
calibrate acceptance thresholds.

## Historical pipeline comparison

The repository's v8/v9 training cascade is ClearVoice, Sidon, light de-essing,
FFmpeg de-clicking and peak limiting, DeepFilterNet3, RNNoise85, and peak-safe
source loudness matching. The production script also has recorded fallbacks
for outputs below the loudness measurement gate, including bypassing ClearVoice
or conditioning the signal before DeepFilterNet. See the
[v8 process](expanded_v8_training_cascade_plan.md) and
[implementation](../scripts/preprocess_training_cascade_audio.py).

The current three-model cascade omits the intermediate DSP, RNNoise, and these
historical fallbacks. It is therefore not a reproduction of the previous
training pipeline. Its scores cannot establish an improvement over that
pipeline. The bundled RNNoise implementation is CPU-only. CPU stages are
permitted, and the unnecessary restriction has been removed. The complete
historical recipe has completed processing and scoring on the matched 160-item
panel with zero errors. Its RNNoise archive matches the pinned checksum, and
the other three neural stages have verified GPU tensor placement. One item
required the production fallback that conditions the signal before DeepFilterNet.
The reproducible launcher checks source/output hashes and processing coverage.
All 194 tests pass after removing the unnecessary CPU-stage restriction.

On these same 160 items, the historical recipe improves SigMOS overall and
Audiobox PQ substantially, but Whisper WER is 17.48% versus the control's
14.46%, and ECAPA similarity falls from 0.949 to 0.798. Parakeet WER is 14.69%
versus 13.91%. Mean burst count rises from 2.075 to 2.619. The results show a
perceived-quality/content/speaker tradeoff; they do not establish that either
recipe is best on every metric. The full 800-item baseline also completed with zero errors and two recorded
fallbacks. Whisper WER is 17.796%, Parakeet WER 13.732%, ECAPA similarity 0.799,
and mean burst count 2.185. Against the control, the paired Whisper WER
difference is +1.688 percentage points (95% interval 0.625–2.753); the ECAPA
difference is -0.154 (interval -0.159 to -0.149). The larger panel confirms
the preservation tradeoff despite substantially higher perceived-quality scores.

## Reviewed processing choice

Use [the frozen profile](../conf/quality_v10_processing.yaml): DeepFilterNet3,
input peak normalized to -3 dBFS, input gain reversed after enhancement, 75%
processed audio plus 25% original canonical audio, and peak-safe source loudness
matching. There is no additional trim, fixed loudness target, or HNR rejection.
See the [decision record](quality_v10_processing_decision.json).

On 800 files, SigMOS overall improves by 0.101 versus the canonical control
(95% paired interval 0.087–0.114) and Audiobox PQ by 0.146 (0.135–0.156).
Both quality means improve in every source. Whisper and Parakeet WER are
15.983% and 13.409%; both differences from control have intervals spanning zero.
ECAPA similarity is 0.945, a -0.0082 cost versus control. This is a fidelity/quality
tradeoff, not a claim of maximizing every score or proving ASR equivalence.
The historical cascade has larger quality-score gains but much larger speaker
loss and worse Whisper WER. ClearVoice at 75% is a gentler alternative with
smaller quality gains; it did not improve SigMOS overall in two clean TTS sources.

The initial sweep, nine local refinements, two full-panel confirmations, and
complete historical recipe have been reviewed. Nearby input peaks and fixed
final loudness did not improve the chosen tradeoff. This is the stopping point
for the declared experiments; no global optimum is claimed. Selection is a
manual configuration review. Threshold flags remain report-only and do not
reject recordings or automatically promote checkpoints. Held-out recordings
were not used for this choice.

A processing-throughput check on the fixed 160-file panel measured 3.10, 5.51,
10.58, and 15.59 files/s between the first and last completed outputs for one,
two, four, and eight independent model workers. End-to-end times, including
startup, were 55.28, 35.20, 20.14, and 15.15 seconds. Every output at every worker
count was byte-identical to the already scored audio. Eight workers are selected;
the observed steady-output rate is about five times the one-worker rate on this
panel. These short trials do not establish full-corpus throughput. Mean sampled
GPU power remained only 12–16 W, so faster preprocessing does not imply GPU
saturation. See [throughput CSV](quality_v10_processing_throughput.csv) and
[JSONL](quality_v10_processing_throughput.jsonl). Each model process has independent
state; merged reports require complete unique coverage, and any worker failure
aborts the run. These initial short measurements used long-lived workers. All 195 tests pass, including parallel identity/resume/failure checks.

The subsequent long-lived-worker corpus pass showed growing memory retention
and was stopped before exhausting memory. Production now bounds each model
process to 64 files and releases unused host allocations every eight files.
A complete 800-file validation of this configuration finished processing and
byte-identity verification in 62.77 seconds, with no output differences and no
memory-reserve breach. See the [validation record](quality_v10_bounded_processing_validation.json).
This full-panel validation supersedes the short throughput experiment for the
production execution configuration.

Full-corpus preparation uses eight independent model processes with 64-file
lifetimes, two CPU postprocessing workers per process, and one CPU math thread
per process. Dependent calibration and training jobs are re-chained to this run. Successful preparation
releases the calibration job; successful calibration releases the 100,000-step
training job, which reads the measured recommendation. A failure blocks dependent
jobs. Training has now started with the measured configuration described below.
Milestone evaluation follows the completed training run.

## Full-corpus processing outcome

The selected profile completed all **80,751 recordings / 110.272 hours** on the
eight non-VOA sources. Every output preserves the canonical input frame count.
All model-device records report CUDA, with one processing-adapter version.
The peak-safe loudness cap applied to 7,065 files. One very quiet Telegram
recording fell below the loudness measurement gate and retained the documented
peak-cap-only fallback; it was not rejected.

The monitored processing and subsequent manifest-audit attempt took 82.55 minutes,
with mean sampled GPU power 19.21 W and maximum 24.69 W. Available unified memory
stayed above 94.39 GiB. The job then failed during JSONL export because Parquet
list fields were NumPy arrays. Audio files and their verified hashes were intact.
The manifest exporter now preserves those arrays as JSON lists and writes
atomically, with regression coverage for real Parquet input and failed exports.
All 198 training tests pass. Strict serialization succeeds for all 80,751 source
records. The resumed full input/output/profile audit, manifest export, phoneme
generation, and dataset validation passed. All 80,751 recordings remain present,
with zero clipped files and no validation errors. Calibration and training remain
dependent on successful embedding and statistics preparation.

Embedding extraction initially enabled cuDNN benchmarking for changing audio
lengths. On the fixed 160-file training refinement panel, isolated fresh processes
took 13.714 seconds with benchmarking enabled and 1.514 seconds with it disabled,
both at batch size eight. These timings cover model forward work after audio
loading/resampling, not end-to-end preparation. The minimum cross-process vector
cosine similarity was 0.99999970, with maximum absolute element difference 0.04237;
all vectors were finite. They are numerically close, not byte-identical. Sequential
tests in one process reuse a kernel cache and obscure the initial search cost.
Larger batches did not improve throughput in the exploratory test. A bootstrap
patch now disables this repeated kernel search while retaining the existing batch
size, weights, precision, and padding. See [isolated benchmark results](quality_v10_embedding_benchmark.json).

Preparation is now complete. The final resumed pass, including repeated manifest
validation, embeddings, and statistics, finished successfully in 16.82 minutes.
All 80,751 speaker vectors have 192 finite elements and a nonzero norm. Feature, pitch,
and energy caches cover all 77,468 training and 1,595 development utterances;
their aggregate statistics are finite. See the [preparation audit](quality_v10_preparation.json).
Eight calibration trials at two and four million batch bins completed 100
updates in both GAN optimizers with finite checkpoint weights. The first
eight-million-bin trial depleted the memory reserve and was stopped. Its final
telemetry sampled only 0.074 GiB available while termination was taking effect;
resources recovered afterward. The remaining trials at that size were stopped
or skipped. This is a resource failure, not a completed timing measurement.

The supervisor now checks memory every second independently of GPU telemetry,
uses a 24 GiB reserve for calibration/full training, and kills remaining process
group members after a two-second grace period. All 203 tests pass, including a
descendant that ignores SIGTERM. The intermediate six-million-bin trial also
crossed the reserve (23.97 GiB available) and was stopped. Its incomplete run is
excluded from throughput selection. The recommendation selected four million
batch bins, eight data workers, FP32, and TF32 disabled. It measured 157.31 seconds
for 100 training updates, 2.543 million batch bins/s, mean sampled power 48.62 W,
and maximum sampled power 57.18 W. Batch bins/s is a work proxy, not a measured
audio-sample rate; power averages include initialization and validation.

The 100,000-step run has started with that configuration and the 24 GiB reserve.
The successful calibration trials establish 100-update behavior only; memory
stability over a longer run is still being monitored. See the full
[calibration CSV](quality_v10_training_calibration.csv),
[JSONL](quality_v10_training_calibration.jsonl), and
[configuration decision](quality_v10_training_configuration.json).

## Remaining work

Complete the 100K run, verify sustained memory stability, and compare its checkpoints. See
[run instructions](../quality/README.md).
