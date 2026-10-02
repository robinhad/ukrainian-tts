# V11: selected non-VOA corpus and 50K training

The requested policy is to run ClearVoice → Sidon → DeepFilterNet3 at full
strength, select its output only when overall SigMOS exceeds the native
original, and retain the chosen item only at SigMOS ≥3.5. Training will start
from scratch for 50,000 updates. The stopped V10 checkpoint remains preserved.

The complete 80,751-recording corpus will be measured. The earlier estimate of
about 28–29.5 retained hours is provisional; the full run reports actual counts,
durations, score distributions and Ukrainian letter coverage. Train/dev keep
their original split assignments. Evaluation retains the fixed unfiltered
population and the same 88-item Ukrainian quality panel.

The streaming pipeline stores only selected audio, bounds temporary storage
and model-process lifetimes, and verifies hashes/provenance before reuse.
Enhancement and SigMOS neural operations use CUDA. Original winners retain
native timing and are rendered as mono 24 kHz PCM24; their rendered SigMOS is
also reported. The choice and cutoff use the native-original/cascade scores.
Checkpoint evaluation remains report-only and has no HNR rejection.

Initial validation recovered and scored 16 recordings from all eight sources.
Original SigMOS reproduced the previous panel exactly. The unseeded cascade
varied by up to 0.237 SigMOS relative to that earlier sweep. Investigation
identified ClearVoice's random Kaldi feature dither and a first-call numerical
change from Sidon's profiled TorchScript executor. The V11 profile now uses
per-recording seeds and disables that executor optimization. Three repeated
passes of a diagnostic recording then produced identical stage outputs and
SigMOS. See [reproducibility measurements](quality_v11_reproducibility.json).

A larger benchmark caught malformed UTF-8 container tags. Native decoding now
replaces invalid metadata bytes while leaving decoded PCM unchanged; a real
malformed-tag regression fixture verifies this behavior. The relevant 23
pipeline tests passed in the GPU allocation, and the additional metadata test
passed separately. The 256-recording, eight-source benchmark completed in
100.85 seconds with four workers and 85.72 seconds with eight workers. All 256
processed scores, selection decisions and retained audio hashes matched exactly
between worker counts. Eight workers reached 50.8 W peak power with at least
81.6 GiB available memory. See the [benchmark report](quality_v11_processing_benchmark.json).
Production uses eight workers with 64-file lifetimes, preserving a 24 GiB
available-memory guard. The longer production lifetimes and corpus source mix
may change throughput and memory use. Training starts after full-corpus selection
and preparation complete.

In the benchmark, 19 of 93 retained items had rendered 24 kHz scores below 3.5
after choosing their higher-scoring native originals. These remain retained
under the requested native-original selection rule; the output scores and
counts are reported separately so they are not mistaken for a guarantee about
the rendered training waveform.

Run instructions and artifact locations are in the [quality README](../quality/README.md).

Training and validation scalar metrics are configured for the `ukrainian-tts`
W&B project, under run name `quality-v11-50k`. Authentication and run creation
were verified before training. The uploader polls local TensorBoard scalars
every 30 seconds and preserves optimizer iteration numbers. Model checkpoints,
audio, source files, machine metadata, and console logs are not uploaded.
Native ESPnet model logging is explicitly disabled. Scalar filtering, resume
cursors, and same-step epoch updates passed the uploader regression checks.

## Duration comparison on the completed pilot

Using the earlier completed 800-recording, source-balanced evaluation, choosing
the higher overall SigMOS and retaining scores ≥3.5 leaves 267 recordings.
Native-original duration versus selected duration has mean 5.759 → 5.775 s,
median 5.280 → 5.280 s, middle 50% 3.416–6.903 → 3.403–6.990 s, and
5th–95th percentile 2.303–12.240 → 2.323–12.702 s. The share at least 10 s
increases from 9.625% to 11.236%.

These aggregate similarities combine two effects: retained items had longer
native originals (mean 6.249 s), and choosing the processed version where it
improved quality removed an average 0.474 s across the retained items. Original
winners keep their native timing. The sample contains 100 items per source
before filtering, so its length distribution does not represent full-corpus
source proportions. These are the earlier pilot scores, not final V11 results.

The [duration chart](quality_v11_pilot_duration_distribution.png),
[summary CSV](quality_v11_pilot_duration_distribution.csv), and
[duration bands](quality_v11_pilot_duration_distribution_bins.csv) are generated
by `training/scripts/report_selected_duration_distribution.py`. The same
reports are available as PDF/JSONL, with opaque per-item duration exports.

## Full-corpus selection progress

The [progress snapshot](quality_v11_selection_progress.json) records 10,304
completed Common Voice decisions, retaining 1,897 recordings (2.524 hours):
1,143 processed winners and 754 original winners. Retained selected SigMOS has
median 3.636 and mean 3.670. After rendering as training audio, 44 retained items
score below 3.5; the native-original selection policy remains unchanged.
This snapshot covers only the first source being processed and must not be
extrapolated directly to the complete eight-source corpus. Training has not
started at this snapshot; processing and monitoring continue.
