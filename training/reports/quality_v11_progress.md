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
passed separately. A 256-recording, eight-source benchmark is comparing four
and eight workers before the full run. Training has not started yet.

Run instructions and artifact locations are in the [quality README](../quality/README.md).
