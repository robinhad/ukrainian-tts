# Random Training-Audio Enhancement Comparison

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Input Selection

The comparison uses 10 random records from the `expanded_v4_trim_only` train
split. The fixed random seed is `20260813`. The process must include at least
one VOA record. This data revision is the latest processed training revision
that includes VOA and has no prior speech denoiser. Thus, it is a fair input
for the five enhancement profiles.

The process copies each input WAV to the comparison directory. It does not
change or replace a training file.

## Processing Controls

The process uses these five profiles:

1. `rnnoise85` uses 85 percent Xiph RNNoise output and 15 percent input audio.
2. `deepfilternet3` uses the default DeepFilterNet3 pretrained model. It does
   not set an attenuation limit. It does not use the optional post-filter.
3. `resemble_denoise` uses only the Resemble Enhance denoiser.
4. `resemble_enhance` uses the complete Resemble Enhance model. It does not use
   the `denoise_only` mode. It uses the command defaults: NFE 64, midpoint
   solver, lambda 1.0, and tau 0.5.
5. `clearervoice` uses the default pretrained `MossFormer2_SE_48K` model.

A model can resample audio only when its input rate requires this operation.
The process returns the result to the input sample rate. It then restores the
exact input sample count and channel count.

The process measures the input integrated loudness with FFmpeg EBU R128. It
applies a linear gain to match each output to this input value. It caps this
gain when the output can exceed -0.1 dBFS. It does not use dynamic loudness
normalization. It does not use de-clicking, silence trim, VAD removal, a
high-pass filter, de-essing, or compression.

All final outputs are PCM 24-bit WAV files. The metadata records HNR, the
estimated noise floor, the peak, integrated loudness, duration, sample count,
sample rate, and channel count before and after the process.

## Licenses and Fixed Revisions

The selected implementations and models permit commercial use:

- Xiph RNNoise code and the included default model use BSD-3-Clause. The code
  commit is `70f1d256acd4b34a572f999a05c87bf00b67730d`. The downloaded model
  archive SHA-256 value starts with `0a8755f8` and is fixed by the RNNoise build
  script.
- DeepFilterNet code and DeepFilterNet3 use MIT or Apache-2.0. The installed
  package version is `0.5.6`.
- Resemble Enhance code and model use MIT. The code commit is
  `8e978149bfe8abab3eb77d965d579a111afdb0ff`. The model revision is
  `4e3510ce4a8391159f665903544c5150bee7b2cb`.
- ClearerVoice code and the `MossFormer2_SE_48K` model use Apache-2.0. The code
  commit is `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`. The model revision is
  `eff8c97925c8bec812af707814b3e5d777fd4503`.

License sources:

- <https://github.com/xiph/rnnoise/blob/main/COPYING>
- <https://github.com/Rikorose/DeepFilterNet>
- <https://github.com/resemble-ai/resemble-enhance/blob/main/LICENSE>
- <https://huggingface.co/ResembleAI/resemble-enhance>
- <https://github.com/modelscope/ClearerVoice-Studio/blob/main/LICENSE>
- <https://huggingface.co/alibabasglab/MossFormer2_SE_48K>

## Commands

Build the pinned RNNoise source:

```bash
git clone https://github.com/xiph/rnnoise.git training/vendor/rnnoise-src
git -C training/vendor/rnnoise-src checkout \
  70f1d256acd4b34a572f999a05c87bf00b67730d
cd training/vendor/rnnoise-src
./autogen.sh
./configure --disable-shared --enable-static
make -j"$(nproc)"
```

Run the comparison:

```bash
training/scripts/run_random_fair_enhancement_comparison.sh
```

## Completed Run

The run completed on 2026-08-13. The automatic validation status is PASS.

- The random selection has 10 input files. Seven files are from VOA.
- The directory has 50 enhanced output files and 50 metadata files.
- All 50 output files use PCM 24-bit encoding.
- All output files have the same sample rate, channel count, duration, and
  sample count as the matched input.
- The maximum absolute loudness difference is 0.02 LU.
- The directory has no symbolic links.
- The complete training test suite has 144 passed tests.

Resemble Enhance ran on GPU 0. Its maximum measured power was 296.03 W.
ClearerVoice ran on GPU 1. Its maximum measured power was 121.24 W. See
`random_training_enhancement_comparison_v1_gpu_power.csv` for the power log.

The output directory is
`training/eval/generated/random_training_enhancement_comparison_v1`.
Use `manifest.tsv` for the listening order. Use `metrics.tsv` for the combined
before-and-after measurements.
