# Audio Enhancement Steps

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Scope

New enhanced-data runs use this process. The process makes a new audio copy.
It does not change the source audio. Completed datasets keep their original
processing revision and configuration hash.

All training WAV files that this process makes use the complete sequence in
this document. The speaker embedding process is separate. It uses source audio
for 50 percent of the records. It uses enhanced audio for the other 50 percent
of the records.

## Processing Sequence

Do these steps in the specified sequence:

1. Read the source audio as 32-bit floating-point samples.
2. Mix all input channels to one channel.
3. Remove silence from the start and the end of the audio. Use a 40 dB relative
   frame-RMS threshold. Use a frame length of 1,024 samples and a hop length of
   256 samples. Keep 100 ms of boundary padding.
4. Resample the audio to 48 kHz for DeepFilterNet3.
5. Remove noise with DeepFilterNet3 version 0.5.6. Disable its post-filter. Use
   padding. Limit attenuation to 18 dB.
6. Apply a 70 Hz high-pass filter. Use two poles and a Q value of 0.707.
7. Apply the FFmpeg `deesser` filter. Use intensity `0.15`, maximum de-essing
   `0.25`, frequency control `0.50`, and output mode `o`.
8. Apply the FFmpeg RMS compressor. Use threshold `0.1258925412`, ratio `1.5`,
   attack `20 ms`, release `250 ms`, makeup gain `1`, and knee `2.828427`.
9. Run the first EBU R128 loudness pass. Measure integrated loudness, loudness
   range, true peak, threshold, and target offset. Use targets of `-23 LUFS`,
   `7 LU` loudness range, and `-1 dBTP` true peak.
10. Run the second EBU R128 loudness pass. Supply all measurements from the
    first pass. Use linear normalization.
11. Keep the normalized result as 32-bit floating-point audio.
12. Apply the FFmpeg `adeclick` filter. Use window `55`, overlap `75`,
    autoregression order `2`, threshold `4`, and burst fusion `2`.
13. Apply the FFmpeg `alimiter` filter. Use limit `0.891251`, attack `5 ms`,
    release `80 ms`, automatic level `false`, and latency compensation `true`.
14. Write the final WAV as PCM 24-bit.

The final stage does not set a sample rate or a channel count. It keeps these
values from its enhanced input. The DeepFilterNet training path supplies a
24 kHz mono input to this stage.

The FFmpeg mastering filter is equivalent to this filter graph:

```text
highpass=f=70:p=2:t=q:w=0.707,
deesser=i=0.15:m=0.25:f=0.50:s=o,
acompressor=threshold=0.1258925412:ratio=1.5:attack=20:release=250:makeup=1:knee=2.828427:detection=rms
```

The final FFmpeg filter graph is:

```text
aformat=sample_fmts=fltp,adeclick=w=55:o=75:a=2:t=4:b=2,alimiter=limit=0.891251:attack=5:release=80:level=false:latency=true
```

## Output Checks

Check each output file:

- The file exists and is not empty.
- The sample rate is 24 kHz.
- The audio has one channel.
- The WAV subtype is PCM 24-bit.
- All waveform values are finite.
- The sample peak does not exceed `0.891251`.
- The integrated loudness is within 1 LU of `-23 LUFS`.
- The true peak does not exceed `-1 dBTP`.
- The manifest contains the audio SHA-256 value and the enhancement
  configuration hash.

Do not use an output file if processing fails or if the loudness check fails.

## Implementation

The shared final stage is in
`training/audio_enhancement/postprocess.py`. The single-file and directory
entry point is `training/scripts/declick_and_limit_audio.py`. The DeepFilterNet
batch entry point is `training/scripts/preprocess_enhanced_audio.py`. The Sidon
batch entry point is `training/scripts/preprocess_sidon_deess_audio.py`.

The default values are in `training/conf/audio_postprocess.yaml`. Use
`--postprocess-config` to select another YAML or JSON file. Use `--ffmpeg` to
select a compatible FFmpeg executable. The automatic selection uses the pinned
`imageio-ffmpeg` package.
