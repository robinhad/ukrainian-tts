# Audio Enhancement Steps

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

## Scope

The expanded-v5 model uses the enhanced expanded-v3 training audio. The
process makes a new audio copy. It does not change the source audio.

All training WAV files use the complete process in this document. The speaker
embedding process is separate. It uses source audio for 50 percent of the
records. It uses enhanced audio for the other 50 percent of the records.

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
11. Write one-channel PCM 16-bit WAV at 24 kHz.

The FFmpeg mastering filter is equivalent to this filter graph:

```text
highpass=f=70:p=2:t=q:w=0.707,
deesser=i=0.15:m=0.25:f=0.50:s=o,
acompressor=threshold=0.1258925412:ratio=1.5:attack=20:release=250:makeup=1:knee=2.828427:detection=rms
```

## Output Checks

Check each output file:

- The file exists and is not empty.
- The sample rate is 24 kHz.
- The audio has one channel.
- All waveform values are finite.
- The integrated loudness is within 1 LU of `-23 LUFS`.
- The true peak does not exceed `-1 dBTP`.
- The manifest contains the audio SHA-256 value and the enhancement
  configuration hash.

Do not use an output file if processing fails or if the loudness check fails.

## Implementation

The implementation is in
`training/audio_enhancement/pipeline.py`. The batch entry point is
`training/scripts/preprocess_enhanced_audio.py`.
