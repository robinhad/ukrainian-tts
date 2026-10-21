# Multi-speaker Ukrainian VITS transfer

This run adapts [ESPNet's LibriTTS x-vector VITS](https://huggingface.co/espnet/kan-bayashi_libritts_xvector_vits)
from its English weights for **50,000 Ukrainian steps**. The model card declares
CC BY 4.0. Revision and SHA-256 pins are in `vits.inputs.lock.json`. This is a
separate experiment from the Tetiana JETS continuation.

## Conversion and learning rates

The source has 86 ARPAbet tokens and 192-dimensional embeddings. The target uses
the same 100-token Ukrainian inventory and `TransferFrontend` as JETS:
eSpeak-ng Ukrainian, preserved stress and phone boundaries, explicit lexical
stress handling, and a verified implementation snapshot. Equivalent IPA sounds
copy donor rows; other phones use PHOIBLE matching or independent random rows
where reliable mapping is unavailable. Ukrainian `r` explicitly copies English
`R` (`ɹ`), following the rhotic experiment. Rows remain independently trainable.

The converter replaces `tts.generator.text_encoder.emb.weight` and verifies
that all **1,062 other tensors** are unchanged after strict loading. The source
scale-discriminator uses plain weights, so its advertised normalization option
is disabled to match them; no discriminator tensors are reinitialized. All
token mappings are saved as JSON/CSV. Donor relative-position attention is
retained: the plain JETS FlashAttention adapter is incompatible with it.

All weights are trainable. AdamW starts fresh: **0.0002 for text embeddings**,
**0.00002 for the remaining generator and discriminator**, donor betas/eps and
zero weight decay. Exponential decay is 0.999875 per 1,000 steps. Batch size is
8, seed 777, FP32, and the donor's discriminator-first update order is retained.

## Audio and statistics

Audio is **22,050 Hz**, FFT 1024, hop 256. The posterior encoder receives 513-bin
**linear amplitude spectra without dataset mean/std normalization**. Waveform
mel loss retains the donor's 80 mel bins, fmin 0 and Nyquist fmax. There is no
pitch or energy extractor and no acoustic normalization module. Therefore
**no dataset-level acoustic statistics need recalculation**. Configuration and
loaded-model checks enforce this. JETS statistics must not be copied into VITS;
learned neural normalization parameters remain in the preserved weights.

Existing V12 non-VOA processing is retained. The ≥3.5 SigMOS filter was applied
at 24 kHz, before preparation at 22.05 kHz. Listening scores measure the actual
22.05 kHz files and can be below 3.5. No new filter is silently applied.

## Speaker conditioning and data

This donor requires **512D Kaldi SITW x-vectors**, not the 192D ECAPA vectors from
the earlier Ukrainian JETS model. `kaldi_xvectors.py` uses the same
[SITW system 1a](https://kaldi-asr.org/models/m8) and
[Kaldi extraction recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/sre08/v1/sid/nnet3/xvector/extract_xvectors.sh)
as ESPnet LibriTTS: 16 kHz audio, 30 MFCCs, 25 ms frames, 20–7600 Hz, no edge
snipping, energy VAD, and centered 300-frame mean normalization without variance
normalization. The extractor uses `tdnn6.affine`, minimum chunk 25 and maximum
chunk 10,000. VITS performs its own vector normalization and projection.

These are per-recording operations, not corpus statistics. The installed Kaldi
build runs on CPU; training, synthesis, phone recognition and MOS run on GPU.
All 10,187 train and 196 development recordings have valid vectors; none were
dropped. Inference uses the same extractor's reference vector. Validation
synthesis receives text and a vector, not reference durations or posterior
acoustic features.

Training totals **16.5049 hours**, development **0.3352 hours**. All eight
training sources are used:

| Source | Items |
|---|---:|
| Ukrainian dialects | 4,670 |
| Available Ukrainian Common Voice | 3,822 |
| OpenTTS Mykyta | 581 |
| OpenTTS Tetiana | 553 |
| FLEURS Ukrainian | 352 |
| OpenTTS Lada | 91 |
| Ukrainian Telegram voices | 86 |
| UA SER | 32 |

Many speaker labels are per-recording pseudo-identities, not verified people.
Conditioning uses audio-derived vectors. Existing train/dev splits are retained,
with no overlapping utterance IDs. The fixed listening panel takes up to two
development items per source: 13 recordings from seven sources. UA SER has no
development items in this split.

Checkpoints and validation run every 1,000 steps; the fixed listening panel is
synthesized every 5,000 steps and scored with SigMOS, Audiobox PQ and independent
rhotic recognition, plus clipping/duration flags. Selection stays report-only.
Optimizer and scheduler state survive evaluation boundaries. Only numeric
metrics go to W&B project `ukrainian-tts`; no model/audio/code artifacts are
uploaded. Resource checks and Kyiv ETA are recorded every minute.

## Reproduce

Use the existing training environment and verified prepared Ukrainian data.
Download the donor and Kaldi archive with the pinned lock:

```bash
source training/activate.sh
export UKTTS_ESPEAK_DATA_HASH_FILE=training/quality_runs/phonetic_transfer/frontend/ESPEAK_NG_DATA_HASH
VITS_ROOT=training/quality_runs/vits_transfer
python -m training.transfer.download --lock training/transfer/vits.inputs.lock.json --output "$VITS_ROOT"
tar -xzf "$VITS_ROOT/kaldi/0008_sitw_v2_1a.tar.gz" -C "$VITS_ROOT/kaldi"
python -m training.transfer.vits convert \
  --source-config "$VITS_ROOT/donor/exp/tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space/config.yaml" \
  --source-model "$VITS_ROOT/donor/exp/tts_train_xvector_vits_raw_phn_tacotron_g2p_en_no_space/train.total_count.ave_10best.pth" \
  --target-config training/quality_runs/phonetic_transfer/converted/config.yaml \
  --features training/quality_runs/phonetic_transfer/phoible-segments-features.tsv \
  --output "$VITS_ROOT/converted"
python -m training.transfer.kaldi_xvectors \
  --prepared training/quality_runs/phonetic_transfer \
  --model "$VITS_ROOT/kaldi/0008_sitw_v2_1a/exp/xvector_nnet_1a" \
  --prefix training/quality_runs/mfa_trim/tools/env \
  --output "$VITS_ROOT/xvectors" --workers 8
python -m training.transfer.vits prepare --bundle "$VITS_ROOT/converted" \
  --prepared training/quality_runs/phonetic_transfer --xvectors "$VITS_ROOT/xvectors" \
  --manifests training/data/quality_v12/manifests \
  --output "$VITS_ROOT/multispeaker_50000" --steps 50000 --batch-size 8
```

The reusable audio/text comes from `training.transfer.prepare`. The VITS
preparer verifies its rate, frontend snapshot and manifest hashes. Provide a
Kaldi prefix with MFCC, VAD, CMVN and nnet3 executables; the existing MFA runtime
contains them. For the native alignment implementation, install `Cython==3.3.0`
into the training environment and run `python setup.py build_ext --inplace` in
`espnet2/gan_tts/vits/monotonic_align` under the ESPnet source tree. Build products
stay untracked.

Within a GPU allocation, run smoke and baseline evaluation; then submit:

```bash
python -m training.transfer.vits smoke --root "$VITS_ROOT/multispeaker_50000"
python -m training.transfer.vits_evaluate --root "$VITS_ROOT/multispeaker_50000" --step 0
sbatch --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=96G --time=48:00:00 \
  --export=ALL,VITS_ROOT="$VITS_ROOT/multispeaker_50000" \
  training/scripts/run_vits_transfer.sh
```

Add a dependency if another GPU job is running. A failed stage stops and
preserves logs/checkpoints. Resume with `vits train --root RUN --epochs
TARGET_EPOCH --resume`; do not rerun the fresh-start runner over existing output.

Inference shares the preparation frontend:

```bash
python -m training.transfer.vits infer --root "$VITS_ROOT/multispeaker_50000" \
  --checkpoint "$VITS_ROOT/multispeaker_50000/train_model/5epoch.pth" \
  --reference-id REFERENCE_UTTERANCE_ID --text 'Ранок. Лада читає.' --output sample.wav
```

Choose a reference ID from the run's train/dev `xvector.scp`. Listening pages
are `evaluation_STEP/preview.html`; per-file JSONL and summaries sit alongside.
For an on-demand listening evaluation, run `python -m
training.transfer.vits_evaluate --root RUN --step STEP` within a GPU allocation.
Any saved 1K-step checkpoint can be evaluated; automatic evaluations remain
every 5K steps. The evaluator requires a new `evaluation_STEP` directory and
preserves existing evaluations by refusing to overwrite them.

## Verification

GPU smoke tests exercise both GAN updates, finite gradients, the 10× embedding
LR, valid synthesis and different output waveforms when speaker vectors change.
A separate trainer ran two ten-step epochs with a save/resume boundary: both
optimizers reached step 20 and both schedulers epoch 2. Initial run weights were
not modified. The inference CLI was checked against the shared text frontend.
Portable results and complete mappings are in `training/reports/vits_transfer_preflight.json`
and `training/reports/vits_transfer_mapping.json`.
