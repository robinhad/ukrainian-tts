# Setup environment
Link: https://espnet.github.io/espnet/installation.html

```sh
sudo apt-get install cmake sox libsndfile1-dev ffmpeg
git clone --branch v.202301 https://github.com/espnet/espnet
cd ./espnet/tools
./setup_anaconda.sh anaconda espnet 3.10
. ./activate_python.sh
make
pip install --upgrade torch torchaudio # or setup same versions
make
. ./activate_python.sh; python3 check_install.py
```

# Run training

ESPNET is a dynamic framework. For the latest guide, please refer to https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/tts1

This page provides general launching steps on how training was performed for reference, and this doesn't cover data preparation.

NOTE: before running the script below, copy [./train_vits.yaml](./train_vits.yaml) or [./finetune_joint_tacotron2_hifigan.yaml](./finetune_joint_tacotron2_hifigan.yaml) to your `<espnet_root>/egs2/ljspeech/tts1/conf/tuning/` folder


```sh
cd ../egs2/ljspeech/tts1
pip install torchvision # to save figures
pip install speechbrain # for x-vectors
# option 1: train VITS
./run.sh \
    --stage 6 \
    --min_wav_duration 0.38 \
    --use_xvector true \
    --xvector_tool speechbrain \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --train_config ./conf/tuning/train_vits.yaml \
    --inference_config ./conf/tuning/decode_vits.yaml
# option 2: train tacotron2 and hifigan jointly
./run.sh \
    --stage 6 \
    --min_wav_duration 0.38 \
    --use_xvector true \
    --xvector_tool speechbrain \
    --fs 22050 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --dumpdir dump/22k \
    --expdir exp/22k \
    --train_config ./conf/tuning/finetune_joint_tacotron2_hifigan.yaml \
    --tts_task gan_tts

```


