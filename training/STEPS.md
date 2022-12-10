Setup env
Link: https://espnet.github.io/espnet/installation.html

0. `sudo apt-get install cmake sox libsndfile1-dev ffmpeg`
1. `git clone --branch v.202209 https://github.com/espnet/espnet`
2. `cd ./espnet/tools`
./setup_anaconda.sh anaconda espnet 3.8
3. `CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..`
./setup_anaconda.sh ${CONDA_TOOLS_DIR} espnet 3.8
5. `make`
pip install --upgrade torch torchaudio # or setup same versions
make
7. `. ./activate_python.sh; python3 check_install.py`

# run training

cd ../egs2/ljspeech/tts1
./run.sh 

./run.sh \
    --stage 2 \
    --use_sid true \
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