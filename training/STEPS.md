Setup env
Link: https://espnet.github.io/espnet/installation.html

0. `sudo apt-get install cmake sox libsndfile1-dev ffmpeg`
1. `git clone https://github.com/espnet/espnet@cb06bb1a9e5e5355a02d8c1871a8ecfafd54754d`
`conda create -p ./.venv python=3.8`
 `conda install -c anaconda cudatoolkit` 
2. `cd ./espnet/tools`
3. `CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..`
./setup_anaconda.sh ${CONDA_TOOLS_DIR} espnet 3.8
5. `make`
pip install --upgrade torch torchaudio # or setup same versions
make
7. `. ./activate_python.sh; python3 check_install.py`

# run training

cd ../egs2/ljspeech/tts1
./run.sh 