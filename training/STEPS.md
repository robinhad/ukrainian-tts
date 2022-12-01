Setup env
Link: https://espnet.github.io/espnet/installation.html

1. `git clone https://github.com/espnet/espnet`
2. `cd ./espnet/tools`
3. `CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..`
4. `./setup_anaconda.sh ${CONDA_TOOLS_DIR} espnet 3.8`
5. `make`
6. ??? CUDA ???
7. `. ./activate_python.sh; python3 check_install.py`