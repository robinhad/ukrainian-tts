#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="ukrainian-tts",
    version="6.0",
    description="Ukrainian TTS using ESPNET",
    author="Yurii Paniv",
    author_email="mr.robinhad@gmail.com",
    url="https://github.com/robinhad/ukrainian-tts",
    license="MIT",
    packages=find_packages(),
    python_requires=">3.6.0",
    install_requires=[
        "espnet==202301",
        "typeguard<3",
        "num2words @ git+https://github.com/savoirfairelinux/num2words.git@3e39091d052829fc9e65c18176ce7b7ff6169772",
        "ukrainian-word-stress==1.1.0",
        "ukrainian_accentor @ git+https://github.com/egorsmkv/ukrainian-accentor.git@5b7971c4e135e3ff3283336962e63fc0b1c80f4c",
        "stanza==1.7",  #for ukrainian-word-stress
    ],
)
