#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='ukrainian-tts',
      version='3.0',
      description='Ukrainian TTS using Coqui TTS',
      author='Yurii Paniv',
      author_email='mr.robinhad@gmail.com',
      url='https://github.com/robinhad/ukrainian-tts',
      packages=find_packages(),
      python_requires='>3.6.0',
      install_requires=[
         "torch>=1.9",
         "TTS==0.8.0",
         "ukrainian-word-stress==1.0.1",
         "ukrainian_accentor @ git+https://github.com/egorsmkv/ukrainian-accentor.git@5b7971c4e135e3ff3283336962e63fc0b1c80f4c"
      ],
)