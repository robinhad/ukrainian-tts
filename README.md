---
title: "Ukrainian TTS"
emoji: 🐌
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version : 3.40.1
python_version: 3.10.3
app_file: app.py
pinned: false
---

# Ukrainian TTS 📢🤖
Ukrainian TTS (text-to-speech) using ESPNET.

![pytest](https://github.com/robinhad/ukrainian-tts/actions/workflows/hf-sync.yml/badge.svg)
[![Open In HF🤗 Space ](https://img.shields.io/badge/Open%20Demo-%F0%9F%A4%97%20Space-yellow)](https://huggingface.co/spaces/robinhad/ukrainian-tts)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robinhad/ukrainian-tts/blob/main/tts_example.ipynb)
[![Open Bot](https://img.shields.io/badge/Open%20Bot%20🤖-Telegram-blue)](https://t.me/uk_tts_bot)
[![chat](https://img.shields.io/badge/chat-Telegram-blue)](https://t.me/speech_recognition_uk)

Link to online demo -> [https://huggingface.co/spaces/robinhad/ukrainian-tts](https://huggingface.co/spaces/robinhad/ukrainian-tts)  
Note: online demo saves user input to improve user experience; by using it, you consent to analyze this data.   
Link to source code and models -> [https://github.com/robinhad/ukrainian-tts](https://github.com/robinhad/ukrainian-tts)  
Telegram bot -> [https://t.me/uk_tts_bot](https://t.me/uk_tts_bot)  

# Features ⚙️
- Completely offline
- Multiple voices
- Automatic stress with priority queue: `acute` -> `user-defined` > `dictionary` > `model`
- Control speech speed
- Python package works on Windows, Mac (x86/M1), Linux(x86/ARM)
- Inference on mobile devices (inference models through `espnet_onnx` without cleaners)


# Support ❤️
If you like my work, please support ❤️ -> [https://send.monobank.ua/jar/48iHq4xAXm](https://send.monobank.ua/jar/48iHq4xAXm)   
You're welcome to join UA Speech Recognition and Synthesis community: [Telegram https://t.me/speech_recognition_uk](https://t.me/speech_recognition_uk)
# Examples 🤖

`Oleksa (male)`:

https://github.com/robinhad/ukrainian-tts/assets/5759207/ace842ef-06d0-4b1f-ad49-5fda92999dbb


<details>
  <summary>More voices 📢🤖</summary>

`Tetiana (female)`:

https://github.com/robinhad/ukrainian-tts/assets/5759207/a6ecacf6-62ae-4fc5-b6d5-41e6cdd3d992

`Dmytro (male)`:

https://github.com/robinhad/ukrainian-tts/assets/5759207/67d3dac9-6626-40ef-98e5-ec194096bbe0

`Lada (female)`:

https://github.com/robinhad/ukrainian-tts/assets/5759207/fcf558b2-3ff9-4539-ad9e-8455b52223a4

`Mykyta (male)`:

https://github.com/robinhad/ukrainian-tts/assets/5759207/033f5215-3f09-4021-ba19-1f55158445ca


</details>


# How to use: 📢

## Quickstart

Install using: 
```bash
!pip install git+https://github.com/robinhad/ukrainian-tts.git
```
Code example:
```python
from ukrainian_tts.tts import TTS, Voices, Stress
import IPython.display as ipd

tts = TTS(device="cpu") # can try gpu, mps
with open("test.wav", mode="wb") as file:
    _, output_text = tts.tts("Привіт, як у тебе справи?", Voices.Dmytro.value, Stress.Dictionary.value, file)
print("Accented text:", output_text)

ipd.Audio(filename="test.wav")
```

See example notebook: [tts_example.ipynb](./tts_example.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robinhad/ukrainian-tts/blob/main/tts_example.ipynb)

# How to contribute: 🙌

Look into this list with current problems: https://github.com/robinhad/ukrainian-tts/issues/35

# How to train: 🏋️
Link to guide: [training/STEPS.md](training/STEPS.md)


# Attribution 🤝

- Model training - [Yurii Paniv @robinhad](https://github.com/robinhad)   
- [Open Source Ukrainian Text-to-Speech dataset](https://github.com/egorsmkv/ukrainian-tts-datasets) - [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)   
- Dmytro voice - [Dmytro Chaplynskyi @dchaplinsky](https://github.com/dchaplinsky)  
- Silence cutting using [HMM-GMM](https://github.com/proger/uk) - [Volodymyr Kyrylov @proger](https://github.com/proger)  
- Autostress (with dictionary) using [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) - [Oleksiy Syvokon @asivokon](https://github.com/asivokon)    
- Autostress (with model) using [ukrainian-accentor](https://github.com/egorsmkv/ukrainian-accentor) - [Bohdan Mykhailenko @NeonBohdan](https://github.com/NeonBohdan) + [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)    
