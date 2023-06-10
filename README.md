---
title: "Ukrainian TTS"
emoji: 🐌
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version : 3.34
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

`Tetiana (female)`:

https://user-images.githubusercontent.com/5759207/224504324-d8236cad-7302-4dfd-9696-7a42b9f05fce.mp4


<details>
  <summary>More voices 📢🤖</summary>

`Dmytro (male)`:

https://user-images.githubusercontent.com/5759207/224504354-f84f74d3-fa46-497c-9604-4b63ba45989f.mp4


`Lada (female)`:

https://user-images.githubusercontent.com/5759207/224504360-ec198ac2-647c-4238-99ef-b6f074d633fd.mp4


`Mykyta (male)`:

https://user-images.githubusercontent.com/5759207/224504363-0227e8bf-8c1c-49ad-8602-8cbf8feaa82b.mp4

</details>


# How to use: 📢

See example notebook: [tts_example.ipynb](./tts_example.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/robinhad/ukrainian-tts/blob/main/tts_example.ipynb)

# How to train: 🏋️
Link to guide: [training/STEPS.md](training/STEPS.md)


# Attribution 🤝

- Model training - [Yurii Paniv @robinhad](https://github.com/robinhad)   
- [Open Source Ukrainian Text-to-Speech dataset](https://github.com/egorsmkv/ukrainian-tts-datasets) - [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)   
- Dmytro voice - [Dmytro Chaplynskyi @dchaplinsky](https://github.com/dchaplinsky)  
- Silence cutting using [HMM-GMM](https://github.com/proger/uk) - [Volodymyr Kyrylov @proger](https://github.com/proger)  
- Autostress (with dictionary) using [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) - [Oleksiy Syvokon @asivokon](https://github.com/asivokon)    
- Autostress (with model) using [ukrainian-accentor](https://github.com/egorsmkv/ukrainian-accentor) - [Bohdan Mykhailenko @NeonBohdan](https://github.com/NeonBohdan) + [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)    
