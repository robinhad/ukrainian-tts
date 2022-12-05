---
title: "Ukrainian TTS"
emoji: 🐌
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version : 3.3
python_version: 3.9
app_file: app.py
pinned: false
---

# Ukrainian TTS 📢🤖
Ukrainian TTS (text-to-speech) using Coqui TTS.

![pytest](https://github.com/robinhad/ukrainian-tts/actions/workflows/hf-sync.yml/badge.svg) [![Open In HF🤗 Space ](https://img.shields.io/badge/Open%20Demo-%F0%9F%A4%97%20Space-yellow)](https://huggingface.co/spaces/robinhad/ukrainian-tts)

Link to online demo -> [https://huggingface.co/spaces/robinhad/ukrainian-tts](https://huggingface.co/spaces/robinhad/ukrainian-tts)  
Note: online demo saves user input to improve user experience, by using it you give your consent to analyze this data.   
Link to source code and models -> [https://github.com/robinhad/ukrainian-tts](https://github.com/robinhad/ukrainian-tts)  
Telegram bot -> [https://t.me/uk_tts_bot](https://t.me/uk_tts_bot)  

Code is licensed under `MIT License`, models are under `GNU GPL v3 License`. 
# Support ❤️
If you like my work, please support ❤️ -> [https://send.monobank.ua/jar/48iHq4xAXm](https://send.monobank.ua/jar/48iHq4xAXm)  
For collaboration and question please contact me here:  
[Telegram https://t.me/robinhad](https://t.me/robinhad)  
[Twitter https://twitter.com/robinhad](https://twitter.com/robinhad)  
You're welcome to join UA Speech Recognition and Synthesis community: [Telegram https://t.me/speech_recognition_uk](https://t.me/speech_recognition_uk)
# Examples 🤖

`Mykyta (male)`:

https://user-images.githubusercontent.com/5759207/190852232-34956a1d-77a9-42b9-b96d-39d0091e3e34.mp4


`Olena (female)`:

https://user-images.githubusercontent.com/5759207/190852238-366782c1-9472-45fc-8fea-31346242f927.mp4


`Dmytro (male)`:

https://user-images.githubusercontent.com/5759207/190852251-db105567-52ba-47b5-8ec6-5053c3baac8c.mp4

`Olha (female)`:

https://user-images.githubusercontent.com/5759207/190852259-c6746172-05c4-4918-8286-a459c654eef1.mp4

`Lada (female)`:


https://user-images.githubusercontent.com/5759207/190852270-7aed2db9-dc08-4a9f-8775-07b745657ca1.mp4

# How to use: 📢
## As a package:
1. Install using command:
```
pip install git+https://github.com/robinhad/ukrainian-tts.git
```
2. Run a code snippet:
```python
from ukrainian_tts.tts import TTS, Voices, Stress

tts = TTS(use_cuda=False)
with open("test.wav", mode="wb") as file:
    _, text = tts.tts("Привіт", Voices.Dmytro.value, Stress.Model.value, file)
print("Accented text:", text)
```

## Run manually:  
`Caution: this won't use normalizer and autostress like a web demo. ` 
1. `pip install -r requirements.txt`.
2. Download `model.pth` and `speakers.pth` from "Releases" tab.
3. Launch as one-time command:  
```
tts --text "Text for TTS" \
    --model_path path/to/model.pth \
    --config_path path/to/config.json \
    --speaker_idx dmytro \
    --out_path folder/to/save/output.wav
```
or alternatively launch web server using:
```
tts-server --model_path path/to/model.pth \
    --config_path path/to/config.json
```

# How to train: 🏋️
1. Refer to ["Nervous beginner guide"](https://tts.readthedocs.io/en/latest/tutorial_for_nervous_beginners.html) in Coqui TTS docs.
2. Instead of provided `config.json` use one from this repo.


# Attribution 🤝

- Model training - [Yurii Paniv @robinhad](https://github.com/robinhad)   
- Mykyta, Olena, Lada, Dmytro, Olha dataset - [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)   
- Dmytro voice - [Dmytro Chaplynskyi @dchaplinsky](https://github.com/dchaplinsky)  
- Silence cutting using [HMM-GMM](https://github.com/proger/uk) - [Volodymyr Kyrylov @proger](https://github.com/proger)  
- Autostress (with dictionary) using [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) - [Oleksiy Syvokon @asivokon](https://github.com/asivokon)    
- Autostress (with model) using [ukrainian-accentor](https://github.com/egorsmkv/ukrainian-accentor) - [Bohdan Mykhailenko @NeonBohdan](https://github.com/NeonBohdan) + [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)    
