---
title: "Ukrainian TTS"
emoji: 🇺🇦
colorFrom: green
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Ukrainian TTS 📢🤖
Ukrainian TTS (text-to-speech) using Coqui TTS.

Link to online demo -> [https://huggingface.co/spaces/robinhad/ukrainian-tts](https://huggingface.co/spaces/robinhad/ukrainian-tts)

Code is licensed under `MIT License`, models are under `GNU GPL v3 License`. 
# Support
If you like my work, please support -> ![mono](https://www.monobank.ua/favicon.ico) [SUPPORT LINK](https://send.monobank.ua/jar/48iHq4xAXm)
# Example

`Mykyta (male)`:

https://user-images.githubusercontent.com/5759207/178158485-29a5d496-7eeb-4938-8ea7-c345bc9fed57.mp4

`Olena (female)`:

https://user-images.githubusercontent.com/5759207/178158492-8504080e-2f13-43f1-83f0-489b1f9cd66b.mp4

# How to use :
1. `pip install -r requirements.txt`.
2. Download model from "Releases" tab.
3. Launch as one-time command:  
```
tts --text "Text for TTS" \
    --model_path path/to/model.pth \
    --config_path path/to/config.json \
    --out_path folder/to/save/output.wav
```
or alternatively launch web server using:
```
tts-server --model_path path/to/model.pth \
    --config_path path/to/config.json
```

# How to train:
1. Refer to ["Nervous beginner guide"](https://tts.readthedocs.io/en/latest/tutorial_for_nervous_beginners.html) in Coqui TTS docs.
2. Instead of provided `config.json` use one from this repo.


# Attribution

- Model training - [Yurii Paniv @robinhad](https://github.com/robinhad)   
- Mykyta and Olena dataset - [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)   
- Autostress (with dictionary) using [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) - [Oleksiy Syvokon @asivokon](https://github.com/asivokon)    
- Autostress (with model) using [ukrainian-accentor](https://github.com/egorsmkv/ukrainian-accentor) - [Bohdan Mykhailenko @NeonBohdan](https://github.com/NeonBohdan) + [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)    
