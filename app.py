import tempfile
from typing import Optional

import gradio as gr
import numpy as np

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import requests
from os.path import exists

MODEL_NAMES = [
    "uk/mai/glow-tts"
]
MODELS = {}

manager = ModelManager()


def download(url, file_name):
    if not exists(file_name):
        print(f"Downloading {file_name}")
        r = requests.get(url, allow_redirects=True)
        with open(file_name, 'wb') as file:
            file.write(r.content)
    else:
        print(f"Found {file_name}. Skipping download...")


for MODEL_NAME in MODEL_NAMES:
    print(f"downloading {MODEL_NAME}")
    model_path, config_path, model_item = manager.download_model(
        f"tts_models/{MODEL_NAME}")
    vocoder_name: Optional[str] = model_item["default_vocoder"]
    release_number = "0.0.1"
    vocoder_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/v{release_number}/vocoder.pth.tar"
    vocoder_config_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/v{release_number}/vocoder_config.json"

    vocoder_path = "vocoder.pth.tar"
    vocoder_config_path = "vocoder_config.json"

    download(vocoder_link, vocoder_path)
    download(vocoder_config_link, vocoder_config_path)

    synthesizer = Synthesizer(
        model_path, config_path, None, vocoder_path, vocoder_config_path,
    )
    MODELS[MODEL_NAME] = synthesizer


def tts(text: str, model_name: str):
    print(text, model_name)
    synthesizer = MODELS.get(model_name, None)
    if synthesizer is None:
        raise NameError("model not found")
    wavs = synthesizer.tts(text)
    # output = (synthesizer.output_sample_rate, np.array(wavs))
    # return output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        synthesizer.save_wav(wavs, fp)
        return fp.name


iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.inputs.Textbox(
            label="Input",
            default="Привіт, як твої справи?",
        ),
        gr.inputs.Radio(
            label="Виберіть TTS модель",
            choices=MODEL_NAMES,
        ),
    ],
    outputs=gr.outputs.Audio(label="Output"),
    title="🐸💬🇺🇦 - Coqui TTS",
    theme="huggingface",
    description="Україномовний🇺🇦 TTS за допомогою Coqui TTS",
    article="Якщо вам подобається, підтримайте за посиланням: [SUPPORT LINK](https://send.monobank.ua/jar/48iHq4xAXm)",
)
iface.launch()
