import tempfile

import gradio as gr

from TTS.utils.synthesizer import Synthesizer
import requests
from os.path import exists
from formatter import preprocess_text
from datetime import datetime
from enum import Enum
import torch


class StressOption(Enum):
    AutomaticStress = "Автоматичні наголоси (за словником)"
    AutomaticStressWithModel = "Автоматичні наголоси (за допомогою моделі)"


class VoiceOption(Enum):
    FemaleVoice = "Олена (жіночий)"
    MaleVoice = "Микита (чоловічий)"


def download(url, file_name):
    if not exists(file_name):
        print(f"Downloading {file_name}")
        r = requests.get(url, allow_redirects=True)
        with open(file_name, "wb") as file:
            file.write(r.content)
    else:
        print(f"Found {file_name}. Skipping download...")


print("downloading uk/mykyta/vits-tts")
release_number = "v2.0.0"
model_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/model-inference.pth"
config_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/config.json"
speakers_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/speakers.pth"

model_path = "model.pth"
config_path = "config.json"
speakers_path = "speakers.pth"

download(model_link, model_path)
download(config_link, config_path)
download(speakers_link, speakers_path)

badge = (
    "https://visitor-badge-reloaded.herokuapp.com/badge?page_id=robinhad.ukrainian-tts"
)

synthesizer = Synthesizer(
    model_path,
    config_path,
    speakers_path,
    None,
    None,
)

if synthesizer is None:
    raise NameError("model not found")


def tts(text: str, voice: str, stress: str):
    print("============================")
    print("Original text:", text)
    print("Voice", voice)
    print("Stress:", stress)
    print("Time:", datetime.utcnow())
    autostress_with_model = True if stress == StressOption.AutomaticStressWithModel.value else False
    speaker_name = "male1" if voice == VoiceOption.MaleVoice.value else "female3"
    text = preprocess_text(text, autostress_with_model)
    text_limit = 1200
    text = (
        text if len(text) < text_limit else text[0:text_limit]
    )  # mitigate crashes on hf space
    print("Converted:", text)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        with torch.no_grad():
            wavs = synthesizer.tts(text, speaker_name=speaker_name)
            synthesizer.save_wav(wavs, fp)
        return fp.name, text


iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.inputs.Textbox(
            label="Input",
            default="Введіть, будь ласка, своє р+ечення.",
        ),
        gr.inputs.Radio(
            label="Голос",
            choices=[option.value for option in VoiceOption],
            default=VoiceOption.FemaleVoice.value
        ),
        gr.inputs.Radio(
            label="Наголоси",
            choices=[option.value for option in StressOption],
        ),
    ],
    outputs=[
        gr.outputs.Audio(label="Output"),
        gr.outputs.Textbox(label="Наголошений текст"),
    ],
    title="🐸💬🇺🇦 - Coqui TTS",
    theme="huggingface",
    description="Україномовний🇺🇦 TTS за допомогою Coqui TTS (щоб вручну поставити наголос, використовуйте + перед голосною)",
    article="Якщо вам подобається, підтримайте за посиланням: [SUPPORT LINK](https://send.monobank.ua/jar/48iHq4xAXm),  "
    + "Github: [https://github.com/robinhad/ukrainian-tts](https://github.com/robinhad/ukrainian-tts)   \n"
    + "Model training - [Yurii Paniv @robinhad](https://github.com/robinhad)   \n"
    + "Mykyta and Olena dataset - [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)   \n"
    + "Autostress (with dictionary) using [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) - [Oleksiy Syvokon @asivokon](https://github.com/asivokon)    \n"
    + "Autostress (with model) using [ukrainian-accentor](https://github.com/egorsmkv/ukrainian-accentor) - [Bohdan Mykhailenko @NeonBohdan](https://github.com/NeonBohdan) + [Yehor Smoliakov @egorsmkv](https://github.com/egorsmkv)    \n"
    + f'<center><img src="{badge}" alt="visitors badge"/></center>',
    examples=[
        [
            "Введіть, будь ласка, своє речення.",
            VoiceOption.FemaleVoice.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Введіть, будь ласка, своє речення.",
            VoiceOption.MaleVoice.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Вв+едіть, будь ласка, св+оє реч+ення.",
            VoiceOption.MaleVoice.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Привіт, як тебе звати?",
            VoiceOption.FemaleVoice.value,
            StressOption.AutomaticStress.value,
        ],
        [
            "Договір підписано 4 квітня 1949 року.",
            VoiceOption.FemaleVoice.value,
            StressOption.AutomaticStress.value,
        ],
    ],
)
iface.launch(enable_queue=True, prevent_thread_lock=True)
