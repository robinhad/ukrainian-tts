import tempfile

import gradio as gr

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import requests
from os.path import exists

MODEL_NAMES = [
    "uk/mai/vits-tts"
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
    release_number = "1.0.0"
    model_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/v{release_number}/model.pth.tar"
    config_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/v{release_number}/config.json"

    model_path = "model.pth.tar"
    config_path = "config.json"

    download(model_link, model_path)
    download(config_link, config_path)

    synthesizer = Synthesizer(
        model_path, config_path, None, None, None,
    )
    MODELS[MODEL_NAME] = synthesizer


def tts(text: str, model_name: str):
    text = text if len(text) < 500 else text[0:500] # mitigate crashes on hf space
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
            default="Введіть, будь ласка, своє речення.",
        ),
        gr.inputs.Radio(
            label="Виберіть TTS модель",
            choices=MODEL_NAMES,
        ),
    ],
    outputs=gr.outputs.Audio(label="Output"),
    title="🐸💬🇺🇦 - Coqui TTS",
    theme="huggingface",
    description="Україномовний🇺🇦 TTS за допомогою Coqui TTS (для апострофа використовуйте символ у трикутних дужках: <'>)",
    article="Якщо вам подобається, підтримайте за посиланням: [SUPPORT LINK](https://send.monobank.ua/jar/48iHq4xAXm),  " +
    "Github: [https://github.com/robinhad/ukrainian-tts](https://github.com/robinhad/ukrainian-tts)",
)
iface.launch()
