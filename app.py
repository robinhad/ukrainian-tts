import tempfile
from typing import Optional

import gradio as gr
import numpy as np

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

MODEL_NAMES = [
    "uk/mai/glow-tts"
]
MODELS = {}

manager = ModelManager()

for MODEL_NAME in MODEL_NAMES:
    print(f"downloading {MODEL_NAME}")
    model_path, config_path, model_item = manager.download_model(
        f"tts_models/{MODEL_NAME}")
    vocoder_name: Optional[str] = model_item["default_vocoder"]
    vocoder_path = None
    vocoder_config_path = None
    if vocoder_name is not None:
        vocoder_path, vocoder_config_path, _ = manager.download_model(
            vocoder_name)

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
            label="Pick a TTS Model",
            choices=MODEL_NAMES,
        ),
    ],
    outputs=gr.outputs.Audio(label="Output"),
    title="🐸💬 - Coqui TTS",
    theme="huggingface",
    description="🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production",
    article="more info at https://github.com/coqui-ai/TTS",
)
iface.launch()
