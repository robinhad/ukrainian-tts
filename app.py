import tempfile
import gradio as gr
from datetime import datetime
from enum import Enum
from ukrainian_tts.tts import TTS, Stress, Voices
from torch.cuda import is_available
from os import getenv
from data_logger import log_data
from threading import Thread
from queue import Queue
from time import sleep


def check_thread(logging_queue: Queue):
    logging_callback = log_data(
        hf_token=getenv("HF_API_TOKEN"), dataset_name="uk-tts-output", private=True
    )
    while True:
        sleep(60)
        batch = []
        while not logging_queue.empty():
            batch.append(logging_queue.get())

        if len(batch) > 0:
            try:
                logging_callback(batch)
            except:
                print(
                    "Error happened while pushing data to HF. Puttting items back in queue..."
                )
                for item in batch:
                    logging_queue.put(item)


if getenv("HF_API_TOKEN") is not None:
    log_queue = Queue()
    t = Thread(target=check_thread, args=(log_queue,))
    t.start()


class VoiceOption(Enum):
    Tetiana = "Тетяна (жіночий) 👩"
    Mykyta = "Микита (чоловічий) 👨"
    Lada = "Лада (жіночий) 👩"
    Dmytro = "Дмитро (чоловічий) 👨"
    Oleksa = "Олекса (чоловічий) 👨"


print(f"CUDA available? {is_available()}")

ukr_tts = TTS(device="cuda" if is_available() else "cpu")


def tts(text: str, voice: str):
    print("============================")
    print("Original text:", text)
    print("Voice", voice)
    print("Time:", datetime.utcnow())

    voice_mapping = {
        VoiceOption.Tetiana.value: Voices.Tetiana.value,
        VoiceOption.Mykyta.value: Voices.Mykyta.value,
        VoiceOption.Lada.value: Voices.Lada.value,
        VoiceOption.Dmytro.value: Voices.Dmytro.value,
        VoiceOption.Oleksa.value: Voices.Oleksa.value,
    }

    speaker_name = voice_mapping[voice]
    text_limit = 2000
    text = (
        text if len(text) < text_limit else text[0:text_limit]
    )  # mitigate crashes on hf space

    if getenv("HF_API_TOKEN") is not None:
        log_queue.put(
            [text, speaker_name, Stress.Dictionary.value, 1, str(datetime.utcnow())]
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        _, text = ukr_tts.tts(text, speaker_name, Stress.Dictionary.value, fp)
        return fp.name, text


with open("README.md") as file:
    article = file.read()
    article = article[article.find("---\n", 4) + 5 : :]


iface = gr.Interface(
    fn=tts,
    inputs=[
        gr.components.Textbox(
            label="Input",
            value="Введіть, будь ласка, своє речення.",
        ),
        gr.components.Radio(
            label="Голос",
            choices=[option.value for option in VoiceOption],
            value=VoiceOption.Tetiana.value,
        ),
    ],
    outputs=[
        gr.components.Audio(label="Output"),
        gr.components.Textbox(label="Наголошений текст"),
    ],
    title="🤖💬🇺🇦 - ESPNET",
    description="Україномовний🇺🇦 TTS за допомогою ESPNET (щоб вручну поставити наголос, використовуйте + перед голосною).",
    article=article,
    examples=[
        [
            "Привіт, як тебе звати?",
            VoiceOption.Tetiana.value,
        ],
        [
            "Введіть, будь ласка, св+оє реч+ення.",
            VoiceOption.Dmytro.value,
        ],
        [
            "Введіть, будь ласка, своє речення.",
            VoiceOption.Oleksa.value,
        ],
        [
            "Введіть, будь ласка, своє речення.",
            VoiceOption.Mykyta.value,
        ],
        [
            "Договір підписано 4 квітня 1949 року.",
            VoiceOption.Lada.value,
        ],
    ],
)
iface.queue()  # for HF specifically
iface.launch()
