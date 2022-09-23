from io import BytesIO
import requests
from os.path import exists
from TTS.utils.synthesizer import Synthesizer
from enum import Enum
from .formatter import preprocess_text
from torch import no_grad

class Voices(Enum):
    Olena = "olena"
    Mykyta = "mykyta"
    Lada = "lada"
    Dmytro = "dmytro"
    Olga = "olga"


class StressOption(Enum):
    Dictionary = "dictionary"
    Model = "model"


class TTS:
    def __init__(self, cache_folder=None) -> None:
        self.__setup_cache(cache_folder)


    def tts(self, text: str, voice: str, stress: str, output_fp=BytesIO()):
        autostress_with_model = (
            True if stress == StressOption.Model.value else False
        )

        if voice not in [option.value for option in Voices]:
            raise ValueError("Invalid value for voice selected! Please use one of the following values: {', '.join([option.value for option in Voices])}.")

        text = preprocess_text(text, autostress_with_model)

        with no_grad():
            wavs = self.synthesizer.tts(text, speaker_name=voice)
            self.synthesizer.save_wav(wavs, output_fp)

        output_fp.seek(0)

        return output_fp


    def __setup_cache(self, cache_folder=None):
        print("downloading uk/mykyta/vits-tts")
        release_number = "v3.0.0"
        model_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/model-inference.pth"
        config_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/config.json"
        speakers_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/speakers.pth"

        model_path = "model.pth"
        config_path = "config.json"
        speakers_path = "speakers.pth"

        self.__download(model_link, model_path)
        self.__download(config_link, config_path)
        self.__download(speakers_link, speakers_path)

        self.synthesizer = Synthesizer(
            model_path,
            config_path,
            speakers_path,
            None,
            None,
        )

        if self.synthesizer is None:
            raise NameError("model not found")


    def __download(self, url, file_name):
        if not exists(file_name):
            print(f"Downloading {file_name}")
            r = requests.get(url, allow_redirects=True)
            with open(file_name, "wb") as file:
                file.write(r.content)
        else:
            print(f"Found {file_name}. Skipping download...")


