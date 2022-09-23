from io import BytesIO
import requests
from os.path import exists, join
from TTS.utils.synthesizer import Synthesizer
from enum import Enum
from .formatter import preprocess_text
from torch import no_grad

class Voices(Enum):
    """List of available voices for the model."""
    Olena = "olena"
    Mykyta = "mykyta"
    Lada = "lada"
    Dmytro = "dmytro"
    Olga = "olga"


class StressOption(Enum):
    """Options how to stress sentence.
    - `dictionary` - performs lookup in dictionary, taking into account grammatical case of a word and its' neighbors
    - `model` - stress using transformer model"""
    Dictionary = "dictionary"
    Model = "model"


class TTS:
    """
    
    """
    def __init__(self, cache_folder=None, use_cuda=False) -> None:
        """
        Class to setup a text-to-speech engine, from download to model creation.  \n
        Downloads or uses files from `cache_folder` directory.  \n
        By default stores in current directory."""
        self.__setup_cache(cache_folder, use_cuda=use_cuda)


    def tts(self, text: str, voice: str, stress: str, output_fp=BytesIO()):
        """
        Run a Text-to-Speech engine and output to `output_fp` BytesIO-like object.
        - `text` - your model input text.
        - `voice` - one of predefined voices from `Voices` enum.
        - `stress` - stress method options, predefined in `StressOption` enum.
        - `output_fp` - file-like object output. Stores in RAM by default.
        """
        autostress_with_model = (
            True if stress == StressOption.Model.value else False
        )

        if voice not in [option.value for option in Voices]:
            raise ValueError(f"Invalid value for voice selected! Please use one of the following values: {', '.join([option.value for option in Voices])}.")

        text = preprocess_text(text, autostress_with_model)

        with no_grad():
            wavs = self.synthesizer.tts(text, speaker_name=voice)
            self.synthesizer.save_wav(wavs, output_fp)

        output_fp.seek(0)

        return output_fp, text


    def __setup_cache(self, cache_folder=None, use_cuda=False):
        """Downloads models and stores them into `cache_folder`. By default stores in current directory."""
        print("downloading uk/mykyta/vits-tts")
        release_number = "v3.0.0"
        model_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/model-inference.pth"
        config_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/config.json"
        speakers_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/speakers.pth"

        if cache_folder is None:
            cache_folder = "."

        model_path = join(cache_folder, "model.pth")
        config_path = join(cache_folder, "config.json")
        speakers_path = join(cache_folder, "speakers.pth")

        self.__download(model_link, model_path)
        self.__download(config_link, config_path)
        self.__download(speakers_link, speakers_path)

        self.synthesizer = Synthesizer(
            model_path,
            config_path,
            speakers_path,
            None,
            None,
            use_cuda=use_cuda
        )

        if self.synthesizer is None:
            raise NameError("Model not found")


    def __download(self, url, file_name):
        """Downloads file from `url` into local `file_name` file."""
        if not exists(file_name):
            print(f"Downloading {file_name}")
            r = requests.get(url, allow_redirects=True)
            with open(file_name, "wb") as file:
                file.write(r.content)
        else:
            print(f"Found {file_name}. Skipping download...")


