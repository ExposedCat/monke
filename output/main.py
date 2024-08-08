import torch
import os
from TTS.api import TTS
from playsound import playsound

tts = None


def preload_model():
    global tts
    if not torch.cuda.is_available():
        print("Failed to start: ROCm torch not found")
        exit(1)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")


def play_text(text: str):
    output = "/tmp/talking-monkey-output.mp3"
    tts.tts_to_file(
        text=text, speaker_wav="sample.mp3", language="en", file_path=output
    )
    playsound(output)
    os.remove(output)
