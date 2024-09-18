import torch
import numpy as np
import sounddevice as sd
from TTS.api import TTS


def load_tts():
    if not torch.cuda.is_available():
        print("Failed to start: ROCm torch not found")
        exit(1)

    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")


def play_text(text: str, tts: TTS):
    wav = tts.tts(text=text, speaker="Sofia Hellen", language="en")

    audio_array = np.array(wav)
    sd.play(audio_array, samplerate=22050)
    sd.wait()
