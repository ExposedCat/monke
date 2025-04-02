import torch
# import os
from TTS.api import TTS
# from playsound import playsound
import sounddevice as sd
import numpy as np
from utils import state
import re

tts = None


def preload_model():
	global tts
	if not torch.cuda.is_available():
		print("Failed to start: ROCm torch not found")
		exit(1)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")


def play_text(text: str):
	if tts is None:
		print("TTS model not loaded")
		return

	print("Playing sound")

	audio = tts.tts(
		text=text, speaker_wav="sasha.mp4.mp3", language="ru"
	)
	audio = np.array(audio, dtype=np.float32)
	audio /= np.max(np.abs(audio))  # Normalize to [-1,1]
	sample_rate = 24000

	sd.play(audio, samplerate=sample_rate)
	sd.wait()

	print("Playing sound - done")
	state.is_busy = False
