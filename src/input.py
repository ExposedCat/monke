from typing import Callable
import sounddevice as sd
import numpy as np
import whisper
import time
import torch


def load_stt():
    model = whisper.load_model("medium.en")
    return model


def transcribe(raw: bytes, model: whisper.Whisper) -> str:
    audio_np = np.frombuffer(raw, dtype=np.float32)
    result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
    return result["text"].strip()


def on_high_volume(
    callback: Callable[[bytes], None],
    volume_threshold=0.01,  # Threshold of "high volume"
    silence_threshold=0.5,  # Seconds to allow silence between high volume
    samplerate=16000,
    channels=1,
    chunk_duration=0.1,
):
    chunk_size = int(samplerate * chunk_duration)

    buffer = b""
    recording = False
    silence_start_time = 0

    def audio_callback(indata, _frames, _ctime, _status):
        nonlocal buffer, recording, silence_start_time

        audio_bytes = indata.tobytes()
        volume = np.linalg.norm(indata) / np.sqrt(len(indata))

        if volume > volume_threshold:
            recording = True
            buffer += audio_bytes
            silence_start_time = time.time()
        elif recording:
            silence_duration = time.time() - silence_start_time
            if silence_duration > silence_threshold:
                silence_start_time = time.time()
                recording = False
                callback(buffer)
                buffer = b""

    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=channels,
            samplerate=samplerate,
            blocksize=chunk_size,
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        return
