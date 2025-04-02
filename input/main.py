import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timezone, timedelta
from queue import Queue
from time import sleep
import typing
from utils import state


def start_recording(
    handle_io: typing.Callable[[str], None], record_timeout=2, phrase_timeout=3
):
    data_queue = Queue()

    # Create recorder
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    # Pull & load model
    model = whisper.load_model("medium.en")

    # I/O handler
    text_input = ""

    # Setup microphone
    microphone = sr.Microphone(sample_rate=16000)

    # Start recording audio in background
    def record_callback(_, audio: sr.AudioData) -> None:
        if state.is_busy:
            print("...")
            return
        data = audio.get_raw_data()
        data_queue.put(data)

    with microphone as source:
        recorder.adjust_for_ambient_noise(source)

    stop_listening = recorder.listen_in_background(
        microphone, record_callback, phrase_time_limit=record_timeout
    )

    # Main loop
    print("Recording started")

    last_talking_at = None

    while True:
        print("loop")
        try:
            now = datetime.now(timezone.utc)
            if not data_queue.empty():
                last_talking_at = now

                # Combine audio data from queue
                audio_data = b"".join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                # Convert audio to text
                result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
                print("Sound result", result)
                text = result["text"].strip()
                if text:
                    print("Found voice -> adding")
                    text_input += f'{" " if text_input else ""}{text}'
            else:
                # On a threshold, handle I/O
                if (
                    text_input
                    and last_talking_at
                    and now - last_talking_at > timedelta(seconds=phrase_timeout)
                ):
                    last_talking_at = None
                    state.is_busy = True
                    handle_io(text_input)
                    print(
                        "Calling callback - done"
                    )
                    text_input = ""
                sleep(0.25)
        except KeyboardInterrupt:
            break

    stop_listening()
    print("Recording stopped")
