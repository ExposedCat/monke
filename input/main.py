import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep

def get_input_devices():
    return list((index, name) for (index, name) in enumerate(sr.Microphone.list_microphone_names()))

def start_recording(handle_io, mic_index=0, record_timeout=2, phrase_timeout=3):
    data_queue = Queue()

    # Create recorder
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    # Setup microphone
    microphone = sr.Microphone(sample_rate=16000, device_index=mic_index)
    with microphone:
        recorder.adjust_for_ambient_noise(microphone)

    # Pull & load model
    model = whisper.load_model("medium.en")

    # I/O handler
    text_input = ''
    def handle():
        nonlocal text_input
        handle_io(text_input)
        text_input = ''

    # Start recording audio in background
    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(microphone, record_callback, phrase_time_limit=record_timeout)

    # Main loop
    print("Recording started")

    last_talking_at = None

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                last_talking_at = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Convert audio to text
                result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                if text:
                  text_input += f'{" " if text_input else ""}{text}'
            else:
                # On a threshold, handle I/O
                if text_input and last_talking_at and now - last_talking_at > timedelta(seconds=phrase_timeout):
                  last_talking_at = None
                  handle()
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("Recording stopped")