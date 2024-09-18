import threading
from input import load_stt, on_high_volume, transcribe
from output import load_tts, play_text
from processor import generate_response, load_llm

print("Loading STT..")
stt = load_stt()

print("Loading LLM..")
load_llm()

print("Loading TTS..")
tts = load_tts()


def high_volume_callback(raw: bytes):
    print(f"Detected speech, transcribing..")
    transcription = transcribe(raw, stt)
    if len(transcription) > 0:
        print(f"Transcription: '{transcription}'")
        print(f"Generating response..")
        response = generate_response(transcription)
        print(f"Response: '{response}'")
        print(f"Playing response..")
        play_text(response, tts)
    else:
        print("No speech detected in sound")


input_thread = threading.Thread(
    target=on_high_volume, args=(high_volume_callback, 0.002, 1)
)

print("Listening for input..")
input_thread.start()
