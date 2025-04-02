import torch
from input.main import start_recording
from processing.main import generate_response
from output.main import play_text, preload_model
# import logging

if not torch.cuda.is_available():
    print("Failed to start: ROCm torch not found")
    exit(1)
else:
    print("Starting using ROCm torch")


def update_line(text: str):
    print(f"\r\033[K{text}", end="")


if __name__ == "__main__":
    # setup_logging()

    preload_model()

    def handle_io(text_input: str):
        print(f"> {text_input}")
        update_line("< ...")
        response = generate_response(text_input)
        play_text(response)
        update_line(f"< {response}\n\n")

    start_recording(handle_io=handle_io)
