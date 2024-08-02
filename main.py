import torch
from input.main import get_input_devices, start_recording
from processing.main import generate_response
from output.main import play_text

if not torch.cuda.is_available():
  print('Failed to start: ROCm torch not found')
  exit(1)
else:
  print('Starting using ROCm torch')

if __name__ == "__main__":
  print('Choose an input device:')
  devices = get_input_devices()
  for index, name in devices:
    print(f'[{index}] {name}')
  device_index = int(input('> '))

  def update_line(text: str):
    print(f"\r\033[K{text}", end='')

  def handle_io(text_input: str):
    print(f'> {text_input}')
    update_line('< ...')
    response = generate_response(text_input)
    play_text(response)
    update_line(f'< {response}\n\n')

  start_recording(handle_io=handle_io, mic_index=device_index)