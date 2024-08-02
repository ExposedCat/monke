import torch

if not torch.cuda.is_available():
  print('ROCm NOT found')
else:
  print('ROCm found')
