import torch

if torch.cuda.is_available():
    print("A GPU is available !!")
print("A GPU is NOT available...")
