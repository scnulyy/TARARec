from safetensors.torch import load_file
import torch
import os

path = os.path.join('lora-alpaca', "adapter_model.safetensors")
model = load_file(path)
torch.save(model, 'adapter_model.bin')