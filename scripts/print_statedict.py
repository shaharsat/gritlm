import sys
import torch

sd_path = sys.argv[1]

prefix = 'model.gpt_neox.'
layer_prefix = 'model.gpt_neox.layers.'

sd = torch.load(sd_path)

print(sd.keys())