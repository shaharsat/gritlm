import sys
import torch

sd_path = sys.argv[1]

prefix = 'model.gpt_neox.'
layer_prefix = 'model.gpt_neox.layers.'
model_prefix = 'model.'

sd = torch.load(sd_path)

new_sd = {}
for k, v in sd.items():
    if k.startswith(layer_prefix):
        new_value = 'layers.blocks.' + k[len(layer_prefix):]
    elif k.startswith(prefix):
        new_value = k[len(prefix):]
    elif k.startswith(model_prefix):
        new_value = k[len(model_prefix):]
    else:
        new_value = k

    new_sd[new_value] = v

torch.save(new_sd, sd_path)