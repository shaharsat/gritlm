import sys
import torch

sd_path = sys.argv[1]

prefix = 'model.gpt_neox.'
layer_prefix = 'model.gpt_neox.layers.'

sd = torch.load(sd_path)
# Check if already reformatted by checking if first key has model. prefix
if not list(sd.keys())[0].startswith(prefix):
    print('SD seems already reformatted: ', sd.keys())
    sys.exit(0)
# Remove model i.e. model.h.1 -> h.1

new_sd = {}
for k, v in sd.items():
    if k.startswith(layer_prefix):
        new_value = 'layers.blocks.' + k[len(layer_prefix):]
    elif k.startswith(prefix):
        new_value = k[len(prefix):]
    else:
        new_value = k

    new_sd[new_value] = v

torch.save(new_sd, sd_path)