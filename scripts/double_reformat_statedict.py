import sys
import torch

sd_path = sys.argv[1]

model_prefix = 'model.'

sd = torch.load(sd_path)

new_sd = {}
for k, v in sd.items():
    if k.startswith(model_prefix):
        new_value = k[len(model_prefix):]
    else:
        new_value = k

    new_sd[new_value] = v

torch.save(new_sd, sd_path)