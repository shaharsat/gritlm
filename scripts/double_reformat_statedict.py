import sys
import torch

sd_path = sys.argv[1]

prefix = 'model.gpt_neox.layers.'

sd = torch.load(sd_path)
# Check if already reformatted by checking if first key has model. prefix
if not list(sd.keys())[0].startswith(prefix):
    print('SD seems already reformatted: ', sd.keys())
    sys.exit(0)
# Remove model i.e. model.h.1 -> h.1
sd = {'layers.blocks.' + k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
torch.save(sd, sd_path)