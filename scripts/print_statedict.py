import sys
import torch

sd_path = sys.argv[1]

sd = torch.load(sd_path)

print(sd.keys())