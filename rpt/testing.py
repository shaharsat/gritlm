"""
import sys
import torch

sys.path.append('/tmp/shahar/gritlm/')
from rpt.neox_model_torch import GPTNeoXForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_model = GPTNeoXForCausalLM.from_pretrained('Shahar603/neox-rpt-1', torch_dtype=torch.bfloat16).to(device)
encode_output = hf_model.encode(['hello'*100, 'world'], 1)
print(encode_output)
"""
import torch
from transformers import AutoTokenizer

from rpt.neox_model_torch import GPTNeoXForCausalLM

#hf_model = GPTNeoXForCausalLM.from_pretrained('Shahar603/neox-rpt-1-bf16')
#encode_output = hf_model.encode(['hello'*100, 'world'], 1)
#print(encode_output)

tokenizer = AutoTokenizer.from_pretrained(
    'EleutherAI/gpt-neox-20b',
    padding_side="right",
)

values = tokenizer.decode([   34,  3416,  6306,   327,   247,  1029, 28900,   327,   581,  1791,
          2819,   689,   247,  8281,    15])


print(values)