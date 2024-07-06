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

values = tokenizer.decode([29,    84,  2730,    93,  4537, 49651,   187, 35012,   253,  1650,
           323,   253,  1563,  4836,    27, 10300,   247,  2278,  2505,   432,
          7001,   251,   285,   697, 33487,   313, 26597,  2762,   390,  4016,
           481, 48533,  3662,   346,  5088,     3,   604,  1677,  6197,   285,
           697, 33487,  3761,    13,  5010,  6635,  3662,   346,  5653,  3446,
           329,  2278,   310,  2783,  2762,   604,   253, 37317,   310, 10048,
           342,   253,  1885,    15, 17501,    13,   352,   310,  2783,  4016,
            15,   187,    29,    93, 24224, 49651,   187, 18658,    27, 26733,
           272,   752,   253,  7085,   556,  4592,   281,   253,  9447,  4809,
           273,  4980,  2448,  4466,  3736,   831,  1984,   310,   247,   298,
         15912,    13,   973,  9125,  8813,   273,   253,  2969,   958,   326,
           359,   452,  2489,   594,  9106,  7106,   327,   776,  2060, 33238,
            52,   326,   359,   452,  4336, 12841,    13,   285,  1014, 13031,
           264,    13,   776,  3367,   285,  1345, 19715,    15,   380,  2457,
          2380,   285, 17612,   273,  6911,   416,  4940,   434,  1682, 24451,
            13,   418, 16569,  2637, 16341,  6651,  1852,    18,    15,   831,
           310,   247,  1270,   673,   323,  3780,   665,   310, 20538, 15172,
           342,   253,  1766,   263,   763,  1375,   273,  2448,  3420,    13,
          8672,   285, 25200,   275,  2087,    13,   281,  2590,   616,  9851,
           323,   253,  6832,  7881,   326,  7027,  1078,   441,   417,   760,
           347,  7108,    13,   533,   347,  1966, 14965,    15,   329,  1270,
          1984,  2490,  3130, 15752,    27, 32725])


print(values)