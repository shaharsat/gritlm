import sys

sys.path.append('/tmp/shahar/gritlm/')


from rpt.neox_model_torch import GPTNeoXForCausalLM

hf_model = GPTNeoXForCausalLM.from_pretrained('Shahar603/neox-rpt-1')
encode_output = hf_model.encode(['hello'*100, 'world'], 1)
print(encode_output)
