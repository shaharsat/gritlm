from rpt.neox_model_torch import GPTNeoXForCausalLM

hf_model = GPTNeoXForCausalLM.from_pretrained('Shahar603/neox-rpt-1')
hf_model.config.input_length = 2048
encode_output = hf_model.encode(['hello'*100, 'world'], 1)
print(encode_output)
