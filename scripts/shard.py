import sys

sys.path.append('/a/home/cc/students/cs/ohadr/netapp/shahar_s/gritlm')
sys.path.append('/Users/shahar.satamkar/Desktop/research/gritlm')

from rpt import GPTNeoXForCausalLM

model = GPTNeoXForCausalLM.from_pretrained(
    sys.argv[1],
    torch_dtype="auto",
)
output_path = sys.argv[2]
model.save_pretrained(
    output_path, 
    max_shard_size="5GB",
    safe_serialization=False,
)