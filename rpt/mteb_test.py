import torch
import sys
sys.path.append('/a/home/cc/students/cs/ohadr/netapp/shahar_s/gritlm')
sys.path.append('/a/home/cc/students/cs/ohadr/netapp/shahar_s/gritlm/rpt')
sys.path.append('/Users/shahar.satamkar/Desktop/research/gritlm')
print(sys.path)
from mteb import AmazonReviewsClassification, Banking77Classification
from rpt import GPTNeoXForCausalLMEval
import mteb
from sys import argv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'neox_rpt_model'
hf_model = GPTNeoXForCausalLMEval.from_pretrained(argv[1])
hf_model.to(device=device)

evaluation = mteb.MTEB(tasks=[Banking77Classification(hf_subsets=["en"], batch_size=32, n_experiments=1)])
results = evaluation.run(hf_model, output_folder=f"results/{model_name}")

