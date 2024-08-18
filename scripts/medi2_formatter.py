from datasets import load_dataset

#ds = load_dataset("GritLM/MEDI2")

#for split, split_dataset in ds.items():
#    split_dataset.to_json(f"medi2-{split}.jsonl")

import json
from tqdm import tqdm

training_data = []

with open('medi2-train.jsonl') as f:
    for line in tqdm(f):
        training_example = json.loads(line)
        training_example['query'][0] = ''
        training_example['pos'][0][0] = ''
        training_example['neg'][0][0] = ''

        training_data.append({
            'query': training_example['query'][1],
            'pos': training_example['pos'],
            'neg': training_example['neg'],
        })

with open('medi2-train-no-instruct.jsonl', 'w') as f:
    for line in training_data:
        f.write(json.dumps(line) + '\n')