from datasets import load_dataset

ds = load_dataset("GritLM/MEDI2")

for split, split_dataset in ds.items():
    split_dataset.to_json(f"medi2-{split}.jsonl")

import json
from tqdm import tqdm

training_data = []

with open('medi2-train.jsonl') as f:
    for line in tqdm(f):
        training_example = json.loads(line)
        query = training_example['query'][1:]
        positives = [pos[1:] for pos in training_example['pos']]
        negatives = [neg[1:] for neg in training_example['neg']]

        training_data.append({
            'query': query,
            'pos': positives[0],
            'neg': negatives[0],
        })

with open('medi2-train-no-instruct.jsonl', 'w') as f:
    for line in training_data:
        f.write(json.dumps(line) + '\n')