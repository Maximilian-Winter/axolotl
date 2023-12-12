import json


def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line) for line in file]

data = load_jsonl('pippa_10k_uncensored.jsonl')
loaded_data = load_jsonl('flan10k_uncensored.jsonl')

data.extend(loaded_data)

def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

save_jsonl(data, "mingus_dataset_v6.jsonl")
