import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")


def load_jsonl(filename):
    with open(filename, 'r') as file:
        return [json.loads(line) for line in file]

dataset = load_jsonl('mingus_dataset_v6.jsonl')


count_deleted = 0

new_data = []
for date in dataset:
    length = tokenizer(date['text'], return_tensors="pt").input_ids.cuda().shape[1]
    if length <= 4096:
        new_data.append(date)
    else:
        count_deleted += 1

print(count_deleted)


def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

save_jsonl(new_data, "mingus_dataset_v6-cleaned.jsonl")
