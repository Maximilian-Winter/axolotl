import json
import random


def get_random_entries_from_jsonl(filename, num_entries=1):
    # Read all entries from the JSONL file
    with open(filename, 'r') as file:
        entries = [json.loads(line) for line in file]

    # Randomly select X entries
    return random.sample(entries, num_entries)


def get_entries_from_multiple_files(filenames, num_entries_per_file=100):
    all_entries = []

    for filename in filenames:
        random_entries = get_random_entries_from_jsonl(filename, num_entries_per_file)
        all_entries.extend(random_entries)

    return all_entries

# Example usage:
filenames = ["flan2k_uncensored.jsonl", "pippa_2k_uncensored.jsonl", "airoboros_2k_uncensored.jsonl"]
data = get_entries_from_multiple_files(filenames, 500)

import random



def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

save_jsonl(data, "mingus_trial.jsonl")
