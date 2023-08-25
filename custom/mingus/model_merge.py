from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import sys


model = AutoPeftModelForCausalLM.from_pretrained(
    "../qlora-out-mingus-13b-v2",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
# Save the merged model
merged_model.save_pretrained("./llama-2-13b-mingus-v2",safe_serialization=True)
tokenizer.save_pretrained("./llama-2-13b-mingus-v2")

# push merged model to the hub
# merged_model.push_to_hub("user/repo")
# tokenizer.push_to_hub("user/repo")
