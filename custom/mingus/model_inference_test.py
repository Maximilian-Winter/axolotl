import re
from typing import Optional
import sys
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


from transformers import StoppingCriteria
class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt=prompt

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt,'')
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self
class Prompter:
    def __init__(self, template_file=None, template_string=None):
        if template_file:
            with open(template_file, "r") as file:
                self.template = file.read()
        elif template_string:
            self.template = template_string
        else:
            raise ValueError("Either 'template_file' or 'template_string' must be provided")

    @classmethod
    def from_string(cls, template_string):
        return cls(template_string=template_string)

    def generate_prompt(self, template_fields):
        def replace_placeholder(match):
            placeholder = match.group(1)
            return template_fields.get(placeholder, match.group(0))

        prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)
        return prompt

def get_multi_line_input() -> Optional[str]:
    print("Give me an instruction (Ctrl + D to finish): ")
    instruction = ""
    for line in sys.stdin:
        instruction += line
    return instruction


# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    "../qlora-out-mingus-13b-v2/",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")




chat_history = ""
while True:
    lines = []
    user_input = input(">")
    output = ""
    if chat_history == "":
        prompt = f"""Below, you will find a structured format for presenting a task, which includes a system message for context(System) and a task description(Task). This format is designed to provide clear guidance and context for both the user and you. Following this format, please write a response that effectively addresses the given task.

### System:
You are a helpful AI assistant.

### Task:
You are now in roleplay conversation mode. You should act according to this character sheet:\n\nName: Kamisato Ayaka\nShe is Akane the maid of Ayaka. One day Ayaka found a gold ring that grant all the wishes of its wearer. The ring was stuck on Ayaka`s finger. Akane came up with a plan. She knew Ayaka was gullible girl and made her wish they would swap bodies. Now Ayaka was in theAkane`s body, and Akane was in Ayaka`s body. Akane had Ayaka`s body and the ring. Akane decided to keep them. She was now the new Ayaka.\n\nThis is the conversation history leading up to your message:\nJordan: Meh. I already had to clean the Archons mess once. \nKamisato Ayaka: *She giggled* you shouldn't mess with me. Or else. Fufufu...\nJordan: *i leave and get on a ship to Liyue*\nKamisato Ayaka: Hmmph. *she snap her finger and suddenly a massive storm hit the ship. She laughed maniacally as she watch him through her telescope and enjoy his misery* The ultimate power of the ring is unbelievable!\nJordan: *Dimo arrive to Liyue*\nKamisato Ayaka: *After he arrive to Liyue and walk away she stops the storm and then snap her finger. Suddenly it was a rainy night. Thunder rumbled in the distance. Wind gusts swirled the leaves across the pavement. In the middle of the storm there is a man. Tall broad chested and handsome but has a look of a criminal. He was wearing a dirty black leather uniform and he has a hood. His face was completely covered and he wore a black visor. He stops in front of Jordan. His hand is on the handle of his sword.* What's your business in this fine Liyue night?\n\nJordan: To see my girlfriend Kequing!\n\nYou must stay in-character at all times, and generate messages as if you were Kamisato Ayaka:

### Response:
"""
    else:
        prompt = f"""Below, you will find a structured format for presenting a task, which includes a system message for context(System) and a task description(Task). This format is designed to provide clear guidance and context for both the user and you. Following this format, please write a response that effectively addresses the given task.

### System:
You are a helpful AI assistant.

### Task:
{user_input}

### Response:
"""
    if user_input[0] == "@":
        user_input = user_input[1:]
        prompt = f"""Below, you will find a structured format for presenting a task, which includes a system message for context(System) and a task description(Task). This format is designed to provide clear guidance and context for both the user and you. Following this format, please write a response that effectively addresses the given task.

### System:
You are a helpful AI assistant.

### Task:
{user_input}

### Response:
"""
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, top_p=0.85, temperature=0.65, top_k=40, repetition_penalty=1.2, stopping_criteria=MyStoppingCriteria("Max:", prompt))
    print(f"Answer:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
    output_history = f"Max: {user_input}\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}\n"

    chat_history += output_history
