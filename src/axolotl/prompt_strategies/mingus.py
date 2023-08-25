from typing import Generator, Union, Tuple

from axolotl.prompt_strategies.alpaca_w_system import OpenOrcaPromptTokenizingStrategy
from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter

from typing import Generator, Tuple, Union

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle


class InstructionWSystemPromptMingusTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str, str]:
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["output"],
            prompt["system"],
        )

    def tokenize_prompt(self, prompt):
        # pylint: disable=duplicate-code
        (
            instruction,
            input,  # pylint: disable=redefined-builtin
            response,
            system,
        ) = self.parse_instruction_fields(prompt)
        user_prompt = next(
            iter(
                self.prompter.build_prompt_w_system(
                    system,
                    instruction,
                    input,
                )
            )
        )
        tokenized_prompt = self._tokenize(user_prompt, add_eos_token=False)
        if not self.train_on_inputs:
            user_prompt_len = len(tokenized_prompt["input_ids"])
            # TODO this could be sped up using numpy array slicing
            tokenized_prompt["labels"] = [-100] * user_prompt_len
        tokenized_res_prompt = self._tokenize(
            response, strip_bos_token=True, add_eos_token=True
        )
        tokenized_prompt["input_ids"] += tokenized_res_prompt["input_ids"]
        tokenized_prompt["attention_mask"] += tokenized_res_prompt["attention_mask"]
        tokenized_prompt["labels"] += tokenized_res_prompt["input_ids"]

        return tokenized_prompt


class MingusWSystemPrompter:
    """
    Alpaca Style Prompter that uses system prompts from the dataset
    """

    system_format: str = "Below, you will find a structured format for presenting a task, which includes a system message for context(System) and a task description(Task). This format is designed to provide clear guidance and context for both the user and you. Following this format, please write a response that effectively addresses the given task.\n\n### System:\n{system}\n\n"
    turn_format: str = "### Task:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    turn_no_input_format: str = "### Task:\n{instruction}\n\n### Response:\n"

    def match_prompt_style(self):
        # pylint: disable=duplicate-code
        self.system_format = "Below, you will find a structured format for presenting a task, which includes a system message for context(System) and a task description(Task). This format is designed to provide clear guidance and context for both the user and you. Following this format, please write a response that effectively addresses the given task.\n\n### System:\n{system}\n\n"
        self.turn_format = "### Task:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        self.turn_no_input_format = "### Task:\n{instruction}\n\n### Response:\n"

    def build_prompt_w_system(
        self,
        system: str,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.

        formatted_sys_prompt = (
            self.system_format.format(system=system)
            if system and self.system_format
            else ""
        )
        if input:
            res = formatted_sys_prompt + self.turn_format.format(
                instruction=instruction, input=input
            )
        else:
            res = formatted_sys_prompt + self.turn_no_input_format.format(
                instruction=instruction
            )
        if output:
            res = f"{res}{output}"
        yield res

def load_mingus(tokenizer, cfg):
    return InstructionWSystemPromptMingusTokenizingStrategy(
        MingusWSystemPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
