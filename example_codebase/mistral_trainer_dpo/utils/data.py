import re
from typing import List, Literal, Optional
from datasets import DatasetDict, load_dataset
from utils.config_mistral_dpo import SYSTEM_MESSAGE

def apply_chat_template(
    example, tokenizer, task: Literal["sft","dpo"], assistant_prefix="<|assistant|>\n"
):
    
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)
    
    if task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            prompt_messages = [
                {
                    "role":"system",
                    "content": SYSTEM_MESSAGE
                },
                {
                    "content": example["prompt"],
                    "role":"user"
                    }
                ]
             
            
            chosen_messages = [
                {
                    "role":"assistant",
                    "content": example["chosen"]
                }
            ]
            
            rejected_messages = [
                {
                    "role":"assistant",
                    "content": example["rejected"]
                }
            ]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
        
        example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
        example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)

    elif task == "sft":
        messages = [
            {
                "role":"system",
                "content": SYSTEM_MESSAGE
            },
            {
                "content": example["prompt"],
                "role":"user"
            },
            {
                "content": example["postive_response"],
                "role":"assistant"
            }
            ]
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
    return example

def generate_dataset(
    train_dataset_path: str,
    test_dataset_path: str
) -> DatasetDict:

    raw_datasets = DatasetDict()
    
    raw_datasets["train"] = load_dataset(
      "csv",
      data_files = train_dataset_path,
      split="train"
    )
    
    raw_datasets["test"] = load_dataset(
      "csv",
      data_files = test_dataset_path,
      split="train"
    )
    
    return raw_datasets