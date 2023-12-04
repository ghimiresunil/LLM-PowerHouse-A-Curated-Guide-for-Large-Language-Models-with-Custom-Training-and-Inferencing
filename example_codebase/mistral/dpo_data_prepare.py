import re
import json
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

INTRO = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_FORMAT = (
    """{intro} ### Instruction: {instruction} ### Input: {input} ### Response: """
)


def load_model_tokenizer_for_generate(
    pretrained_model_name_or_path: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path
    )
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    return model, tokenizer


def generate_response(
    instruction: str,
    input_text: str,
    *,
    model,
    tokenizer,
    do_sample: bool = True,
    max_new_tokens: int = 250,
    top_p: float = 1.0,
    top_k: int = 300,
    temperature: float = 0.01,
    **kwargs,
) -> str:
    input_ids = tokenizer(
        INSTRUCTION_FORMAT.format(
            intro=INTRO, instruction=instruction, input=input_text
        ),
        return_tensors="pt",
    ).input_ids
    input_ids = input_ids.to(model.device)
    gen_tokens = model.generate(
        input_ids = input_ids,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        repetition_penalty=1.1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        **kwargs,
    )
    decoded = tokenizer.batch_decode(gen_tokens)[0]

    # The response appears after "### Response:".  The model has been trained to append "### End" at the end.
    m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", decoded, flags=re.DOTALL)

    response = None
    if m:
        response = m.group(1).strip()
    else:
        m = re.search(r"#+\s*Response:\s*(.+)", decoded, flags=re.DOTALL)
        if m:
            response = m.group(1).strip()
        else:
            print(f"Failed to find response in:\n{decoded}")

    return response

def process_item(item, instruction, model, tokenizer):
    input_text = item["input"]
    response = generate_response(
        instruction=instruction,
        input_text=input_text,
        model=model,
        tokenizer=tokenizer,
    )
    return {"input": input_text, "chosen": item['actual_data'], "rejected": response}


if __name__ == "__main__":
    trained_model, trained_tokenizer = load_model_tokenizer_for_generate(
        "meta-llama/Llama-2-7b-chat-hf"
    )
    trained_model.to("cuda")
    eval_prepare_data = open('eval_data/prepare_eval_data.json')
    load_eval_prepare_data = json.load(eval_prepare_data)
    results = []
    instruction="your prompt here...."
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        partial_process_item = partial(process_item, instruction=instruction, model=trained_model, tokenizer=trained_tokenizer)
        futures = [executor.submit(partial_process_item, item) for item in tqdm(load_eval_prepare_data)]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)
            
    with open('data/dpo_final_data.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)  
