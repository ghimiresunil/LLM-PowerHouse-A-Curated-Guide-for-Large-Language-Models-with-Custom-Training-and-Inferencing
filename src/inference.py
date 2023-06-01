import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
    Trainer,
    GPT2TokenizerFast,
)

INTRO = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_FORMAT = (
    """{intro} ### Instruction: {instruction} ### Input: {input} ### Response: """
)


def load_model_tokenizer_for_generate(
    pretrained_model_name_or_path: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, padding_side="left"
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
    max_new_tokens: int = 128,
    top_p: float = 0.92,
    top_k: int = 0,
    **kwargs,
) -> str:
    input_ids = tokenizer(
        INSTRUCTION_FORMAT.format(
            intro=INTRO, instruction=instruction, input=input_text
        ),
        return_tensors="pt",
    ).input_ids

    gen_tokens = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
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
        # The model might not generate the "### End" sequence before reaching the max tokens.  In this case, return
        # everything after "### Response:".
        m = re.search(r"#+\s*Response:\s*(.+)", decoded, flags=re.DOTALL)
        if m:
            response = m.group(1).strip()
        else:
            print(f"Failed to find response in:\n{decoded}")

    return response


if __name__ == "__main__":
    trained_model, trained_tokenizer = load_model_tokenizer_for_generate(
        "output_peft_lora"
    )
    response = generate_response(
        instruction="You are an interviewing agent for Fusemachines, an AI services company, conduct an interview with a candidate who is being evaluated for their skills and knowledge in the technical domain of machine learning and data science based on topic: Introduction",
        input_text="In supervised learning, we have a labeled dataset, which means we have both input features and output labels. The goal is to learn a mapping from the input features to the output labels so that it can make accurate predictions on unseen data.",
        model=trained_model,
        tokenizer=trained_tokenizer,
    )
    print(response)
