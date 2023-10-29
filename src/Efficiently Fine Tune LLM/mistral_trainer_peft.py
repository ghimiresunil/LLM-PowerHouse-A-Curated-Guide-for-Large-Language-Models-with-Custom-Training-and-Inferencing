import torch
import torch.nn as nn
import numpy as np
from functools import partial
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from data_loader import load_dataset_from_file

RESPONSE_KEY = " ### Response:"
DEFAULT_INPUT_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 2e-5
EPOCHS = 10

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

seed = 42

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)
        response_token_ids = response_token_ids[2:5]
        labels = batch["labels"].clone()
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                if np.array_equal(
                    response_token_ids,
                    batch["labels"][i, idx : idx + len(response_token_ids)],
                ):
                    response_token_ids_start_idx = idx
                    break
            if response_token_ids_start_idx is None:
                breakpoint()
                raise RuntimeError("Could not find response key token IDs")
            response_token_ids_end_idx = response_token_ids_start_idx + len(
                response_token_ids
            )
            labels[i, :response_token_ids_end_idx] = -100
        batch["labels"] = labels
        return batch

def preprocess_batch(batch, tokenizer: AutoTokenizer, max_length: int):
    """Preprocess a batch of inputs for the language model."""

    batch["input_ids"] = tokenizer(batch["text"], max_length=max_length, truncation=True).input_ids
    return batch

def load_training_dataset(data):
    data = data.filter(lambda rec: not rec["text"].strip().startswith(" ### Response:"))

    def _func(rec):
        rec["text"] += "\n\n### End"
        return rec

    data = data.map(_func)
    return data

def preprocess_dataset(dataset, tokenizer: AutoTokenizer, max_length: int, seed=seed):
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )

    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)

    dataset = dataset.shuffle(seed=seed)

    return dataset

def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable Parameters: {trainable_params} || All Parameters: {all_param} || Trainable %: {100 * trainable_params / all_param}"
    )
    
def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    gradient_checkpointing: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True
        )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
    model = get_peft_model(model, peft_config)  
    print_trainable_parameters(model)
    return model

def get_tokenizer_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    gradient_checkpointing: bool = False
):
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    return tokenizer, model

def train(
    local_output_dir,
    epochs,
    per_device_train_batch_size,
    lr,
    seed,
    test_size=200,
):
    set_seed(seed)
    tokenizer, model = get_tokenizer_model()
    model_conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model_conf, length_setting, None)
        if max_length:
            break
    if not max_length:
        max_length = 1024
    
    data = load_dataset_from_file("dataset/final_df.csv")
    dataset = load_training_dataset(data)
    processed_dataset = preprocess_dataset(dataset, tokenizer=tokenizer, max_length=max_length, seed=seed)
    split_dataset = processed_dataset.train_test_split(test_size=200, seed=seed)
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        bf16=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        disable_tqdm=True,
        remove_unused_columns=True,
        weight_decay=0.01,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
    )
    print('Trainer Initialize')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_model(output_dir=local_output_dir)
    torch.cuda.empty_cache()

def main(**kwargs):
    train(**kwargs)
    
if __name__ == "__main__":
    try:
        med_tune = {
            "local_output_dir": "output/",
            "epochs": EPOCHS,
            "per_device_train_batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "seed": seed,
            "test_size": 200,
        }
        main(**med_tune)
    except Exception:
        raise
