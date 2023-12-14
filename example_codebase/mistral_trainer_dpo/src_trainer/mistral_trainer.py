import torch
import numpy as np
from functools import partial
from utils.config_trainer import model_args, data_args, training_args
from utils.models import get_peft_config, get_quantization_config
from transformers import (
    set_seed,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from accelerate import Accelerator
from src_trainer.data_loader import load_dataset_from_file

RESPONSE_KEY = " ### Response:"
DEFAULT_INPUT_MODEL = model_args["MODEL_NAME"]
peft_config = get_peft_config(model_args)

model_kwargs = dict(
        revision = model_args["MODEL_REVISION"],
        use_flash_attention_2=model_args["USE_FLASH_ATTENTION_2"],
        use_cache=False if training_args["gradient_checkpointing"] else True,
        quantization_config=get_quantization_config(model_args)
      )

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

    # dataset = dataset.map(_func)
    data = data.map(_func)
    return data

def preprocess_dataset(dataset, tokenizer: AutoTokenizer, max_length: int, seed=training_args["seed"]):
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
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = data_args["TRUNCATION_SIDE"]
    tokenizer.model_max_length = data_args["TOKENIZER_MODEL_MAX_LENGTH"]
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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        **model_kwargs
        )
    model = model.to(device)
    return model

def get_tokenizer_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL):
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path)
    return tokenizer, model


def train():
    set_seed(training_args["seed"])
    accelerator = Accelerator()
    tokenizer, model = get_tokenizer_model()
    max_length = training_args["max_seq_length"]
    data=load_dataset_from_file("dataset/final_df.csv")
    dataset = load_training_dataset(data)
    processed_dataset = preprocess_dataset(dataset, tokenizer=tokenizer, max_length=max_length, seed=training_args["seed"])
    split_dataset = processed_dataset.train_test_split(test_size=20, seed=training_args["seed"])
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )
    training_args_hf = TrainingArguments(
        output_dir=training_args["output_dir"],
        seed = training_args["seed"],
        do_eval = training_args["do_eval"],
        per_device_train_batch_size=training_args["per_device_train_batch_size"],
        per_device_eval_batch_size = training_args["per_device_eval_batch_size"],
        bf16 = training_args["bf16"],
        fp16 = training_args["fp16"],
        learning_rate=training_args["learning_rate"],
        num_train_epochs=training_args["num_train_epochs"],
        lr_scheduler_type = training_args["lr_scheduler_type"],
        gradient_checkpointing=training_args["gradient_checkpointing"],
        save_strategy="epoch",
        save_steps=200,
        save_total_limit=training_args["save_total_limit"],
        load_best_model_at_end=True,
        disable_tqdm=True,
        remove_unused_columns=training_args["remove_unused_columns"],
        push_to_hub=training_args["push_to_hub"],
        # weight_decay=0.01,
        gradient_accumulation_steps = training_args["gradient_accumulation_steps"],
        warmup_ratio= training_args["warmup_ratio"],
        evaluation_strategy = training_args["evaluation_strategy"],
        logging_strategy="epoch",
        overwrite_output_dir = training_args["overwrite_output_dir"],
        tf32 = training_args["tf32"],
    )
    print('Trainer Initialize')
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]
    trainer = Trainer(
        model=model,
        args=training_args_hf,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    print("Training")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_data)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    if training_args["do_eval"]:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_data)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(output_dir = training_args["output_dir"])
    
    if accelerator.is_main_process:
        kwargs = {"finetuned_from": model_args["MODEL_NAME"]}
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args["output_dir"])

        if training_args["push_to_hub"] is True:
            print("Pushing to hub...")
            trainer.push_to_hub()

    accelerator.wait_for_everyone()
        
def main():
    train()
    
if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise