import os
import numpy as np
import pandas as pd
from functools import partial
from datasets import load_dataset, Dataset
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

RESPONSE_KEY = " ### Response:"
DEFAULT_INPUT_MODEL = "EleutherAI/gpt-neo-125M"
seed = 42
MAX_LENGTH = 1024


def load_dataset_from_file(dataset_path):
    df = pd.read_csv(dataset_path)
    data = Dataset.from_pandas(df)
    return data


data = load_dataset_from_file("dataset/final_df.csv")


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)
        # print("RTI:",response_token_ids)

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


def preprocess_batch(batch, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
    return tokenizer(batch["text"], max_length=max_length, truncation=True)


def load_training_dataset(training_data_id=data):
    # dataset: Dataset = load_dataset(training_data_id)

    dataset = training_data_id
    # Remove the response key from the text
    dataset = dataset.filter(
        lambda rec: not rec["text"].strip().startswith(" ### Response:")
    )

    def _func(rec):
        rec["text"] += "\n\n### End"
        return rec

    dataset = dataset.map(_func)
    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, use_fast=True
    )
    # print(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing: bool = False
):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True,
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    *,
    gradient_checkpointing: bool = False
):
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing
    )
    return model, tokenizer


def preprocess_dataset(
    tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH, seed=seed
):
    dataset = load_training_dataset()

    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )

    dataset = dataset.shuffle(seed=seed)
    return dataset


def train(
    local_output_dir,
    epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    lr,
    seed,
    test_size=500,
):
    set_seed(seed)
    model, tokenizer = get_model_tokenizer()
    processed_dataset = preprocess_dataset(tokenizer=tokenizer, seed=seed)
    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir=local_output_dir)


def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    try:
        dolly_interview_agent = {
            "local_output_dir": "output/",
            "epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "lr": 0.001,
            "seed": seed,
            "test_size": 500,
        }
        main(**dolly_interview_agent)
    except Exception:
        raise
