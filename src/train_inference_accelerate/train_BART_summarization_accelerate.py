import transformers
from accelerate import Accelerator

from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed
)

from datasets import load_dataset
from datasets import load_metric

import nltk
import math

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import json
import os

nltk.download("punkt", quiet=True)

def get_model_tokenizer(model_name):

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def get_dataset(dataset_name):

    dataset = load_dataset(dataset_name)
    column_names = dataset['train'].column_names
    text_column = column_names[0]
    summary_column = column_names[1]
    return dataset, text_column , summary_column, column_names


def preprocess_function(examples, text_column, summary_column, tokenizer, padding, max_target_length):
    inputs = examples[text_column]
    targets = examples[summary_column]

    model_inputs = tokenizer(inputs,
                           max_length= 1024,
                           padding=padding,
                           truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets,
                           max_length=max_target_length,
                           padding=padding,
                           truncation=True)

    if padding=="max_length":
        labels["input_ids"] = [
            [(l if l!=tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


if __name__ == "__main__":
    accelerator = Accelerator()

    model_name = "sshleifer/distilbart-cnn-12-6" #chose any BART model
    tokenizer, model = get_model_tokenizer(
        model_name= model_name
        )
    
    dataset_name = "cnn_dailymail"
    data, text_column, summary_column, column_names = get_dataset(
        dataset_name= dataset_name
    )

    
    training_args = {
        "padding": "max_length",
        "max_target_length": 1024,
        "label_pad_token_id" : -100,
        "train_batch_size" :2,
        "eval_batch_size" : 2,
        "weight_decay" : 0.01,
        "learning_rate" : 0.0001,
        "gradient_accumulation_steps" : 5,
        "train_epoch" : 5,
        "scheduler_type" : "linear",
        "num_warmup_steps" : 5,
        "output_dir" : "out_model"
    }

    with accelerator.main_process_first():
        processed_datasets = data.map(
              preprocess_function,
              fn_kwargs= {
                  "text_column": text_column,
                  "summary_column": summary_column,
                  "tokenizer":tokenizer,
                  "padding": training_args["padding"],
                  "max_target_length": training_args["max_target_length"]
                  },
              batched=True,
              remove_columns=column_names,
              desc="Running tokenizer on dataset",
          )
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=training_args["label_pad_token_id"]
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args["train_batch_size"],
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=training_args["eval_batch_size"])
    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=training_args["eval_batch_size"])
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr= training_args["learning_rate"])

    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader)
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args["gradient_accumulation_steps"])

    max_train_steps = training_args["train_epoch"] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
            name=training_args["scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=training_args["num_warmup_steps"],
            num_training_steps=max_train_steps,
        )
    metric = load_metric("src/train_inference_accelerate/rouge.py")
    total_batch_size = training_args["train_batch_size"] * accelerator.num_processes * training_args["gradient_accumulation_steps"]

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_val_score = 0
    best_epoch = -1

    for epoch in range(training_args["train_epoch"]):
        model.train()

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)

            loss = outputs.loss
            loss = loss/training_args["gradient_accumulation_steps"]

            accelerator.backward(loss)

            if step % training_args["gradient_accumulation_steps"] == 0 or step == len(train_dataloader)-1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps +=1

            if completed_steps >= max_train_steps:
                break

        model.eval()
        val_max_target_length = training_args["max_target_length"]

        gen_kwargs = {
        "max_length": val_max_target_length,
        "num_beams":4
        }
        if epoch>=0:
            to_dump = []

            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():

                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask = batch["attention_mask"],
                        **gen_kwargs
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )

                    labels = batch["labels"]
                    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                    labels = accelerator.gather(labels).cpu().numpy()

                    # Replace -100 in the labels as we can't decode them.(ignore pad for loss)
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]

                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)

                    decoded_source = tokenizer.batch_decode(batch["input_ids"],skip_special_tokens=True)

                    for source,true,pred in zip(decoded_source,decoded_labels,decoded_preds):
                        sample={"source":source,"true":true,"pred":pred}
                        to_dump.append(sample)


            with open(training_args["output_dir"]+"prediction_epoch{}.json".format(epoch),"w") as f:
                json.dump(to_dump, f,indent=4)

            result = metric.compute(use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

            result = {k: round(v, 4) for k, v in result.items()}

            if np.mean(list(result.values())) > best_val_score:
                best_val_score = np.mean(list(result.values()))
                best_epoch = epoch

            print(f'At epoch {epoch}, the evaluation result is {result}, the best score is {best_val_score} at epoch {best_epoch}')

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        os.makedirs(training_args["output_dir"]+f'/{epoch}/', exist_ok=True)
        unwrapped_model.save_pretrained(training_args["output_dir"]+f'/{epoch}', save_function=accelerator.save)






