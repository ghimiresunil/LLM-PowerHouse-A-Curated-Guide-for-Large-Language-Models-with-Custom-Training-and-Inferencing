from utils.config_mistral_dpo import model_args, data_args, training_args, DEFAULT_CHAT_TEMPLATE, SYSTEM_MESSAGE
from utils.data import apply_chat_template, generate_dataset
from utils.models import get_tokenizer, get_peft_config, get_quantization_config
from trl import DPOTrainer
from transformers import set_seed, TrainingArguments
from accelerate import Accelerator


def train(model_args, data_args, training_args):
    accelerator = Accelerator()
    set_seed(training_args["seed"])
    
    raw_datasets = generate_dataset(
        data_args["TRAIN_FILE_PATH"],
        data_args["TEST_FILE_PATH"]
    )
    # print(raw_datasets)

    column_names = list(raw_datasets["train"].features)
    tokenizer = get_tokenizer(model_args, data_args)
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task":"dpo"},
        num_proc=data_args["PREPROCESSING_NUM_WORKERS"],
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    for split in ["train","test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {
                'text_prompt':"prompt",
                'text_rejected':"rejected", 
                'text_chosen':"chosen"
            }
        )

    model_kwargs = dict(
          use_flash_attention_2=model_args["USE_FLASH_ATTENTION_2"],
          use_cache=False if training_args["gradient_checkpointing"] else True,
          quantization_config=get_quantization_config(model_args)
      )
    
    model = model_args["MODEL_NAME"]
    ref_model = model
    ref_model_kwargs = model_kwargs
         
    if model_args["USE_PEFT"]:
        ref_model = None
        ref_model_kwargs = None

    peft_config = get_peft_config(model_args)

    training_args_hf = TrainingArguments(
        output_dir = training_args["output_dir"],
        seed = training_args["seed"],
        do_eval = training_args["do_eval"],
        evaluation_strategy = training_args["evaluation_strategy"],
        per_device_train_batch_size = training_args["per_device_train_batch_size"],
        per_device_eval_batch_size = training_args["per_device_eval_batch_size"],
        gradient_accumulation_steps = training_args["gradient_accumulation_steps"],
        learning_rate = training_args["learning_rate"],
        num_train_epochs = training_args["num_train_epochs"],
        lr_scheduler_type = training_args["lr_scheduler_type"],
        warmup_ratio = training_args["warmup_ratio"],
        log_level = training_args["log_level"],
        logging_first_step = training_args["logging_first_step"],
        logging_strategy = training_args["logging_strategy"],
        save_strategy = training_args["save_strategy"],
        save_total_limit = training_args["save_total_limit"],
        bf16 = training_args["bf16"],
        fp16 = training_args["fp16"],
        remove_unused_columns = training_args["remove_unused_columns"],
        optim = training_args["optim"],
        gradient_checkpointing = training_args["gradient_checkpointing"],
        push_to_hub = training_args["push_to_hub"],
        hub_model_id = training_args["hub_model_id"]
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args_hf,
        beta=training_args["beta"],
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args["max_length"],
        max_prompt_length=training_args["max_prompt_length"],
        peft_config=peft_config,
    )    
    train_result = dpo_trainer.train()
    metrics = train_result.metrics

    metrics["train_samples"] = len(raw_datasets["train"])
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    print("Training Complete")
                                   
    if training_args["do_eval"]:
        print("*** Evaluate ***")
        metrics = dpo_trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)

    dpo_trainer.save_model(training_args["output_dir"])
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args["MODEL_NAME"]
        }
        dpo_trainer.create_model_card(**kwargs)
        dpo_trainer.model.config.use_cache = True
        dpo_trainer.model.config.save_pretrained(training_args["output_dir"])
        if training_args["push_to_hub"] is True:
            dpo_trainer.push_to_hub()

    print("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()
    print("*** Run complete! ***")
    
if __name__ == "__main__":
    train(model_args, data_args, training_args)    
    
