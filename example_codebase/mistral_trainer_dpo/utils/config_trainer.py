model_args = {
    "MODEL_NAME":"mistralai/Mistral-7B-v0.1",
    "MODEL_REVISION":"main",
    "TORCH_DTYPE":"bfloat16",
    "USE_PEFT":False,
    "LORA_R":16,
    "LORA_ALPHA":32,
    "LORA_DROPOUT": 0.05,
    "LORA_TARGET_MODULES": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "LORA_MODULES_TO_SAVE": None,
    "USE_FLASH_ATTENTION_2": True,
    "load_in_8bit":False,
    "load_in_4bit":False,
    "bnb_4bit_quant_type":"nf4",
    "use_bnb_nested_quant":False
}

data_args = {
    "TRUNCATION_SIDE":"left",
    "TOKENIZER_MODEL_MAX_LENGTH":2048,
    "PREPROCESSING_NUM_WORKERS": 4
}

training_args = {
    "seed": 42,
    "gradient_checkpointing": True,
    "beta":0.1,
    "logging_first_step":True,
    "optim":"Adam",
    "remove_unused_columns":True,
    "bf16": True,
    "fp16":False,
    "do_eval":True,
    "evaluation_strategy":"epoch",
    "gradient_accumulation_steps":1,
    "learning_rate":5.0e-7,
    "logging_steps":5,
    "lr_scheduler_type":"cosine",
    "warmup_ratio": 0.1,
    "num_train_epochs":1,
    "output_dir":"model/trainer-medical-mistral",
    "per_device_train_batch_size":4,
    "per_device_eval_batch_size":4,
    "push_to_hub":True,
    "save_strategy":"no",
    "save_total_limit":None,
    "log_level":"info",
    "max_seq_length":2048,
    "overwrite_output_dir": True,
    "tf32":True,
    "hub_model_id": "mistral-7b-bfloat16-sft-full"
}