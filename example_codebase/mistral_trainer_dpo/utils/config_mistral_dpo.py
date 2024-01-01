model_args = {
    "MODEL_NAME":"ghimiresunil/mistral-7b-bfloat16-trainer-full",
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
    "PADDING_SIDE":"left",
    "TOKENIZER_MODEL_MAX_LENGTH":2048,
    "TRAIN_FILE_PATH":"dpo_data/train_data.csv",
    "TEST_FILE_PATH":"dpo_data/test_data.csv",
    "PREPROCESSING_NUM_WORKERS": 4
}

training_args = {
    "seed": 42,
    "gradient_checkpointing": True,
    "beta":0.1,
    "logging_first_step":True,
    "max_prompt_length":1024,
    "max_length":2048,
    "optim":"rmsprop",
    "remove_unused_columns":False,
    "bf16": True,
    "fp16":False,
    "do_eval":True,
    "evaluation_strategy":"epoch",
    "gradient_accumulation_steps":1,
    "hub_model_id":"mistral-7b-bfloat16-dpo-full",
    "learning_rate":5.0e-7,
    "logging_strategy":"epoch",
    "lr_scheduler_type":"linear",
    "num_train_epochs":3,
    "output_dir":"dpo-medical-mistral",
    "per_device_train_batch_size":4,
    "per_device_eval_batch_size":4,
    "push_to_hub":True,
    "save_strategy":"epoch",
    "save_total_limit":2,
    "warmup_ratio":0.1,
    "log_level":"info"
}

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

SYSTEM_MESSAGE = """
INSTRUCTIONS:

If you are a doctor, please answer the medical questions based on the patient's description.
"""