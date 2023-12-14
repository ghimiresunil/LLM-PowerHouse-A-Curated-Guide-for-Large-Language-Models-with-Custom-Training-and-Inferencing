from transformers import AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftConfig

import re
import torch
from typing import Dict
from utils.config_mistral_dpo import DEFAULT_CHAT_TEMPLATE

def get_tokenizer(
    model_args: Dict,
    data_args: Dict
)-> PreTrainedTokenizer:
        
      tokenizer = AutoTokenizer.from_pretrained(
          model_args["BASE_MODEL"]
      )
    
      tokenizer.pad_token_id = tokenizer.eos_token_id
      tokenizer.truncation_side = data_args["TRUNCATION_SIDE"]
      tokenizer.model_max_length = data_args["TOKENIZER_MODEL_MAX_LENGTH"]
      tokenizer.padding_side = data_args["PADDING_SIDE"]
      tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
      return tokenizer

def get_peft_config(
    model_args: Dict
)-> PeftConfig:

    if model_args["USE_PEFT"]:
        return LoraConfig(
            r = model_args["LORA_R"],
            lora_alpha = model_args["LORA_ALPHA"],
            lora_dropout = model_args["LORA_DROPOUT"],
            bias = "none",
            task_type = "CAUSAL_LM",
            target_modules = model_args["LORA_TARGET_MODULES"],
            modules_to_save = model_args["LORA_MODULES_TO_SAVE"]
        )
    
    return None

def get_quantization_config(
    model_args: Dict
)-> BitsAndBytesConfig:

    if model_args["load_in_4bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype=torch.float16,  # For consistency with model weights, we use the same value as `torch_dtype` which is float16 for PEFT models
            bnb_4bit_quant_type=model_args["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=model_args["use_bnb_nested_quant"],
        )
    
    elif model_args["load_in_8bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit = True
        )
        
    else:
        quantization_config = None
    
    return quantization_config