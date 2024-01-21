import os
import torch
from time import perf_counter
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def create_prompt(sample):
    """
    This will format our question into the prompt format used by mistral-7B-instruct
    """
    bos_token = "<s>"
    original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM."
    response = sample.replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + original_system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + input
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + response
    full_prompt += eos_token

    return full_prompt

def download_vllm_model():
    MODEL_DIR = '/model'
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download('mistralai/Mistral-7B-Instruct-v0.1', local_dir=MODEL_DIR, token="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")
    return MODEL_DIR

def generate_vllm_outputs(instructions, model_dir):
    sampling_params = SamplingParams(temperature=0.75,
            top_p=1,
            max_tokens=8000,
            presence_penalty=1.15,)
    llm = LLM(model=model_dir, dtype=torch.float16)
    prompts = [instruction for instruction in instructions]
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def calculate_vllm_num_of_words(outputs):
    num_of_words = 0
    for output in outputs:
        generated_text = output.outputs[0].text
        num_of_words += len(generated_text.split("Generated text:")[0].split(" "))

    return num_of_words

def calculate_throughput(num_of_words, total_time_taken):
    throughput = num_of_words / total_time_taken
    return throughput

def prompt_latency(num_of_words, time_taken_for_a_query):
    latency = num_of_words / time_taken_for_a_query
    return latency

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_cache=False
        )
    model = model.to(dtype=torch.float16, device='cuda')
    model.to("cuda")
    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def generate_llm_response(prompt, model, tokenizer):
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')
    generated_ids = model.generate(**model_inputs, max_new_tokens=8000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.batch_decode(generated_ids)
    return decoded_output[0].replace(prompt, "")

def calculate_llm_num_words(instructions, model, tokenizer):
    num_of_words = 0
    for instruction in instructions:
        output = generate_llm_response(instruction, model, tokenizer)
        num_of_words += len(output.split(" "))
    return num_of_words

if __name__ == '__main__':
    MODEL_DIR = download_vllm_model()
    default_model_name = 'mistralai/Mistral-7B-v0.1'
    instructions = [
        "Elaborate on the cultural heritage of Nepal.",
        "How did the Industrial Revolution impact European societies?",
        "Provide a concise overview of the theory of relativity by Albert Einstein.",
        "Explain the principles behind blockchain technology.",
        "Who were the key figures in the Renaissance and their contributions to art and science?",
        "Describe the process of photosynthesis and its significance in ecosystems.",
        "What are the main features of the Great Barrier Reef and its ecological importance?",
        "Explore the origins and development of jazz music in the United States.",
        "Give a brief history of the internet and its transformative effects on communication.",
        "What are the major causes and consequences of climate change?",
        "Who was Ada Lovelace, and what role did she play in the development of computer programming?",
        "Examine the impact of the Silk Road on cultural exchange between East and West.",
        "What is dark matter, and why is it important in our understanding of the universe?",
        "Explore the history and significance of the Rosetta Stone in deciphering ancient languages."
    ]
    
    # Measure time taken for vLLM generation
    start_time_vllm = perf_counter()
    vllm_outputs = generate_vllm_outputs(instructions, MODEL_DIR)
    end_time_vllm = perf_counter()
    total_time_taken_for_generation_vllm = end_time_vllm - start_time_vllm

    # Calculate time taken for a single query for vLLM
    time_taken_for_a_query_vllm = total_time_taken_for_generation_vllm / len(instructions)

    # Print vLLM results
    vllm_num_of_words = calculate_vllm_num_of_words(vllm_outputs)
    throughput_vllm = calculate_throughput(vllm_num_of_words, total_time_taken_for_generation_vllm)
    vllm_prompt_latency = prompt_latency(vllm_num_of_words, time_taken_for_a_query_vllm)

    print("Number of words/tokens generated by vLLM: ", vllm_num_of_words)
    print("Throughput with vLLM: ", throughput_vllm)
    print("Latency for a prompt with vLLM: ", vllm_prompt_latency)
    print("Total time taken for vLLM generation: ", total_time_taken_for_generation_vllm)
    print("Time taken for a single query with vLLM: ", time_taken_for_a_query_vllm)

    # Measure time taken for LLM generation
    model = load_model(default_model_name)
    tokenizer = load_tokenizer(default_model_name)

    start_time_llm = perf_counter()
    llm_num_of_words = calculate_llm_num_words(instructions, model, tokenizer)
    end_time_llm = perf_counter()
    total_time_taken_for_generation_llm = end_time_llm - start_time_llm

    # Calculate time taken for a single query for LLM
    time_taken_for_a_query_llm = total_time_taken_for_generation_llm / len(instructions)

    # Print LLM results
    throughput_llm = calculate_throughput(llm_num_of_words, total_time_taken_for_generation_llm)
    llm_prompt_latency = prompt_latency(llm_num_of_words, time_taken_for_a_query_llm)

    print("\nNumber of words/tokens generated by LLM: ", llm_num_of_words)
    print("Throughput with LLM: ", throughput_llm)
    print("Latency for a prompt with LLM: ", llm_prompt_latency)
    print("Total time taken for LLM generation: ", total_time_taken_for_generation_llm)
    print("Time taken for a single query with LLM: ", time_taken_for_a_query_llm)