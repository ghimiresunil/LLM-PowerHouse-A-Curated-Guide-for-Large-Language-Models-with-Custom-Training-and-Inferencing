from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from accelerate import Accelerator


def get_model_tokenizer(base_model, trained_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            trained_model_path
        )

    # If stored in private repo on huggingface

    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #         trained_model,
    #         use_auth_token="PLACE_YOUR_HF_TOKEN"
    #     )

    return tokenizer, model



def generate_inference(input_text, tokenizer, model, padding, accelerator, is_saved_model=True):
    model_input = tokenizer(input_text, 
                                max_length=1024, 
                                padding=padding, 
                                truncation=True,
                                return_tensors="pt")
    if is_saved_model:
        generated_tokens = model.generate(
                        **model_input,
                        **gen_kwargs,
                    )
    
    else:
        generated_tokens = accelerator.unwrap_model(model).generate(
            **model_input,
            **gen_kwargs
        )

        generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
        
        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded_preds

if __name__ == "__main__":

    is_saved_model = True #True if we have saved the unwrap model and want to use it it. And False if we have just completed the training and want to do inference immediately without saving
    base_model_name = "sshleifer/distilbart-cnn-12-6" #base model used during training
    trained_model_path = "" #Enter your trained model path
    padding = "max_length"

    gen_kwargs = {
        "max_length": 1024,
        "num_beams": 3,
    }
    accelerator = Accelerator()
    inp = input("Enter the news context to generate the highlights: ")
    if is_saved_model:
        tokenizer, model = get_model_tokenizer(
            base_model=base_model_name,
            trained_model_path= trained_model_path
        )

        output_highlights = generate_inference(
            input_text= inp,
            tokenizer= tokenizer,
            model= model,
            padding= padding,
            accelerator= accelerator
        )

        print("*"*10)
        print(output_highlights)

