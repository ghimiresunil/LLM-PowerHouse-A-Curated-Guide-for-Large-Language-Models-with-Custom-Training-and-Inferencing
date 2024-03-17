from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import MODEL_NAME
from typing import Dict, Any, Tuple

class DecodingInference:
    """A class for generating text using different decoding strategies."""
    
    def __init__(self) -> None:
        """Initialize the tokenizer and model."""
        self.tokenizer, self.model = self.load_model()

    def load_model(self) -> Tuple[GPT2Tokenizer, GPT2LMHeadModel]:
        """Load the GPT2 tokenizer and model."""
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        return tokenizer, model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text based on the prompt and decoding strategy."""
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
        history_ids = self.model.generate(input_ids, **kwargs)
        output = self.tokenizer.decode(history_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return output
    
    def generation_using_different_strategies(self, prompt: str, strategy_mode: str = "greedy_search") -> str:
        """Generate text using different decoding strategies."""
        strategies: Dict[str, Dict[str, Any]] = {
            "greedy_search": {},
            "beam_search": {
                "num_beams": 5,
                "early_stopping": True,
                "no_repeat_ngram_size": 2,
                "num_return_sequences": 5,
                "max_length": 50
            },
            "random": {
                "temperature": 0.7,
                "do_sample": True,
                "top_k": 0,
                "max_length": 50
            },
            "top_k": {
                "temperature": 0.7,
                "do_sample": True,
                "top_k": 50,
                "max_length": 50
            },
            "top_p": {
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "max_length": 50
            }
        }
        
        kwargs = strategies.get(strategy_mode.lower())
        if kwargs is None:
            raise ValueError("Please provide a valid decoding strategy: greedy_search, beam_search, random, top_k, top_p")
        
        return self.generate(prompt=prompt, **kwargs)

if __name__ == "__main__":
    decoding_inference = DecodingInference()
    
    while True:
        print("Enter 'exit' to quit.")
        decoding_mode = input("Enter the decoding strategy you want to use (greedy_search, beam_search, random, top_k, top_p): ").lower()
        
        if decoding_mode == "exit":
            print("Goodbye!")
            break
        
        prompt = input("Enter your query: ")
        
        response = decoding_inference.generation_using_different_strategies(
            prompt=prompt,
            strategy_mode=decoding_mode
        )
        
        print("Response:", response)
