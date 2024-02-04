from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model_name = model_name

        try:
            self.model = SentenceTransformer(
                self.model_name
            )
        except Exception as e:
            print("Model not found")
    
    def get_embeddings(self, text_list):
        
        return self.model.encode(
            text_list,
            normalize_embeddings = True
        )