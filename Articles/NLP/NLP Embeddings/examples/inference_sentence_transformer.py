from sentence_transformers import SentenceTransformer
import scipy


class InferenceSentenceTransformers:
    
    def __init__(self, model_name):
        
        self.model = SentenceTransformer(
            model_name
        )
    
    def get_embeddings(self, query):
        
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )
    
    def calculate_cosine_similarity(self, embeds_1, embeds_2):
        
        return scipy.spatial.distance.cdist(embeds_1, embeds_2, "cosine")[0]

if __name__ == "__main__":
    
    infer = InferenceSentenceTransformers(
        model_name= "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    query_1 = input("Enter first query")
    
    embed_1 = infer.get_embeddings(
        query=query_1
    )
    
    query_2 = input("Enter second query")
    
    embed_2 = infer.get_embeddings(
        query= query_2
    )
    
    res = infer.calculate_cosine_similarity(
        embeds_1= embed_1,
        embeds_2= embed_2
    )
    
    score = 1 - res[0]
    print(score)