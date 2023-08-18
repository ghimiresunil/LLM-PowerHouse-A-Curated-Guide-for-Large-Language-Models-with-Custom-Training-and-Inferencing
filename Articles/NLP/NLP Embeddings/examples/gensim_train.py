from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import argparse

def train_word2vec(file_path, size, window_size, min_count, epochs, number_of_workers, output_dir):
    with open(file_path,"r") as file:
        text = file.read()
        
    tokens = text.lower().split()
    print(tokens)
    model = Word2Vec(
        sentences = [tokens],
        vector_size = size,
        window = window_size,
        min_count = min_count,
        epochs = epochs,
        workers = number_of_workers
    )
    
    model.wv.save(output_dir+"gensim_model.model")
    
    return output_dir

def predict(model_path):
    model = KeyedVectors.load(model_path)
    print(model.index_to_key) 
    while True:
        word = input("Enter the word")
        embed = model[word]
        
        print("Embeddins: ", embed)
        
        sims = model.most_similar(word, topn=5)
        
        print("Top 5 similar words are: ", sims)
        

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--file_dir",
        type= str,
        required= True,
        help="Enter the .txt file for training skipgram model"
    )
    
    parser.add_argument(
        "--vector_size",
        type= int,
        default=20,
        help="enter the size of the "
    )
    
    parser.add_argument(
        "--window_size",
        type= int,
        default= 2,
        help="enter the desired window size"
    )
    
    parser.add_argument(
        "--min_count",
        type=int,
        default= 1,
        help="enter the desired mininum count for words"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="enter the iteration number"
    )
    
    parser.add_argument(
        "--number_of_workers",
        type= int,
        default=2,
        help = "enter number of desired workers"
    )
    
    parser.add_argument(
        "--output_dir",
        type= str,
        required= True,
        help="enter the output directory to save the model"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    train_word2vec(args.file_dir, args.vector_size, args.window_size, args.min_count, args.epochs, args.number_of_workers, args.output_dir)
    
    # predict("/home/fm-pc-lt-228/Desktop/sid/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/Articles/NLP/NLP Embeddings/data/gensim_model.model")