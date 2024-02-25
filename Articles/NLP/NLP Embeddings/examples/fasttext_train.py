import os
import gdown
import gensim
import pandas as pd
import warnings
from typing import List, Dict

def download_dataset(data_dir: str):
    """
    Download the Wikipedia Attack Comments dataset if not already present.
    
    Args:
        data_dir (str): Directory where the dataset will be stored.
    """
    filename = os.path.join(data_dir, 'fasttext_data.tsv')
    
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(filename):
        url = 'https://drive.google.com/uc?id=1t1kASkUCi7rQ2GQcZnlTHqjYsSPIVJLH'
        print('Downloading Wikipedia Attack Comments dataset...')
        gdown.download(url, filename, quiet=False)
        print('DONE.')

def preprocess_comments(data_dir: str) -> List[List[str]]:
    """
    Preprocess the comments in the dataset.
    
    Args:
        data_dir (str): Directory containing the dataset.
        
    Returns:
        List of preprocessed comment sentences.
    """
    tsv_file_path = os.path.join(data_dir, 'fasttext_data.tsv')
    comments = pd.read_csv(tsv_file_path, sep='\t')
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " ").replace("TAB_TOKEN", " "))
    sentences = [gensim.utils.simple_preprocess(row.comment) for _, row in comments.iterrows()]
    return sentences

def train_fasttext_model(sentences: List[List[str]], model_params: Dict[str, int], data_dir: str):
    """
    Train a FastText model using the provided sentences and parameters.
    
    Args:
        sentences (List[List[str]]): List of preprocessed sentences.
        model_params (Dict[str, int]): Parameters for the FastText model.
        data_dir (str): Directory where the model will be saved.
    """
    model = gensim.models.FastText(sentences=None, **model_params)

    model.build_vocab(
        sentences,
        progress_per=20000
    )

    print('Training the model...')
    for epoch in range(10):
        model.train(sentences, total_examples=len(sentences), epochs=1, report_delay=10.0)
        print(f'Epoch {epoch+1} completed.')
    print('Training completed.')
    model.save(os.path.join(data_dir, 'fasttext.model'))

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    data_dir = '../data/'
    model_params = {
        'window': 10,
        'min_count': 2,
        'workers': 10,
        'sg': 1,
        'hs': 0,
        'negative': 5,
        'sample': 1e-3,
        'word_ngrams': 1,
        'min_n': 3,
        'max_n': 6,
        'bucket': 2000000
    }
    
    download_dataset(data_dir)
    sentences = preprocess_comments(data_dir)
    train_fasttext_model(sentences, model_params, data_dir)

if __name__ == "__main__":
    main()
