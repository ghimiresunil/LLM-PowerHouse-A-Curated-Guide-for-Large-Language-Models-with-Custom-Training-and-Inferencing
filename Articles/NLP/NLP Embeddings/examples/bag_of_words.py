import re
import pandas as pd 
from typing import List

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by converting to lowercase and removing non-alphanumeric characters.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]','', text)
    return text

def tokenize(text: str) -> List[str]:
    """
    Tokenizes the input text into a list of words.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of tokens (words).
    """
    return text.split()

def create_vocab(tokenized_texts: List[List[str]]) -> List[str]:
    """
    Creates a vocabulary set from a list of tokenized texts.
    
    Args:
        tokenized_texts (list): A list of tokenized texts.

    Returns:
        list: A sorted list of unique words in the vocabulary.
    """
    vocab = set()
    for tokens in tokenized_texts:
        vocab.update(tokens)
    return sorted(vocab)

def create_bow(text: str, vocab: List[str]) -> List[int]:
    """
    Creates a Bag-of-Words (BoW) representation of the input text based on the given vocabulary.

    Args:
        text (str): The input text.
        vocab (list): The vocabulary list.

    Returns:
        list: The BoW vector representing the input text.
    """
    bow = [0] * len(vocab)
    tokenized_text = tokenize(text)
    for token in tokenized_text:
        if token in vocab:
            index = vocab.index(token)
            bow[index] += 1
    return bow

if __name__ == '__main__':
    texts = [
        "This is a simple example.",
        "Another example for demonstration.",
        "Yet another text to process.",
    ]

    tokenized_texts = [tokenize(preprocess_text(text)) for text in texts]
    vocab = create_vocab(tokenized_texts)

    bows = [create_bow(text, vocab) for text in texts]

    print("Vocabulary:", vocab)
    for i, bow in enumerate(bows):
        print(f"BoW {i+1}:", bow)

    data = {"Vocabulary": vocab}
    for i, bow in enumerate(bows):
        data[f"BoW {i+1}"] = bow

    df = pd.DataFrame(data)
    print(df)
