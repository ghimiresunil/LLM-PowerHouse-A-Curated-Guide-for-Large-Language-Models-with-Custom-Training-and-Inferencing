import math
import pandas as pd
from collections import Counter
from typing import List, Dict

def tokenize(sentence: str) -> List[str]:
    """
    Tokenizes a given sentence into a list of lowercase tokens.

    Args:
        sentence (str): The input sentence to be tokenized.

    Returns:
        list: A list of lowercase tokens extracted from the sentence.
    """
    return sentence.lower().split()

def calculate_idf(sentences: List[str]) -> Dict[str, float]:
    """
    Calculates the Inverse Document Frequency (IDF) values for tokens in a list of sentences.

    Args:
        sentences (list): A list of sentences for which to calculate IDF values.

    Returns:
        dict: A dictionary containing tokens as keys and their corresponding IDF values as values.
    """
    idf_dict = Counter()

    for sentence in sentences:
        tokens = set(tokenize(sentence))
        idf_dict.update(tokens)

    total_sentences = len(sentences)
    for token, count in idf_dict.items():
        idf_dict[token] = math.log(total_sentences / count)

    return idf_dict

def calculate_tfidf(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates the TF-IDF values for tokens using given TF and IDF values.

    Args:
        tf (dict): A dictionary containing tokens as keys and their corresponding TF values as values.
        idf (dict): A dictionary containing tokens as keys and their corresponding IDF values as values.

    Returns:
        dict: A dictionary containing tokens as keys and their corresponding TF-IDF values as values.
    """
    tfidf = {token: tf_value * idf.get(token, 0) for token, tf_value in tf.items()}
    return tfidf

def main(sentences: List[str]) -> List[Dict[str, float]]:
    """
    Calculates the TF-IDF values for a list of sentences.

    Args:
        sentences (list): A list of sentences for which to calculate TF-IDF values.

    Returns:
        list: A list of dictionaries, each containing the TF-IDF values for a sentence.
    """
    idf_dict = calculate_idf(sentences)
    tfidf_list = []

    index_labels = [f'Sentence {i + 1}' for i in range(len(sentences))]

    for sentence in sentences:
        tokens = tokenize(sentence)
        tf = {token: count / len(tokens) for token, count in Counter(tokens).items()}
        tfidf = calculate_tfidf(tf, idf_dict)
        tfidf_list.append(tfidf)

    # Construct DataFrame with dynamic index labels
    df = pd.DataFrame(tfidf_list, index=index_labels)
    df.fillna(0, inplace=True)
    return df

if __name__ == '__main__':
    sent_one = "This movie is very scary and long"
    sent_two = "This movie is not scary and is slow"
    sent_three = "This movie is spooky and good"
    sentences = [sent_one, sent_two, sent_three]
    df = main(sentences)
    print(df.head())