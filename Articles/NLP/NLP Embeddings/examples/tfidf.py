import math
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

def calculate_tf(tokens: List[str]) -> Dict[str, float]:
    """
    Calculates the Term Frequency (TF) values of tokens in a list.
    
    Args:
        tokens (list): A list of tokens for which to calculate TF values.
        
    Returns:
        dict: A dictionary containing tokens as keys and their corresponding TF values as values.
    """
    tf_dict = {}
    total_tokens = len(tokens)
    for token in tokens:
        if token in tf_dict:
            tf_dict[token] += 1
        else:
            tf_dict[token] = 1
    for token, count in tf_dict.items():
        tf_dict[token] = count / total_tokens
    return tf_dict

def calculate_idf(sentences: List[str]) -> Dict[str, float]:
    """
    Calculates the Inverse Document Frequency (IDF) values for tokens in a list of sentences.
    
    Args:
        sentences (list): A list of sentences for which to calculate IDF values.
        
    Returns:
        dict: A dictionary containing tokens as keys and their corresponding IDF values as values.
    """
    idf_dict = {}
    total_sentences = len(sentences)
    
    for sentence in sentences:
        tokens = set(tokenize(sentence))
        for token in tokens:
            if token in idf_dict:
                idf_dict[token] += 1
            else:
                idf_dict[token] = 1
                
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
    tfidf = {}
    for token, tf_value in tf.items():
        tfidf[token] = tf_value * idf.get(token, 0)
    return tfidf

if __name__ == '__main__':
    sent_one = "This movie is very scary and long"
    sent_two = "This movie is not scary and is slow"
    sent_three = "This movie is spooky and good"
    sentences = [sent_one, sent_two, sent_three]

    # Calculate IDF
    idf_dict = calculate_idf(sentences)

    # Calculate TF-IDF for each sentence
    tfidf_list = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        tf = calculate_tf(tokens)
        tfidf = calculate_tfidf(tf, idf_dict)
        tfidf_list.append(tfidf)
        print(tfidf)

