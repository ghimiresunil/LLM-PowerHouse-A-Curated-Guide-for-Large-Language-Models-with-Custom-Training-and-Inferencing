import os
import spacy
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import plotly.graph_objs as go
import chart_studio.plotly as py
from sklearn.manifold import TSNE
from typing import List, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Constants
ELMO_URL = "https://tfhub.dev/google/elmo/3"
EXCEL_FILE_PATH = '/content/elmo_data.xlsx'
DESCRIPTION_COLUMN = 'Description'
SEARCH_STRING = "what is thor's weapon"
RESULTS_RETURNED = 3

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from an Excel file into a Pandas DataFrame.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded data in a DataFrame.
    """
    df = pd.read_excel(file_path).reset_index(drop=True)
    return df

def preprocess_text(nlp: Any, text: str) -> List[str]:
    """
        Preprocess text by lowercasing, removing unwanted characters, and splitting into sentences.

    Args:
        nlp: Spacy NLP object for sentence splitting.
        text (str): Input text to be preprocessed.

    Returns:
        list: List of preprocessed sentences.
    """
    text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ')
    sentences = [i.text.strip() for i in nlp(text).sents if len(i) > 1]
    return sentences

def calculate_embeddings(sentences: List[str]) -> np.ndarray:
    """
        Calculate Elmo embeddings for a list of sentences.

    Args:
        sentences (list): List of sentences for which to calculate embeddings.

    Returns:
        np.ndarray: Calculated embeddings.
    """
    embed = hub.load(ELMO_URL)
    embeddings = embed.signatures["default"](tf.constant(sentences))["default"]
    x = embeddings.numpy()
    return x

def reduce_dimensions(x: np.ndarray) -> np.ndarray:
    """
        Reduce dimensions of input data using PCA and TSNE.

    Args:
        x (np.ndarray): Input data for dimensionality reduction.

    Returns:
        np.ndarray: Reduced dimensional representation.
    """
    pca_tsne = TSNE(n_components=2)
    y = pca_tsne.fit_transform(PCA(n_components=50).fit_transform(x))
    return y

def plot_embeddings(y: np.ndarray, sentences: List[str]) -> go.Figure:
    """
        Create a 2D scatter plot for visualizing embeddings.

    Args:
        y (np.ndarray): Reduced dimensional representation of embeddings.
        sentences (list): List of sentences for annotation.

    Returns:
        plotly.graph_objs.Figure: Scatter plot figure.
    """
    data = [
        go.Scatter(
            x=y[:, 0],
            y=y[:, 1],
            mode='markers',
            text=sentences,
            marker=dict(
                size=16,
                color=[len(i) for i in sentences],
                opacity=0.8,
                colorscale='viridis',
                showscale=False
            )
        )
    ]
    layout = dict(
        yaxis=dict(zeroline=False),
        xaxis=dict(zeroline=False)
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(width=900, height=600, title_text='Elmo Embeddings represented in 2 dimensions')
    return fig

def find_similar_sentences(search_string: str, embeddings: np.ndarray, sentences: List[str], results_returned: int) -> Tuple[List[float], List[str]]:
    """
        Find and rank sentences similar to a given search string.

    Args:
        search_string (str): Search string for similarity comparison.
        embeddings (np.ndarray): Embeddings of all sentences.
        sentences (list): List of sentences for comparison.
        results_returned (int): Number of similar sentences to return.

    Returns:
        tuple: Lists of similar scores and similar sentences.
    """
    similar_scores = []
    similar_terms = []
    
    embeddings2 = hub.load(ELMO_URL).signatures["default"](tf.constant([search_string],))["default"]
    search_vect = embeddings2.numpy()
    cosine_similarities = pd.Series(cosine_similarity(search_vect, embeddings).flatten())
    
    for i, j in cosine_similarities.nlargest(int(results_returned)).iteritems():
        similar_score = j
        similar_sentence = ' '.join([word if word.lower() in search_string else word for word in sentences[i].split()])
        
        similar_scores.append(similar_score)
        similar_terms.append(similar_sentence)
    
    return similar_scores, similar_terms

if __name__ == '__main__':
    df = load_data(EXCEL_FILE_PATH)
    nlp = spacy.load('en_core_web_sm')
    sentences = preprocess_text(nlp, ' '.join(df[DESCRIPTION_COLUMN]))
    embeddings = calculate_embeddings(sentences)
    reduced_embeddings = reduce_dimensions(embeddings)
    plot = plot_embeddings(reduced_embeddings, sentences)
    plot.show()

    similar_scores, similar_terms = find_similar_sentences(SEARCH_STRING, embeddings, sentences, RESULTS_RETURNED)
    
    # Create a DataFrame from similarity scores and similar terms
    similarity_df = pd.DataFrame({'Similarity Score': similar_scores, 'Similar Terms': similar_terms})
    print(similarity_df)
