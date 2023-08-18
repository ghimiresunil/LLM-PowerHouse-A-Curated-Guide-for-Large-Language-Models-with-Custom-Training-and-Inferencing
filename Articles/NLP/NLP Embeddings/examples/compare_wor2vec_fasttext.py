import gensim
import numpy as np
import pandas as pd

def compare_results(word: str, model_ft: gensim.models.FastText, model_w2v: gensim.models.Word2Vec) -> pd.DataFrame:
    """
    Compare similarity search results between FastText and Word2Vec models for a given word.
    
    Args:
        word (str): The word to compare.
        model_ft (gensim.models.FastText): The FastText model.
        model_w2v (gensim.models.Word2Vec): The Word2Vec model.

    Returns:
        pd.DataFrame: A DataFrame containing comparison results.
    """
    word_count_ft = model_ft.wv.get_vecattr(word, 'count')
    word_count_w2v = model_w2v.wv.get_vecattr(word, 'count')
    print(f"Word '{word}' has {word_count_ft} samples in training text.")

    print('Running similarity searches...')
    results_ft = model_ft.wv.most_similar(word)
    results_w2v = model_w2v.wv.most_similar(word)

    table_rows = []

    for (word_ft, score_ft), (word_w2v, score_w2v) in zip(results_ft, results_w2v):
        count_ft = model_ft.wv.get_vecattr(word_ft, 'count')
        count_w2v = model_w2v.wv.get_vecattr(word_w2v, 'count')

        table_rows.append(
            (word_ft, f'{count_ft:,}', f'{score_ft:.2f}', word_w2v, f'{count_w2v:,}', f'{score_w2v:.2f}')
        )

    columns = ['fasttext', 'freq_ft', 'score_ft', 'word2vec', 'freq_w2v', 'score_w2v']
    df = pd.DataFrame(table_rows, columns=columns)
    return df

if __name__ == '__main__':
    model_ft = gensim.models.FastText.load('../data/fasttext.model')
    model_w2v = gensim.models.Word2Vec.load('../data/word2vec.model')

    word_to_compare = 'example'
    result_df = compare_results(word_to_compare, model_ft, model_w2v)
    print(result_df.head())

    # Calculate and store vector norms and similarities
    word_vectors_w2v = model_w2v.wv
    word_vectors_ft = model_ft.wv
    
    norm_stupid = np.linalg.norm(word_vectors_w2v['stupid'])
    norm_bwahahahaha = np.linalg.norm(word_vectors_w2v['bwahahahaha'])

    similarity_stupid_dumb_ft = word_vectors_ft.similarity('stupid', 'dumb')
    similarity_stupid_dumb_w2v = word_vectors_w2v.similarity('stupid', 'dumb')

    # Print the results
    print(f"Norm of 'stupid': {norm_stupid:.3f}")
    print(f"Norm of 'bwahahahaha': {norm_bwahahahaha:.3f}")
    print("Similarity between 'stupid' and 'dumb:'")
    print(f"  fasttext: {similarity_stupid_dumb_ft:.2f}")
    print(f"  word2vec: {similarity_stupid_dumb_w2v:.2f}")
