import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class UniversalSentenceEncoder:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_embeddings(self, sentences):
        embeddings = self.model(sentences)
        return embeddings

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        similarity = tf.reduce_sum(embedding1 * embedding2, axis=-1) / (
            tf.norm(embedding1, axis=-1) * tf.norm(embedding2, axis=-1)
        )
        return similarity
    

def visualize_embeddings(embeddings, sentences, dimensions=2):
    pca = PCA(n_components=dimensions)
    reduced_embeddings = pca.fit_transform(embeddings)

    if dimensions == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        for i, sentence in enumerate(sentences):
            plt.annotate(sentence, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Universal Sentence Encoding 2D Visualization")
        
    elif dimensions == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2]
        )
        for i, sentence in enumerate(sentences):
            ax.text(
                reduced_embeddings[i, 0],
                reduced_embeddings[i, 1],
                reduced_embeddings[i, 2],
                sentence
            )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.set_title("Universal Sentence Encoding 3D Visualization")
        
    plt.tight_layout()
    plt.show()
