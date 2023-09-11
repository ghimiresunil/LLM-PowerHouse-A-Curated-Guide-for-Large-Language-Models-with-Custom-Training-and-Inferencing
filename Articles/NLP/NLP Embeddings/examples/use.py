import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class UniversalSentenceEncoder:
    """
    Class to encapsulate Universal Sentence Encoder model for generating sentence embeddings.

    Attributes:
    - model (tensorflow_hub.Module): Loaded Universal Sentence Encoder model.

    Methods:
    - __init__(self): Initializes and loads Universal Sentence Encoder model from TensorFlow Hub.
    - compute_embeddings(self, sentences): Computes embeddings for a list of input sentences using the loaded Universal Sentence Encoder model.
    - compute_similarity(embedding1, embedding2): Computes the cosine similarity between two embeddings.
    """

    def __init__(self):
        """
        Initializes an instance of the UniversalSentenceEncoder class.
        Loads the Universal Sentence Encoder model from TensorFlow Hub.
        """
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def compute_embeddings(self, sentences):
        """
        Computes embeddings for a list of input sentences using the loaded Universal Sentence Encoder model.

        Args:
        - sentences (List[str]): Input sentences for which embeddings need to be computed.

        Returns:
        - embeddings (tf.Tensor): Embeddings of the input sentences.
        """
        embeddings = self.model(sentences)
        return embeddings

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        """
        Computes the cosine similarity between two embeddings.

        Args:
        - embedding1 (tf.Tensor): First embedding.
        - embedding2 (tf.Tensor): Second embedding.

        Returns:
        - similarity (tf.Tensor): Cosine similarity between the two embeddings.
        """
        similarity = tf.reduce_sum(embedding1 * embedding2, axis=-1) / (
            tf.norm(embedding1, axis=-1) * tf.norm(embedding2, axis=-1)
        )
        return similarity


def visualize_embeddings(embeddings, sentences, dimensions=2):
    """
    Visualizes the embeddings of sentences using Principal Component Analysis (PCA) for dimensionality reduction.

    Args:
    - embeddings (numpy.ndarray): Embeddings of sentences to be visualized.
    - sentences (List[str]): Corresponding sentences for each embedding.
    - dimensions (int, Optional): Number of dimensions for visualization (2 or 3).
    """
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


if __name__ == '__main__':
    use = UniversalSentenceEncoder()

    input_sentences = [
        "This is an example sentence.",
        "Machine learning is fascinating.",
        "Natural language processing is challenging."
    ]

    embeddings = use.compute_embeddings(input_sentences)
    visualize_embeddings(embeddings, input_sentences, dimensions=2)
