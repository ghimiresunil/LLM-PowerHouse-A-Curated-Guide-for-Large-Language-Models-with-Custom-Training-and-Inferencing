# Vector Database

Welcome to Vector Database, a versatile tool designed for efficient management and querying of vectors using Milvus! Follow the steps below to seamlessly set up and run the code.

![Vector Database](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/51bb044d-4959-40f4-b2a6-f81eb820c97e)

## Overview

Vector Databases have gained immense popularity with the rise of Foundational Models. Initially associated with Large Language Models, Vector Databases prove valuable in various Machine Learning applications dealing with Vector Embeddings.

### Key Features

- **Efficient Storage and Retrieval:** Enables storage, indexing, and retrieval of high-dimensional vectors.
- **Unstructured Data Handling:** Manages unstructured data efficiently, unsuitable for traditional databases.
- **ANN Algorithms:** Utilizes approximate nearest neighbor (ANN) algorithms for indexing and fast retrieval.
- **Diverse Algorithm Support:** Supports ANN algorithms like HNSW, LSH, IVF, PQ, k-d trees, annoy, etc.
- **Multiple Database Options:** Includes Milvus, Pinecone, Weaviate, ChromaDB, Qdrant, Vespa, Redis, and more.

### Use Cases

- Semantic Search
- Similarity Search
- Clustering and Classification
- Recommendation System
- Anomaly and Fraud Detection

## Retrieval and Approximate Nearest Neighbour (ANN) Search

In the context of Vector Databases, retrieval involves obtaining a set of vectors most similar to a query vector within the same latent space. This retrieval process is known as Approximate Nearest Neighbour (ANN) search.

### Examples of Queries

- Finding similar images based on a given image.
- Retrieving relevant context for a question, transformable into an answer via a Large Language Model (LLM).

## Interaction with Vector Database

- Writing/Updating Data
    - Choose an ML model to generate Vector Embeddings.
    - Embed various types of information: text, images, audio, tabular, based on the data type.
    - Obtain a Vector representation of your data by running it through the chosen Embedding Model.
    - Store additional metadata along with the Vector Embedding for pre-filtering or post-filtering ANN search results.
    - Vector Database indexes Vector Embeddings and metadata separately using methods like Random Projection, Product Quantization, and Locality-sensitive Hashing.

- Reading Data
    - A query typically consists of two parts:
        - Data for ANN search (e.g., an image to find similar ones).
        - Metadata query to filter vectors based on known qualities (e.g., exclude images in a specific location).
    - Execute Metadata Queries against the metadata index before or after the ANN search procedure.
    - Apply ANN search, retrieving a set of Vector embeddings.
    - Popular similarity measures for ANN search include Cosine Similarity, Euclidean Distance, and Dot Product.


## How Vector Databases Operate

Understanding the functioning of vector databases involves breaking down the process into several key steps:

1. **Data Input:**
   - Begin with a dataset containing sentences, each consisting of three words (or tokens). 
   - Real-world datasets may scale to millions or billions of sentences, with tens of thousands of tokens.
   
2. **Word Embeddings:**
   - Retrieve word embedding vectors for each word from a table of 22 vectors, with the vocabulary size being 22. 
   - In practice, vocabulary sizes can extend to tens of thousands, and word embedding dimensions can reach thousands (e.g., 1024, 4096).
   
3. **Encoding:**
   - Feed the sequence of word embeddings into an encoder, generating a sequence of feature vectors (one per word). 
   - The encoder, often a transformer or variant, employs a simple one-layer perceptron (linear layer + ReLU).
   
4. **Mean Pooling:**
   - Merge the sequence of feature vectors into a single vector using mean pooling, which involves averaging across the columns. 
   - This resultant vector is commonly referred to as "text embeddings" or "sentence embeddings." 
   - While other pooling techniques exist, mean pooling is widely adopted.
   
5. **Indexing:**
   - Reduce the dimensions of the text embedding vector using a projection matrix, achieving a 50% reduction (e.g., 4 to 2). 
   - In practice, the values in this matrix are more random. 
   - This reduction aims to provide a concise representation for faster comparison and retrieval, similar to the purpose of hashing. 
   - The dimension-reduced index vector is stored in the vector storage.
   
6. **Processing Queries:**
   - Repeat the steps from word embeddings to indexing for each new input (e.g., "who are you," "who am I").

> With the dataset indexed in the vector database, the querying process follows:

7. **Query Processing:**
   - Generate a 2-dimensional query vector by repeating the steps from word embeddings to indexing for a given query (e.g., "am I you").

8. **Dot Products:**
   - Compute dot products between the query vector and database vectors, all in 2 dimensions. 
   - The objective is to leverage dot products for estimating similarity. Transposing the query vector transforms this step into a matrix multiplication.

9. **Nearest Neighbor Search:**
   - Identify the largest dot product through linear scan. The sentence with the highest dot product is considered the nearest neighbor (e.g., "who am I"). 
   - For efficiency, in real-world scenarios involving billions of vectors, Approximate Nearest Neighbor (ANN) algorithms like Hierarchical Navigable Small Worlds (HNSW) are often employed.

## Vector Search Explained through Real-World Stories:

![image](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/ea31fcb9-20f2-4244-92e4-c6f30ee6488f)

Imagine you're not searching for keywords on Google, but for the essence of what you're looking for. Vector search does exactly that, using stories to illustrate its power:

- **Story 1**: Finding Similar Recipes:
    - You loved your grandma's secret pasta sauce recipe, but can't remember the exact ingredients. Instead of keyword searching, you describe the taste: "rich, garlicky, with a hint of smokiness."
    - Vector search analyzes text descriptions of thousands of recipes, understanding the semantic meaning beyond keywords. It finds recipes with similar "taste vectors," even if they don't mention "smokiness" explicitly. Bingo! You rediscover your grandma's magic.
- **Story 2**: Identifying Music Genre:
    - You hum a melody you heard but can't recall the title or artist. Frustrated, you hum into a vector search app.
    - This app translates your hum into an audio vector, capturing the melody's essence. It compares it to vectors representing millions of songs, identifying similar musical styles and suggesting potential matches. You find the song and relive the memory!
- **Story 3**: Recommending Products:
    - You browse an online clothing store but feel overwhelmed by choices. You describe your ideal outfit: "flowy, bohemian, vibrant colors."
    - Vector search analyzes product descriptions and images, creating "style vectors" for each item. It finds clothes with similar vectors to your description, showcasing options you might have missed. You discover a dress that's exactly what you envisioned!
- **Story 4**: Searching Scientific Literature:
    - You're a researcher searching for specific information in a vast ocean of scientific papers. Keywords might not suffice.
    - Vector search analyzes the papers' content, creating "knowledge vectors" summarizing key concepts. You describe your research question, translated into a query vector. The search engine retrieves papers with "knowledge vectors" closest to your query, offering relevant and efficient results.
- Beyond Keywords:
    - These stories highlight how vector search goes beyond keyword matching. It captures the **meaning** behind words, images, sounds, and data, enabling more relevant and intuitive search experiences across various domains.
> Remember, this is just the beginning! As vector search technology advances, expect even more fascinating and personalized search experiences in the future.

## Prerequisites

Before you begin, make sure you have the following prerequisites installed.

### Create Virtual Environment

It is highly recommended to use a virtual environment for your project. Choose one of the following methods:

#### Using Conda

```bash
conda create --name env_name python==3.9
```

#### Using Python

```bash
python -m venv venv
```

### Activate Virtual Environment

Activate the virtual environment based on your chosen method:

#### Using Conda

```bash
conda activate env_name
```

#### Using Python

```bash
source venv/bin/activate
```

### Install Required Packages

Install the necessary Python packages by running:

```bash
pip install -r requirements.txt
```

### Setting Up Milvus Server

To use Milvus, you need to set up the server locally. Follow these steps:

#### Navigate to the scripts directory.

```bash
cd scripts
```

#### Run Docker Compose to start the Milvus server.

```bash
docker-compose -f milvus-docker-compose.yml up -d
```

### Setting PYTHONPATH (if needed)

If your project requires a specific PYTHONPATH, set it with the following command:

```bash
export PYTHONPATH=your_directory_where_you_clone_this_repo/vector_database/milvus_database
```

## Example for Create, Insert, Update, Delete and Search

This repository provides an example of using Milvus for creating a collection, inserting data, updating records, deleting data, and searching for similar sentences.

### Create Collection
```python
from data_access.utils import ClassificationCollection
from data_access.access_milvus import MilvusDBConnection

if __name__ == "__main__":
    connection = MilvusDBConnection()
    connection.start()
    collection = ClassificationCollection()    
    connection.stop()
```

### Insert Data

```python
import pandas as pd
from data_access.access_milvus import MilvusDBConnection
from collection_service.embedding_generator import EmbeddingGenerator
from collection_service.embedding_service import EmbeddingService

if __name__ == "__main__":
    data_path = 'data/customer_data.csv'
    model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    connection = MilvusDBConnection()
    connection.start()
    embedding = EmbeddingGenerator(
        model_name = model_name
    )
    service = EmbeddingService(embedding_service=embedding)

    ticket_classifcation_df = pd.read_csv(data_path)
    insert_data = service.insert_data(
        data= ticket_classifcation_df
    )
    connection.stop()
```

### Update Data
```python
import pandas as pd
from data_access.access_milvus import MilvusDBConnection
from collection_service.embedding_generator import EmbeddingGenerator
from collection_service.embedding_service import EmbeddingService

if __name__ == "__main__":
    model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    connection = MilvusDBConnection()
    connection.start()
    embedding = EmbeddingGenerator(
        model_name=model_name
    )
    service = EmbeddingService(embedding_service=embedding)
    update_data = {
        'product_purchased':'GoPro Hero',
        'ticket_subject':'Hardware issue',
        'ticket_description': 'I want to buy a GoPro Hero 9 Black.'
    }
    update_data = service.update_data(
        data= update_data
    )
    connection.stop()
```

### Delete Data
```python
from data_access.access_milvus import MilvusDBConnection
from collection_service.embedding_generator import EmbeddingGenerator
from collection_service.embedding_service import EmbeddingService

if __name__ == "__main__":
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    connection = MilvusDBConnection()
    connection.start()
    embedding = EmbeddingGenerator(model_name=model_name)
    service = EmbeddingService(embedding_service=embedding)
    product_purchased = "GoPro Hero"
    ticket_subject = "Hardware issue"
    delete_data = service.delete_data(
        product_purchased=product_purchased, ticket_subject=ticket_subject
    )
    connection.stop()
```

### Search Data
```python
from data_access.access_milvus import MilvusDBConnection
from collection_service.embedding_generator import EmbeddingGenerator
from collection_service.embedding_service import EmbeddingService

if __name__ == "__main__":
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    connection = MilvusDBConnection()
    connection.start()
    embedding = EmbeddingGenerator(model_name=model_name)
    service = EmbeddingService(embedding_service=embedding)
    res = service.sentence_similarity_search(
        query= "I want to buy a GoPro Hero 9 Black",
        thresh= 0.34
    )
    connection.stop()
```

## Conclusion
=======
Now you are all set to explore the capabilities of Vector Database with Milvus! Feel free to reach out if you encounter any issues or have questions. Happy coding!
