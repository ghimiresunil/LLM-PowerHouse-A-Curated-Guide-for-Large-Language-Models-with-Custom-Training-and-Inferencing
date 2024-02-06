# Vector Database

Welcome to Vector Database, a versatile tool designed for efficient management and querying of vectors using Milvus! Follow the steps below to seamlessly set up and run the code.

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

![Vector Database](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/51bb044d-4959-40f4-b2a6-f81eb820c97e)

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

# Conclusion
Now you are all set to explore the capabilities of Vector Database with Milvus! Feel free to reach out if you encounter any issues or have questions. Happy coding!
