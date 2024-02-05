# 01. Vector Database

- Used for efficient storage, indexing, and retrieval of high-dimensional vectors
- Used to work with the unstructured data that is inefficient to store in the traditional database
- Uses approximate nearest neighbor (ANN) for indexing and fast retrieval
- **ANN algorithms**: HNSW, LSH, IVF, PQ, k-d trees, annoy, etc
- **Vector Database Options**: Milvus, Pinecone, Weaviate, ChromaDB, Qdrant, Vespa, Redis, etc
- Use Cases:
    - Semantic Search
    - Similarity Search
    - Clustering and Classification
    - Recommendation System
    - Anomaly and Fraud Detection
  
![image](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/51bb044d-4959-40f4-b2a6-f81eb820c97e)


Welcome to Vector Database, a powerful tool for managing and querying vectors efficiently using Milvus! Follow the steps below to set up and run the code seamlessly.

# 02. Prerequisites
1. It is highly recommended to create a new virtual environment using Python/Conda.
- Conda
```bash
conda create --name env_name python==3.9
```
- Python
```bash
python -m venv venv
```

2. Activate virtual Environment

- conda
```bash
conda create --name env_name python==3.9
```
- Python
```bash
source venv/bin/activate 

```
3. Install the required packages using pip.
```
pip install -r requirements.txt
```

4. Setting Up Milvus Server Locally and run docker-compose to start the Milvus server.
```bash
cd scripts
docker-compose -f milvus-docker-compose.yml up -d
```

5. Setting PYTHONPATH (if needed)
```bash
export PYTHONPATH=your_directory_where_you_clone_this_repo/vector_database/milvus_database
```

# 03. Example for Create, Insert, Update, Delete and Search

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

# 04. Conclusion
Now you are all set to explore the capabilities of Vector Database with Milvus! Feel free to reach out if you encounter any issues or have questions. Happy coding!
