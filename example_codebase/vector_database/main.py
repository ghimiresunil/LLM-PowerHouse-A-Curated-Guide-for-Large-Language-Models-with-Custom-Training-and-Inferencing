import pandas as pd
from data_access.access_milvus import MilvusDBConnection
from data_access.utils import ClassificationCollection
from collection_service.embedding_generator import  EmbeddingGenerator
from collection_service.embedding_service import EmbeddingService

if __name__ == '__main__':
    connection = MilvusDBConnection()
    connection.start()
    collection = ClassificationCollection()
    
    embedding = EmbeddingGenerator(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    service = EmbeddingService(
        embedding_service=embedding
    )
    # # content_name= "hepatitis_b"
    # res = service.sentence_similarity_search(
    #     query= "Who is machine learning Engineer?",
    #     # content_name=content_name,
    #     thresh= 0.01
    # )

    # print("RESULTS: ", res)
    product_purchased = "Dell XPS"
    ticket_subject = "Network problem"
    data = {
        "product_purchased": product_purchased,
        "ticket_subject": ticket_subject,
        "ticket_description":"The GoPro Hero7 Black is a powerful camera that can capture high quality video and photos in various settings."
    }
    
    response = service.update_data(
        data=data
    )
    # data = pd.read_csv('data/customer_data.csv')
    # success =service.delete_data(
    #     data= data
    # )
    connection.stop()
    