import pandas as pd
from pymilvus import utility
from pymilvus import Collection, DataType, CollectionSchema, FieldSchema
from data_access.collection_base import MilvusCollectionBase

class ClassificationCollection(MilvusCollectionBase):
    def __init__(self):
        self.collection_name = "TicketClassification"
        self.collection = None
        self.create_or_load_collection()

    def create_or_load_collection(self):
        if utility.has_collection(self.collection_name):
            print("Collection is already created.")
            collection = Collection(self.collection_name)
            collection.load()
            self.collection = collection
        else:
            self.create_collection()

    def create_collection(self):
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024},
        }
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
        )

        product_purchased_field = FieldSchema(
            name="product_purchased", dtype=DataType.VARCHAR, max_length=100
        )

        ticket_subject_field = FieldSchema(
            name="ticket_subject", dtype=DataType.VARCHAR, max_length=100
        )

        ticket_description_field = FieldSchema(
            name="ticket_description", dtype=DataType.VARCHAR, max_length=1000
        )

        ticket_description_embeddings_field = FieldSchema(
            name="ticket_description_embeddings", dtype=DataType.FLOAT_VECTOR, dim=768
        )

        schema = CollectionSchema(
            fields=[
                id_field,
                product_purchased_field,
                ticket_subject_field,
                ticket_description_field,
                ticket_description_embeddings_field,
            ],
            description="Ticket Collection For Testing.",
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using="default",
            shards_num=2,
            consistency_level="Strong",
        )

        collection.create_index(
            field_name="ticket_description_embeddings", index_params=index_params
        )

        self.collection = collection
        self.collection.load()

    def insert(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")

        try:
            insert_response = self.collection.insert(data)
            return True
        except Exception as e:
            print("Error while inserting data")
            return False

    def delete(self, expr):
        count = self.collection.delete(expr)
        print("Total number of deleted items: ", count)
        return True

    def hybrid_search(
        self,
        embeddings,
        anns_field: str,
        product_purchased: str = None,
        ticket_subject: str = None,
        top_k: int = 5,
    ):
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}

        if product_purchased and ticket_subject:
            print("content_name found")
            search_result = self.collection.search(
                data=embeddings,
                anns_field=anns_field,
                param=search_params,
                limit=top_k,
                expr=f'product_purchased == "{product_purchased}" AND ticket_subject == "{ticket_subject}"',
                output_fields=[
                    "id",
                    "product_purchased",
                    "ticket_subject",
                    "ticket_description",
                ],
            )
        else:
            print("product_purchased and ticket_subject not found")
            search_result = self.collection.search(
                data=embeddings,
                anns_field=anns_field,
                param=search_params,
                limit=top_k,
                output_fields=[
                    "id",
                    "product_purchased",
                    "ticket_subject",
                    "ticket_description",
                ],
            )
        return search_result

    def get_primary_keys_associated(self, product_purchased, ticket_subject):
        exp = f'product_purchased == "{product_purchased}" and ticket_subject == "{ticket_subject}"'
        results = self.collection.query(expr=exp)
        primary_key_list = [result["id"] for result in results]
        return primary_key_list

    def is_content_exist(self, product_purchased, ticket_subject):
        exp = f'product_purchased == "{product_purchased}" and ticket_subject == "{ticket_subject}"'
        res = self.collection.query(expr=exp)
        print("Count:", len(res))
        return len(res) > 0
