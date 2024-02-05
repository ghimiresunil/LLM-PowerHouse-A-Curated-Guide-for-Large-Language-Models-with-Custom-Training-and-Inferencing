import pandas as pd
from data_access.utils import ClassificationCollection


class EmbeddingService:
    def __init__(self, embedding_service):
        self.embedding_processor = embedding_service
        self.ticket_classification_collection = ClassificationCollection()

    @staticmethod
    def filter_search_results(results, thresh):
        final_results = []
        for i in results[0]:
            if i.distance >= thresh:
                final_results.append(
                    (
                        i.entity.get("id"),
                        i.entity.get("product_purchased"),
                        i.entity.get("ticket_subject"),
                        i.entity.get("ticket_description"),
                        i.distance,
                    )
                )
        return final_results

    def insert_data(self, data):
        if type(data) == dict:
            product_purchased_list = [data["product_purchased"]]
            ticket_subject_list = [data["ticket_subject"]]
            ticket_description_list = [data["ticket_description"]]
        elif type(data) == pd.DataFrame:
            data = pd.DataFrame(data)
            data.columns = ["product_purchased", "ticket_subject", "ticket_description"]
            product_purchased_list = list(data["product_purchased"])
            ticket_subject_list = list(data["ticket_subject"])
            ticket_description_list = list(data["ticket_description"])

        ticket_description_embeddings_list = self.embedding_processor.get_embeddings(
            text_list=ticket_description_list
        )
        data_for_insertion = [
            product_purchased_list,
            ticket_subject_list,
            ticket_description_list,
            ticket_description_embeddings_list,
        ]

        success = self.ticket_classification_collection.insert(data=data_for_insertion)

        if success:
            print("Insertion Successful")
        else:
            print("Insertion Failed")

        return success

    def delete_data(self, product_purchased, ticket_subject):
        primary_keys_product_ticket = (
            self.ticket_classification_collection.get_primary_keys_associated(
                product_purchased=product_purchased, ticket_subject=ticket_subject
            )
        )

        if len(primary_keys_product_ticket) > 0:
            expr = f"id in {primary_keys_product_ticket}"
            print(expr)
            self.ticket_classification_collection.delete(expr=expr)
        else:
            print("No Primary Keys found")

    def update_data(self, data):
        product_purchased = data["product_purchased"]
        ticket_subject = data["ticket_subject"]
        update_response = None

        if self.ticket_classification_collection.is_content_exist(
            product_purchased, ticket_subject
        ):
            print("Product Purchaed and Ticket Subject Exist")
            self.delete_data(product_purchased, ticket_subject)
            update_response = self.insert_data(data=data)
        else:
            print("Product Purchaed and Ticket Subject Exist")

        return update_response

    def sentence_similarity_search(
        self, query, product_purchased=None, ticket_subject=None, thresh=0.6
    ):
        if len(query) > 1:
            query_embeddings = self.embedding_processor.get_embeddings(list(query))

            search_result_content_name = (
                self.ticket_classification_collection.hybrid_search(
                    embeddings=query_embeddings,
                    anns_field="ticket_description_embeddings",
                    product_purchased=product_purchased,
                    ticket_subject=ticket_subject,
                )
            )
            final_search_result = self.filter_search_results(
                results=search_result_content_name, thresh=thresh
            )

            print("Results obtained: ", final_search_result)
            return final_search_result

        else:
            print("Query not found")
