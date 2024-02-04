from abc import ABC

class MilvusCollectionBase(ABC):
    def create_collection(self, collection_name):
        raise NotImplementedError
    
    def insert(self, data):
        raise NotImplementedError
    
    def hybrid_search(self, *args):
        raise NotImplementedError
    
    def delete(self, expr):
        raise NotImplementedError