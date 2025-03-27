from pymilvus import connections, Collection, utility
from config import MilvusCloudConfig
import logging

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self._connect_to_cloud()
        self.collection = self._setup_collection()

    def _connect_to_cloud(self):
        """Establish secure connection to Milvus Cloud"""
        try:
            connections.connect(
                alias="default",
                uri=MilvusCloudConfig.URI,
                token=MilvusCloudConfig.API_KEY,
                secure=True
            )
            logger.info("Successfully connected to Milvus Cloud")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus Cloud: {str(e)}")
            raise

    def _setup_collection(self):
        """Create or load the vector collection"""
        if not utility.has_collection(MilvusCloudConfig.COLLECTION_NAME):
            from pymilvus import FieldSchema, CollectionSchema, DataType
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MilvusCloudConfig.VECTOR_DIMENSION)
            ]
            
            schema = CollectionSchema(fields, "Document search collection")
            collection = Collection(MilvusCloudConfig.COLLECTION_NAME, schema)
            
            collection.create_index(
                field_name="embedding",
                index_params=MilvusCloudConfig.INDEX_PARAMS
            )
            logger.info(f"Created new collection: {MilvusCloudConfig.COLLECTION_NAME}")
        else:
            collection = Collection(MilvusCloudConfig.COLLECTION_NAME)
            logger.info(f"Loaded existing collection: {MilvusCloudConfig.COLLECTION_NAME}")

        collection.load()
        return collection

    def search(self, query_embedding, top_k=5):
        """Perform vector similarity search"""
        search_params = {
            "data": [query_embedding],
            "anns_field": "embedding",
            "param": MilvusCloudConfig.SEARCH_PARAMS,
            "limit": top_k,
            "output_fields": ["text"]
        }
        
        try:
            results = self.collection.search(**search_params)
            return [{
                "text": hit.entity.get("text"),
                "score": hit.score,
                "id": hit.id
            } for hit in results[0]]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise