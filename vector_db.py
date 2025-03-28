from pymilvus import connections, Collection, utility
from config import MilvusConfig
import logging
import backoff
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VectorDBError(Exception):
    """Custom exception for Vector DB operations"""
    pass

class VectorDBService:
    def __init__(self):
        self._connect_with_retry()
        self.collection = self._setup_collection()

    @backoff.on_exception(backoff.expo,
                         Exception,
                         max_tries=3,
                         logger=logger)
    def _connect_with_retry(self):
        """Establish secure connection to Milvus with retry logic"""
        try:
            connections.connect(
                alias="default",
                uri=MilvusConfig.URI,
                token=MilvusConfig.API_KEY,
                secure=True
            )
            logger.info("Successfully connected to Milvus")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise VectorDBError(f"Connection failed: {str(e)}")

    def _setup_collection(self) -> Collection:
        """Create or load the vector collection with validation"""
        try:
            if not utility.has_collection(MilvusConfig.COLLECTION_NAME):
                from pymilvus import FieldSchema, CollectionSchema, DataType
                
                
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MilvusConfig.VECTOR_DIMENSION),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
                
                schema = CollectionSchema(fields, "Document search collection")
                collection = Collection(MilvusConfig.COLLECTION_NAME, schema)
                
                collection.create_index(
                    field_name="embedding",
                    index_params=MilvusConfig.get_index_params()
                )
                logger.info(f"Created new collection: {MilvusConfig.COLLECTION_NAME}")
            else:
                collection = Collection(MilvusConfig.COLLECTION_NAME)
                logger.info(f"Loaded existing collection: {MilvusConfig.COLLECTION_NAME}")

            collection.load()
            return collection
        except Exception as e:
            logger.error(f"Collection setup failed: {str(e)}")
            raise VectorDBError(f"Collection setup failed: {str(e)}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search with enhanced error handling"""
        search_params = {
            "data": [query_embedding],
            "anns_field": "embedding",
            "param": MilvusConfig.get_search_params(),
            "limit": top_k,
            "output_fields": ["text", "metadata"]
        }
        
        try:
            results = self.collection.search(**search_params)
            return [{
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata") or {},
                "score": hit.score,
                "id": hit.id
            } for hit in results[0]]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorDBError(f"Search operation failed: {str(e)}")

    def health_check(self) -> bool:
        """Check if the vector DB connection is healthy"""
        try:
            return connections.has_connection("default")
        except Exception:
            return False
