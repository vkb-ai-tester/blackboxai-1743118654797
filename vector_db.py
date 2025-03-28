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
    def __init__(self, reset: bool = False):
        self._connect_with_retry()
        if reset:
            self._reset_collection()
        self.collection = self._setup_collection()

    def _reset_collection(self):
        """Force reset the collection by dropping it if exists"""
        try:
            if utility.has_collection(MilvusConfig.COLLECTION_NAME):
                utility.drop_collection(MilvusConfig.COLLECTION_NAME)
                logger.info(f"Dropped existing collection: {MilvusConfig.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise VectorDBError(f"Collection reset failed: {str(e)}")

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
            from pymilvus import FieldSchema, CollectionSchema, DataType
            
            collection_name = MilvusConfig.COLLECTION_NAME
            logger.info(f"Checking for existing collection: {collection_name}")
            
            if not utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} not found, creating new one")
                
                # Get actual embedding dimension from sample
                from transformers import CLIPProcessor, CLIPModel
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                sample_input = processor(text="sample", return_tensors="pt", padding=True)
                sample_embedding = model.get_text_features(**sample_input)[0].tolist()
                embedding_dim = len(sample_embedding)
                logger.info(f"Detected embedding dimension: {embedding_dim}")
                
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
                
                schema = CollectionSchema(fields, "Multimodal document search collection")
                collection = Collection(collection_name, schema)
                logger.info(f"Created collection schema for {collection_name}")
                
                # Create indexes
                index_params = MilvusConfig.get_index_params()
                logger.info(f"Creating text embedding index with params: {index_params}")
                collection.create_index(
                    field_name="text_embedding",
                    index_params=index_params
                )
                
                search_params = MilvusConfig.get_search_params()
                logger.info(f"Creating image embedding index with params: {search_params}")
                collection.create_index(
                    field_name="image_embedding",
                    index_params=search_params
                )
                
                logger.info(f"Successfully created new collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} exists, loading it")
                collection = Collection(collection_name)
                logger.info(f"Successfully loaded existing collection: {collection_name}")
            
            logger.info(f"Loading collection {collection_name} into memory")
            collection.load()
            logger.info(f"Collection {collection_name} loaded with {collection.num_entities} entities")
            return collection
        except Exception as e:
            logger.error(f"Collection setup failed: {str(e)}")
            raise VectorDBError(f"Collection setup failed: {str(e)}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search using text embeddings"""
        search_params = {
            "data": [query_embedding],
            "anns_field": "text_embedding",
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
