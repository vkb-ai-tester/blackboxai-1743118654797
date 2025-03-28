from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VectorDBError(Exception):
    """Custom exception for Vector DB operations"""
    pass

class QdrantVectorDB:
    def __init__(self, collection_name: str = "document_search", vector_size: int = 512):
        self.client = QdrantClient(":memory:")  # Use in-memory for demo
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        try:
            # Create collection if it doesn't exist
            collections = self.client.get_collections()
            if not any(c.name == collection_name for c in collections.collections):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {str(e)}")
            raise VectorDBError(f"Qdrant initialization failed: {str(e)}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            return [{
                "text": hit.payload.get("text"),
                "metadata": hit.payload.get("metadata") or {},
                "score": hit.score,
                "id": hit.id
            } for hit in results]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorDBError(f"Search operation failed: {str(e)}")

    def insert(self, documents: List[Dict[str, Any]]):
        """Insert documents into the vector DB"""
        try:
            points = [
                models.PointStruct(
                    id=doc.get("id") or idx,
                    vector=doc["vector"],
                    payload={
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {})
                    }
                )
                for idx, doc in enumerate(documents)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Inserted {len(documents)} documents")
        except Exception as e:
            logger.error(f"Insert failed: {str(e)}")
            raise VectorDBError(f"Insert operation failed: {str(e)}")

    def health_check(self) -> Dict[str, Any]:
        """Check database health and collection status"""
        try:
            count = self.client.count(
                collection_name=self.collection_name
            ).count
            return {
                "status": "healthy",
                "collection_stats": {
                    "name": self.collection_name,
                    "exists": True,
                    "count": count
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }