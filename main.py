from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from vector_db import VectorDBService
from sentence_transformers import SentenceTransformer
from config import EmbeddingConfig, MilvusConfig
from pymilvus import utility
import logging
import requests
import io
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model for image embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()
logger = logging.getLogger(__name__)

# Initialize services
try:
    vector_db = VectorDBService()
    embedder = SentenceTransformer(EmbeddingConfig.MODEL_NAME, device=EmbeddingConfig.DEVICE)
except Exception as e:
    logger.error(f"Service initialization failed: {str(e)}")
    raise

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class Document(BaseModel):
    text: str
    text_embedding: List[float] = None
    image_embedding: List[float] = None
    metadata: dict = None

class SearchRequest(BaseModel):
    query: str = None
    image_url: str = None
    top_k: int = 5

@app.post("/search")
async def search(request: SearchRequest):
    try:
        if request.query:
            # Text search
            query_embedding = embedder.encode(request.query).tolist()
            results = vector_db.search(query_embedding, top_k=request.top_k, search_type="text")
        elif request.image_url:
            # Image search
            image_embedding = get_image_embedding(request.image_url)
            results = vector_db.search(image_embedding, top_k=request.top_k, search_type="image")
        else:
            raise HTTPException(status_code=400, detail="Either query or image_url must be provided")
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")

def get_image_embedding(image_url: str) -> List[float]:
    """Generate embedding for product image using CLIP"""
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features[0].tolist()
    except Exception as e:
        logger.error(f"Error generating image embedding: {e}")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")

@app.post("/documents")
async def add_document(document: Document):
    try:
        # Generate embeddings if not provided
        text_embedding = document.text_embedding or embedder.encode(document.text).tolist()
        image_embedding = document.image_embedding
        
        # Prepare document for insertion
        doc = {
            "text": document.text,
            "text_embedding": text_embedding,
            "image_embedding": image_embedding
        }
        
        # Add metadata if provided
        if document.metadata:
            doc["metadata"] = document.metadata
        
        # Insert into vector DB
        vector_db.collection.insert([doc])
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Document insertion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document insertion failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "collection_stats": {
            "name": MilvusConfig.COLLECTION_NAME,
            "exists": utility.has_collection(MilvusConfig.COLLECTION_NAME),
            "count": vector_db.collection.num_entities if utility.has_collection(MilvusConfig.COLLECTION_NAME) else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    from socket import SO_REUSEADDR, SOL_SOCKET
    
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "workers": 1,
        "socket_options": [(SOL_SOCKET, SO_REUSEADDR, 1)]
    }
    
    try:
        uvicorn.run(app, **config)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {config['port']} is in use, trying alternative port...")
            config["port"] = 8001
            uvicorn.run(app, **config)
        else:
            raise