from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from vector_db import VectorDBService
from sentence_transformers import SentenceTransformer
from config import EmbeddingConfig
import logging

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
    metadata: dict = None

@app.post("/search")
async def search(request: SearchRequest):
    try:
        # Generate embedding
        query_embedding = embedder.encode(request.query).tolist()
        
        # Search vectors
        results = vector_db.search(query_embedding, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/documents")
async def add_document(document: Document):
    try:
        # Generate embedding
        embedding = embedder.encode(document.text).tolist()
        
        # Insert into vector DB
        vector_db.collection.insert([{"text": document.text, "embedding": embedding}])
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Document insertion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Document insertion failed")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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