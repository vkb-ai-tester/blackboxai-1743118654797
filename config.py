import configparser
import os
from pathlib import Path

config = configparser.ConfigParser()
config.read(Path(__file__).parent / 'config.ini')

class MilvusConfig:
    URI = config.get('milvus_cloud', 'uri')
    API_KEY = config.get('milvus_cloud', 'api_key')
    PORT = config.get('milvus_cloud', 'port')
    COLLECTION_NAME = config.get('milvus_cloud', 'collection_name')
    VECTOR_DIMENSION = config.getint('milvus_cloud', 'vector_dimension')
    
    INDEX_PARAMS = {
        "metric_type": config.get('index_params', 'metric_type'),
        "index_type": config.get('index_params', 'index_type'),
        "params": {"nlist": config.getint('index_params', 'nlist')}
    }
    
    SEARCH_PARAMS = {
        "metric_type": config.get('search_params', 'metric_type'),
        "params": {"nprobe": config.getint('search_params', 'nprobe')}
    }

class EmbeddingConfig:
    MODEL_NAME = config.get('embedding', 'model_name')
    BATCH_SIZE = config.getint('embedding', 'batch_size')
    DEVICE = config.get('embedding', 'device')