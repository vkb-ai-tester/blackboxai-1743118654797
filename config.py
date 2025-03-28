import configparser
import os
from pathlib import Path
from typing import Dict, Any
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

def load_config() -> configparser.ConfigParser:
    """Load configuration with environment variable fallback"""
    config = configparser.ConfigParser()
    config_file = Path(__file__).parent / 'config.ini'
    
    if not config_file.exists():
        logger.warning("Config file not found at %s, using environment variables", config_file)
        return config
    
    try:
        config.read(config_file)
        logger.info("Loaded configuration from %s", config_file)
    except Exception as e:
        raise ConfigError(f"Failed to read config file: {str(e)}")
    
    return config

config = load_config()

class ConfigValidator:
    @staticmethod
    def validate_section(config: configparser.ConfigParser, section: str, required_keys: list):
        """Validate a config section exists and has all required keys"""
        if not config.has_section(section):
            raise ValueError(f"Missing config section: {section}")
        
        missing_keys = []
        for key in required_keys:
            if not config.has_option(section, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing config keys in {section}: {', '.join(missing_keys)}")

    @staticmethod
    def get_with_fallback(config: configparser.ConfigParser, section: str, key: str, env_var: str = None):
        """Get config value with environment variable fallback"""
        try:
            if config.has_option(section, key):
                return config.get(section, key)
            if env_var and os.getenv(env_var):
                return os.getenv(env_var)
            raise ValueError(f"Missing value for {section}.{key}")
        except Exception as e:
            raise ConfigError(f"Config error: {str(e)}")

class MilvusConfig:
    REQUIRED_KEYS = ['uri', 'api_key', 'collection_name', 'vector_dimension']
    
    try:
        ConfigValidator.validate_section(config, 'milvus_cloud', REQUIRED_KEYS)
        
        URI = ConfigValidator.get_with_fallback(config, 'milvus_cloud', 'uri', 'MILVUS_URI')
        API_KEY = ConfigValidator.get_with_fallback(config, 'milvus_cloud', 'api_key', 'MILVUS_API_KEY')
        PORT = ConfigValidator.get_with_fallback(config, 'milvus_cloud', 'port', 'MILVUS_PORT') or "443"
        COLLECTION_NAME = ConfigValidator.get_with_fallback(config, 'milvus_cloud', 'collection_name')
        VECTOR_DIMENSION = int(ConfigValidator.get_with_fallback(config, 'milvus_cloud', 'vector_dimension'))
    except Exception as e:
        logger.error("Milvus configuration error: %s", str(e))
        raise ConfigError(f"Invalid Milvus configuration: {str(e)}")

    @classmethod
    def get_index_params(cls) -> Dict[str, Any]:
        """Get index parameters with validation"""
        try:
            ConfigValidator.validate_section(config, 'index_params', ['metric_type', 'index_type', 'nlist'])
            return {
                "metric_type": config.get('index_params', 'metric_type'),
                "index_type": config.get('index_params', 'index_type'),
                "params": {"nlist": config.getint('index_params', 'nlist')}
            }
        except Exception as e:
            raise ConfigError(f"Invalid index params: {str(e)}")

    @classmethod
    def get_search_params(cls) -> Dict[str, Any]:
        """Get search parameters with validation"""
        try:
            ConfigValidator.validate_section(config, 'search_params', ['metric_type', 'nprobe'])
            return {
                "metric_type": config.get('search_params', 'metric_type'),
                "params": {"nprobe": config.getint('search_params', 'nprobe')}
            }
        except Exception as e:
            raise ConfigError(f"Invalid search params: {str(e)}")

class EmbeddingConfig:
    REQUIRED_KEYS = ['model_name', 'batch_size', 'device']
    
    try:
        ConfigValidator.validate_section(config, 'embedding', REQUIRED_KEYS)
        
        MODEL_NAME = ConfigValidator.get_with_fallback(config, 'embedding', 'model_name', 'EMBEDDING_MODEL')
        BATCH_SIZE = int(ConfigValidator.get_with_fallback(config, 'embedding', 'batch_size', 'EMBEDDING_BATCH_SIZE'))
        DEVICE = ConfigValidator.get_with_fallback(config, 'embedding', 'device', 'EMBEDDING_DEVICE') or "cpu"
    except Exception as e:
        logger.error("Embedding configuration error: %s", str(e))
        raise ConfigError(f"Invalid embedding configuration: {str(e)}")
