"""
Configuration settings for the LLM Query-Retrieval System
"""

import os
from typing import Optional

class Config:
    """Application configuration"""

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

    # Authentication
    BEARER_TOKEN: str = os.getenv("BEARER_TOKEN", "c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798")

    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "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")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    USE_PINECONE: bool = os.getenv("USE_PINECONE", "false").lower() == "true"
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "llm-query-retrieval")

    # Processing Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    # Performance Settings
    MAX_CONCURRENT_QUERIES: int = int(os.getenv("MAX_CONCURRENT_QUERIES", "10"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes

    # Storage Settings
    INDEX_STORAGE_PATH: str = os.getenv("INDEX_STORAGE_PATH", "./data/indexes")
    TEMP_STORAGE_PATH: str = os.getenv("TEMP_STORAGE_PATH", "./data/temp")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        required_settings = [
            "OPENAI_API_KEY",
            "BEARER_TOKEN"
        ]

        for setting in required_settings:
            if not getattr(cls, setting):
                print(f"Warning: {setting} is not set")
                return False

        return True

# Global config instance
config = Config()
