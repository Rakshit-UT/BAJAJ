#!/usr/bin/env python3
"""
Startup script for LLM Query-Retrieval System
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup environment
from dotenv import load_dotenv
load_dotenv()

from app.utils.logger import setup_logging
from app.utils.config import config

def main():
    """Main startup function"""
    print("="*50)
    print("LLM Query-Retrieval System")
    print("="*50)

    # Setup logging
    setup_logging(level=config.LOG_LEVEL)

    # Validate configuration
    if not config.validate():
        print("Configuration validation failed!")
        sys.exit(1)

    print(f"Starting server on {config.API_HOST}:{config.API_PORT}")
    print(f"Using OpenAI model: {config.OPENAI_MODEL}")
    print(f"Using embedding model: {config.EMBEDDING_MODEL}")

    # Create data directories
    os.makedirs(config.INDEX_STORAGE_PATH, exist_ok=True)
    os.makedirs(config.TEMP_STORAGE_PATH, exist_ok=True)

    # Start the server
    try:
        uvicorn.run(
            "main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            workers=config.API_WORKERS,
            reload=False,
            log_level=config.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
