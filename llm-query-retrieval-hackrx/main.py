"""
LLM-Powered Intelligent Query-Retrieval System
FastAPI Application Entry Point
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.api.endpoints import hackrx_router
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
document_processor = None
embedding_service = None
llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global document_processor, embedding_service, llm_service

    # Startup
    logger.info("Starting up LLM Query-Retrieval System...")
    try:
        document_processor = DocumentProcessor()
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        logger.info("Services initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")
    # Clean up resources if needed

# Create FastAPI app
app = FastAPI(
    title="LLM Query-Retrieval System",
    description="Intelligent document processing and query system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Authentication dependency
async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate Bearer token"""
    expected_token = "c0df38f44acb385ecd42f8e0c02ee14acd6d145835643ee57acd84f79afeb798"

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Include routers
app.include_router(
    hackrx_router,
    prefix="/api/v1",
    dependencies=[Depends(authenticate)]
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "LLM Query-Retrieval System is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "document_processor": document_processor is not None,
            "embedding_service": embedding_service is not None,
            "llm_service": llm_service is not None,
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
