"""
API Endpoints for LLM Query-Retrieval System
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

hackrx_router = APIRouter()

# Request/Response Models
class HackRxRequest(BaseModel):
    """Request model for /hackrx/run endpoint"""
    documents: str  # URL to document
    questions: List[str]

class HackRxResponse(BaseModel):
    """Response model for /hackrx/run endpoint"""
    answers: List[str]

class AdvancedHackRxResponse(BaseModel):
    """Extended response model with citations and rationale"""
    answers: List[str]
    citations: Optional[List[Dict[str, Any]]] = None
    rationale: Optional[List[List[str]]] = None

@hackrx_router.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(request: HackRxRequest) -> HackRxResponse:
    """
    Main endpoint to process documents and answer questions

    Args:
        request: Contains document URL and list of questions

    Returns:
        HackRxResponse with answers for each question
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")

        # Initialize services
        doc_processor = DocumentProcessor()
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        query_processor = QueryProcessor(doc_processor, embedding_service, llm_service)

        # Process the request
        answers = await query_processor.process_request(
            document_url=request.documents,
            questions=request.questions
        )

        logger.info(f"Successfully processed {len(answers)} answers")

        return HackRxResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )

@hackrx_router.post("/hackrx/run-advanced", response_model=AdvancedHackRxResponse)
async def run_hackrx_advanced(request: HackRxRequest) -> AdvancedHackRxResponse:
    """
    Advanced endpoint with citations and explainable rationale

    Args:
        request: Contains document URL and list of questions

    Returns:
        AdvancedHackRxResponse with answers, citations, and rationale
    """
    try:
        logger.info(f"Processing advanced request with {len(request.questions)} questions")

        # Initialize services
        doc_processor = DocumentProcessor()
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        query_processor = QueryProcessor(doc_processor, embedding_service, llm_service)

        # Process the request with full context
        result = await query_processor.process_request_advanced(
            document_url=request.documents,
            questions=request.questions
        )

        logger.info(f"Successfully processed advanced request")

        return AdvancedHackRxResponse(
            answers=result["answers"],
            citations=result.get("citations"),
            rationale=result.get("rationale")
        )

    except Exception as e:
        logger.error(f"Error processing advanced request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process advanced request: {str(e)}"
        )

@hackrx_router.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/hackrx/run",
            "/api/v1/hackrx/run-advanced",
            "/api/v1/status"
        ]
    }
