"""
Service layer tests
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

class TestDocumentProcessor:
    """Test document processing service"""

    def test_initialization(self):
        """Test service initialization"""
        processor = DocumentProcessor()
        assert processor is not None
        assert processor.chunk_size == 1000
        assert processor.overlap == 200

    def test_clean_text(self):
        """Test text cleaning functionality"""
        processor = DocumentProcessor()
        dirty_text = "This   is\ta  test\n\nwith   extra\tspaces."
        clean_text = processor._clean_text(dirty_text)
        assert "  " not in clean_text  # No double spaces
        assert "\t" not in clean_text  # No tabs

class TestEmbeddingService:
    """Test embedding service"""

    def test_initialization(self):
        """Test service initialization"""
        service = EmbeddingService()
        assert service is not None
        assert service.embedding_dim > 0

    def test_embed_texts(self):
        """Test text embedding"""
        service = EmbeddingService()
        texts = ["This is a test document", "Another test sentence"]
        embeddings = service.embed_texts(texts)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == service.embedding_dim

class TestLLMService:
    """Test LLM service"""

    def test_initialization(self):
        """Test service initialization"""
        service = LLMService()
        assert service is not None
        assert service.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_query_intent_parsing(self):
        """Test query intent parsing"""
        service = LLMService()

        # Mock the OpenAI response
        with patch('openai.ChatCompletion.acreate') as mock_openai:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"intent": "test", "domain": "general", "keywords": ["test"]}'
            mock_openai.return_value = mock_response

            intent = await service.parse_query_intent("What is the grace period?")
            assert intent.intent == "test"
            assert intent.domain == "general"

if __name__ == "__main__":
    print("Running service tests...")
    # Run basic tests
    processor_test = TestDocumentProcessor()
    processor_test.test_initialization()
    processor_test.test_clean_text()

    embedding_test = TestEmbeddingService()
    embedding_test.test_initialization()

    print("Service tests completed!")
