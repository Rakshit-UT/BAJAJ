"""
Query Processor
Orchestrates the entire pipeline: document processing, embedding search, and LLM responses
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from app.services.document_processor import DocumentProcessor, DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService, QueryIntent, AnswerWithRationale

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Main orchestrator for the query-retrieval pipeline"""

    def __init__(self, document_processor: DocumentProcessor = None,
                 embedding_service: EmbeddingService = None,
                 llm_service: LLMService = None):

        self.doc_processor = document_processor or DocumentProcessor()
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()

        # Performance settings
        self.max_context_chunks = 5
        self.similarity_threshold = 0.5
        self.max_concurrent_queries = 10

        logger.info("QueryProcessor initialized")

    async def process_request(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Process a request with document URL and questions

        Args:
            document_url: URL to document to process
            questions: List of questions to answer

        Returns:
            List of answers
        """
        start_time = time.time()
        logger.info(f"Processing request with {len(questions)} questions")

        try:
            # Step 1: Process document
            chunks = await self._process_document(document_url)

            # Step 2: Index chunks
            await self._index_chunks(chunks)

            # Step 3: Process questions
            answers = await self._process_questions(questions)

            end_time = time.time()
            logger.info(f"Request processed successfully in {end_time - start_time:.2f} seconds")

            return answers

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise Exception(f"Failed to process request: {str(e)}")

    async def process_request_advanced(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Process request with advanced features (citations, rationale)

        Args:
            document_url: URL to document to process
            questions: List of questions to answer

        Returns:
            Dictionary with answers, citations, and rationale
        """
        start_time = time.time()
        logger.info(f"Processing advanced request with {len(questions)} questions")

        try:
            # Step 1: Process document
            chunks = await self._process_document(document_url)

            # Step 2: Index chunks
            await self._index_chunks(chunks)

            # Step 3: Process questions with full context
            results = await self._process_questions_advanced(questions)

            end_time = time.time()
            logger.info(f"Advanced request processed successfully in {end_time - start_time:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"Error processing advanced request: {e}")
            raise Exception(f"Failed to process advanced request: {str(e)}")

    async def _process_document(self, document_url: str) -> List[DocumentChunk]:
        """Process document from URL"""
        logger.info(f"Processing document: {document_url}")

        try:
            # Process document into chunks
            chunks = await self.doc_processor.process_document_from_url(document_url)

            # Enhance with clause extraction
            enhanced_chunks = self.doc_processor.extract_clauses(chunks)

            logger.info(f"Document processed into {len(enhanced_chunks)} chunks")
            return enhanced_chunks

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    async def _index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks for similarity search"""
        logger.info(f"Indexing {len(chunks)} chunks")

        try:
            # Clear existing index
            self.embedding_service.clear_index()

            # Index new chunks
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    self.embedding_service.index_chunks,
                    chunks
                )

            logger.info("Chunks indexed successfully")

        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise

    async def _process_questions(self, questions: List[str]) -> List[str]:
        """Process multiple questions efficiently"""
        logger.info(f"Processing {len(questions)} questions")

        try:
            # Parse intents for all questions
            intents = await self._parse_intents(questions)

            # Retrieve context for all questions
            contexts = await self._retrieve_contexts(questions, intents)

            # Generate answers
            answers = await self.llm_service.generate_batch_answers(questions, contexts)

            return answers

        except Exception as e:
            logger.error(f"Error processing questions: {e}")
            raise

    async def _process_questions_advanced(self, questions: List[str]) -> Dict[str, Any]:
        """Process questions with advanced features"""
        logger.info(f"Processing {len(questions)} questions with advanced features")

        try:
            # Parse intents
            intents = await self._parse_intents(questions)

            # Retrieve context
            contexts = await self._retrieve_contexts(questions, intents)

            # Generate detailed answers with rationale
            detailed_answers = []
            all_citations = []
            all_rationale = []

            for question, context in zip(questions, contexts):
                result = await self.llm_service.generate_answer(
                    question, context, include_rationale=True
                )
                detailed_answers.append(result.answer)
                all_citations.append(result.citations)
                all_rationale.append(result.rationale)

            return {
                "answers": detailed_answers,
                "citations": all_citations,
                "rationale": all_rationale
            }

        except Exception as e:
            logger.error(f"Error processing advanced questions: {e}")
            raise

    async def _parse_intents(self, questions: List[str]) -> List[QueryIntent]:
        """Parse intents for all questions"""
        tasks = []

        for question in questions:
            task = self.llm_service.parse_query_intent(question)
            tasks.append(task)

        try:
            intents = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_intents = []
            for intent in intents:
                if isinstance(intent, Exception):
                    logger.warning(f"Failed to parse intent: {intent}")
                    # Use default intent
                    processed_intents.append(QueryIntent(
                        intent="general_query",
                        domain="general"
                    ))
                else:
                    processed_intents.append(intent)

            return processed_intents

        except Exception as e:
            logger.error(f"Error parsing intents: {e}")
            raise

    async def _retrieve_contexts(self, questions: List[str], 
                                intents: List[QueryIntent]) -> List[List[Dict[str, Any]]]:
        """Retrieve context for all questions"""
        contexts = []

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            tasks = []

            for question, intent in zip(questions, intents):
                task = loop.run_in_executor(
                    executor,
                    self._retrieve_context_for_question,
                    question,
                    intent
                )
                tasks.append(task)

            try:
                contexts = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions
                processed_contexts = []
                for context in contexts:
                    if isinstance(context, Exception):
                        logger.warning(f"Failed to retrieve context: {context}")
                        processed_contexts.append([])  # Empty context
                    else:
                        processed_contexts.append(context)

                return processed_contexts

            except Exception as e:
                logger.error(f"Error retrieving contexts: {e}")
                raise

    def _retrieve_context_for_question(self, question: str, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Retrieve context for a single question"""
        try:
            # Determine search strategy based on intent
            if intent.domain in ["insurance", "legal"] and hasattr(intent, 'clause_types'):
                # Use clause-specific search
                context = self.embedding_service.get_clause_matches(
                    question,
                    clause_types=getattr(intent, 'clause_types', []),
                    top_k=self.max_context_chunks,
                    score_threshold=self.similarity_threshold
                )
            else:
                # Use general semantic search
                filters = {}
                if intent.filters:
                    for filter_item in intent.filters:
                        filters[filter_item.get("field")] = filter_item.get("value")

                context = self.embedding_service.search_with_filters(
                    question,
                    filters=filters if filters else None,
                    top_k=self.max_context_chunks,
                    score_threshold=self.similarity_threshold
                )

            return context

        except Exception as e:
            logger.error(f"Error retrieving context for question: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        health = {
            "query_processor": True,
            "document_processor": True,
            "embedding_service": False,
            "llm_service": False,
            "timestamp": time.time()
        }

        try:
            # Check embedding service
            stats = self.embedding_service.get_stats()
            health["embedding_service"] = stats.get("is_indexed", False)
            health["embedding_stats"] = stats

        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")

        try:
            # Check LLM service
            model_info = self.llm_service.get_model_info()
            health["llm_service"] = model_info.get("api_key_valid", False)
            health["llm_model_info"] = model_info

        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")

        return health

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "max_context_chunks": self.max_context_chunks,
            "similarity_threshold": self.similarity_threshold,
            "max_concurrent_queries": self.max_concurrent_queries,
            "embedding_stats": self.embedding_service.get_stats()
        }

    def update_settings(self, **kwargs) -> None:
        """Update processor settings"""
        if "max_context_chunks" in kwargs:
            self.max_context_chunks = kwargs["max_context_chunks"]
        if "similarity_threshold" in kwargs:
            self.similarity_threshold = kwargs["similarity_threshold"]
        if "max_concurrent_queries" in kwargs:
            self.max_concurrent_queries = kwargs["max_concurrent_queries"]

        logger.info(f"Updated settings: {kwargs}")
