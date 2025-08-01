"""
LLM Service
Handles OpenAI GPT-4 integration for query processing and response generation
"""

import openai
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import os

logger = logging.getLogger(__name__)

class QueryIntent(BaseModel):
    """Structured query intent"""
    intent: str
    domain: str
    filters: List[Dict[str, Any]] = []
    keywords: List[str] = []

class AnswerWithRationale(BaseModel):
    """Answer with explainable rationale"""
    answer: str
    citations: List[Dict[str, str]] = []
    rationale: List[str] = []
    confidence: float = 0.0

class LLMService:
    """Service for LLM operations using OpenAI GPT-4"""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.api_key = api_key or "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"
        self.model = model
        self.max_tokens = 2000
        self.temperature = 0.1  # Low temperature for consistent, factual responses

        # Initialize OpenAI client
        openai.api_key = self.api_key

        logger.info(f"Initialized LLM service with model: {model}")

    async def parse_query_intent(self, query: str) -> QueryIntent:
        """
        Parse user query into structured intent

        Args:
            query: Natural language query

        Returns:
            QueryIntent object with structured information
        """
        system_prompt = """You are an expert at analyzing queries for document retrieval systems.
Parse the user query and return a JSON object with the following structure:
{
    "intent": "question_type (e.g., coverage_query, claim_process, definitions, exclusions)",
    "domain": "document_domain (e.g., insurance, legal, hr, compliance)",
    "filters": [{"field": "filter_name", "value": "filter_value"}],
    "keywords": ["key", "terms", "to", "search"]
}

Focus on insurance, legal, HR, and compliance domains. Extract key terms that would be useful for semantic search."""

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this query: {query}"}
                ],
                max_tokens=500,
                temperature=self.temperature
            )

            result = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                parsed_result = json.loads(result)
                return QueryIntent(**parsed_result)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return QueryIntent(
                    intent="general_query",
                    domain="general",
                    keywords=query.split()[:10]  # Use first 10 words as keywords
                )

        except Exception as e:
            logger.error(f"Error parsing query intent: {e}")
            # Return basic intent
            return QueryIntent(
                intent="general_query",
                domain="general",
                keywords=query.split()[:10]
            )

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], 
                            include_rationale: bool = False) -> AnswerWithRationale:
        """
        Generate answer based on query and retrieved context

        Args:
            query: User query
            context_chunks: Retrieved document chunks with metadata
            include_rationale: Whether to include step-by-step reasoning

        Returns:
            AnswerWithRationale object
        """
        # Prepare context
        context_text = ""
        citations = []

        for i, chunk in enumerate(context_chunks[:5]):  # Use top 5 chunks
            chunk_text = chunk.get("text", "")
            source_meta = chunk.get("source_meta", {})
            similarity_score = chunk.get("similarity_score", 0.0)

            context_text += f"\n[Context {i+1}] (Score: {similarity_score:.2f})\n{chunk_text}\n"

            # Create citation
            citation = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "source": source_meta.get("source_url", "unknown"),
                "page": str(source_meta.get("page", "unknown")),
                "score": f"{similarity_score:.2f}"
            }
            citations.append(citation)

        # System prompt for answer generation
        system_prompt = """You are an expert assistant specializing in insurance, legal, HR, and compliance document analysis.

Your task is to provide accurate, helpful answers based solely on the provided context. Follow these guidelines:

1. Answer only based on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific and cite relevant details from the context
4. For insurance/legal queries, be precise about terms, conditions, and limitations
5. If asked about specific clauses, quote them directly when possible
6. Maintain a professional, helpful tone"""

        user_prompt = f"""Context from retrieved documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""

        if include_rationale:
            user_prompt += """

Also provide:
1. Step-by-step reasoning for your answer
2. Which parts of the context were most relevant
3. Your confidence level in the answer (0-1 scale)"""

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            answer_text = response.choices[0].message.content.strip()

            # Extract rationale if requested
            rationale = []
            confidence = 0.8  # Default confidence

            if include_rationale:
                # Try to extract rationale from the response
                # This is a simple approach - could be improved with better parsing
                if "reasoning:" in answer_text.lower() or "step" in answer_text.lower():
                    lines = answer_text.split("\n")
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ["step", "reasoning", "because", "based on"]):
                            rationale.append(line.strip())

                # Extract confidence if mentioned
                if "confidence" in answer_text.lower():
                    import re
                    conf_match = re.search(r'confidence[^\d]*([0-9.]+)', answer_text.lower())
                    if conf_match:
                        confidence = float(conf_match.group(1))
                        if confidence > 1:
                            confidence = confidence / 100  # Convert percentage

            return AnswerWithRationale(
                answer=answer_text,
                citations=citations,
                rationale=rationale,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return AnswerWithRationale(
                answer=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                citations=citations,
                rationale=["Error occurred during processing"],
                confidence=0.0
            )

    async def generate_batch_answers(self, queries: List[str], 
                                   context_chunks_list: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Generate answers for multiple queries efficiently

        Args:
            queries: List of user queries
            context_chunks_list: List of context chunks for each query

        Returns:
            List of answers
        """
        tasks = []

        for query, context_chunks in zip(queries, context_chunks_list):
            task = self.generate_answer(query, context_chunks, include_rationale=False)
            tasks.append(task)

        # Process all queries concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            answers = []
            for result in results:
                if isinstance(result, Exception):
                    answers.append(f"Error processing query: {str(result)}")
                else:
                    answers.append(result.answer)

            return answers

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [f"Error processing queries: {str(e)}"] * len(queries)

    async def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """
        Extract key phrases from text for better search

        Args:
            text: Input text
            max_phrases: Maximum number of phrases to return

        Returns:
            List of key phrases
        """
        system_prompt = f"""Extract the {max_phrases} most important key phrases from the following text.
Return them as a JSON list of strings. Focus on:
- Technical terms
- Important concepts
- Domain-specific terminology
- Key entities (names, dates, amounts)

Example: ["insurance coverage", "waiting period", "pre-existing conditions"]"""

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=300,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()

            try:
                phrases = json.loads(result)
                return phrases[:max_phrases]
            except json.JSONDecodeError:
                # Fallback: split by common delimiters
                phrases = [phrase.strip().strip('\"') for phrase in result.split(',')]
                return phrases[:max_phrases]

        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []

    def validate_api_key(self) -> bool:
        """Validate OpenAI API key"""
        try:
            # Make a simple API call to test the key
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="Test",
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key_valid": self.validate_api_key()
        }
