"""
Embedding Service
Handles text embeddings and vector similarity search using FAISS/Pinecone
"""

import numpy as np
import faiss
import logging
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import pickle
import os
from pathlib import Path
import json

from app.services.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for handling embeddings and vector search"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_pinecone: bool = False):
        self.model_name = model_name
        self.use_pinecone = use_pinecone
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

        # Initialize embedding model
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize vector store
        self.index = None
        self.chunks_metadata = []  # Store chunk metadata
        self.is_indexed = False

        if use_pinecone:
            self._init_pinecone()
        else:
            self._init_faiss()

    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            # Use IndexFlatIP for cosine similarity (inner product)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("Initialized FAISS index")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise

    def _init_pinecone(self):
        """Initialize Pinecone (placeholder for now)"""
        # TODO: Implement Pinecone integration
        logger.info("Pinecone integration not implemented yet, falling back to FAISS")
        self._init_faiss()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings
        """
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Generate embeddings for document chunks

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Index document chunks for similarity search

        Args:
            chunks: List of DocumentChunk objects to index
        """
        try:
            logger.info(f"Indexing {len(chunks)} chunks...")

            # Generate embeddings
            embeddings = self.embed_chunks(chunks)

            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))

            # Store metadata
            self.chunks_metadata.extend([chunk.to_dict() for chunk in chunks])

            self.is_indexed = True
            logger.info(f"Successfully indexed {len(chunks)} chunks. Total indexed: {len(self.chunks_metadata)}")

        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            raise

    def search_similar(self, query: str, top_k: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity

        Args:
            query: Query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of search results with chunks and scores
        """
        if not self.is_indexed:
            logger.warning("No chunks indexed yet")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embed_texts([query])

            # Search in FAISS index
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= score_threshold:  # Valid result above threshold
                    chunk_data = self.chunks_metadata[idx].copy()
                    chunk_data["similarity_score"] = float(score)
                    results.append(chunk_data)

            logger.info(f"Found {len(results)} similar chunks for query")
            return results

        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            raise

    def search_with_filters(self, query: str, filters: Dict[str, Any] = None, 
                          top_k: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search with metadata filters

        Args:
            query: Query text
            filters: Dictionary of metadata filters
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of filtered search results
        """
        # Get all similar results first
        results = self.search_similar(query, top_k * 2, score_threshold)  # Get more results for filtering

        if not filters:
            return results[:top_k]

        # Apply filters
        filtered_results = []
        for result in results:
            match = True
            source_meta = result.get("source_meta", {})

            for filter_key, filter_value in filters.items():
                if filter_key not in source_meta:
                    match = False
                    break

                # Handle different filter types
                if isinstance(filter_value, list):
                    if source_meta[filter_key] not in filter_value:
                        match = False
                        break
                elif source_meta[filter_key] != filter_value:
                    match = False
                    break

            if match:
                filtered_results.append(result)

            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def get_clause_matches(self, query: str, clause_types: List[str] = None, 
                          top_k: int = 5, score_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Search for specific clause types

        Args:
            query: Query text
            clause_types: List of clause types to filter for
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of clause matches
        """
        results = self.search_similar(query, top_k * 2, score_threshold)

        if not clause_types:
            return results[:top_k]

        # Filter by clause types
        clause_matches = []
        for result in results:
            source_meta = result.get("source_meta", {})
            result_clause_types = source_meta.get("clause_types", [])

            # Check if any of the result's clause types match the requested ones
            if any(ct in result_clause_types for ct in clause_types):
                clause_matches.append(result)

            if len(clause_matches) >= top_k:
                break

        return clause_matches

    def save_index(self, filepath: str) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.index")

            # Save metadata
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump(self.chunks_metadata, f)

            # Save config
            config = {
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "num_chunks": len(self.chunks_metadata)
            }
            with open(f"{filepath}.config", 'w') as f:
                json.dump(config, f)

            logger.info(f"Saved index to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, filepath: str) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            # Load config
            with open(f"{filepath}.config", 'r') as f:
                config = json.load(f)

            # Verify model compatibility
            if config["model_name"] != self.model_name:
                logger.warning(f"Model mismatch: saved {config['model_name']}, current {self.model_name}")

            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")

            # Load metadata
            with open(f"{filepath}.metadata", 'rb') as f:
                self.chunks_metadata = pickle.load(f)

            self.is_indexed = len(self.chunks_metadata) > 0
            logger.info(f"Loaded index with {len(self.chunks_metadata)} chunks")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def clear_index(self) -> None:
        """Clear the current index"""
        self.index.reset()
        self.chunks_metadata = []
        self.is_indexed = False
        logger.info("Cleared index")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "is_indexed": self.is_indexed,
            "num_chunks": len(self.chunks_metadata),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "index_size": self.index.ntotal if self.index else 0
        }
