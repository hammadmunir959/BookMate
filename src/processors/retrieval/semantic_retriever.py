"""
Semantic Retriever for RAG Retrieval Microservice
Handles vector-based semantic search using ChromaDB
"""

import logging
from typing import List, Dict, Optional, Any
import time

from src.core.config import config
from src.core.document_models import RetrievalQuery, RetrievalResult, QueryType
from src.storage.vector_store import chroma_db

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic retriever using vector similarity search"""
    
    def __init__(self):
        """Initialize semantic retriever"""
        self.vector_store = chroma_db
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load embedding model for query encoding with proper caching"""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = config.model.embedding_model
            # Sanitize model name to create a valid filename
            sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
            cache_key = f"embedding_model_{sanitized_model_name}"

            # Try to load from custom cache first
            cached_model = self._load_model_from_cache(cache_key)
            if cached_model:
                self.embedding_model = cached_model
                logger.info("Embedding model loaded from cache")
                return

            # Load fresh model
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

            # Cache the model for future use
            self._save_model_to_cache(cache_key, self.embedding_model)
            logger.info("Embedding model loaded and cached successfully")

        except ImportError:
            logger.warning("SentenceTransformers not available, semantic search will be limited")
            self.embedding_model = None
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            self.embedding_model = None

    def _load_model_from_cache(self, cache_key: str):
        """Load model from custom cache (not using JSON)"""
        try:
            import pickle
            from pathlib import Path

            cache_dir = Path(config.cache.embedding_cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{cache_key}.pkl"

            if cache_file.exists():
                logger.info(f"Loading cached model from {cache_file}")
                with open(cache_file, 'rb') as f:
                    model = pickle.load(f)
                logger.info("Model loaded from cache successfully")
                return model

        except Exception as e:
            logger.warning(f"Failed to load cached model: {str(e)}")
            return None

    def _save_model_to_cache(self, cache_key: str, model):
        """Save model to custom cache using pickle"""
        try:
            import pickle
            from pathlib import Path

            # Use the verified cache path from config
            cache_dir = Path(config.cache.embedding_cache_path)

            # Double-check directory exists (belt and suspenders approach)
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created cache directory: {cache_dir}")

            cache_file = cache_dir / f"{cache_key}.pkl"

            # Test write permissions before attempting to save
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model cached successfully: {cache_file}")
            except PermissionError:
                logger.warning(f"Permission denied caching model to {cache_file}")
                # Try to save to a fallback location
                fallback_file = cache_dir / f"{cache_key}_fallback.pkl"
                with open(fallback_file, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model cached to fallback location: {fallback_file}")

        except Exception as e:
            logger.warning(f"Failed to cache model: {str(e)}")
            logger.debug(f"Cache directory: {config.cache.embedding_cache_path}")
            logger.debug(f"Cache key: {cache_key}")
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform semantic retrieval
        
        Args:
            query: RetrievalQuery object
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            start_time = time.time()
            
            # Check if semantic search is enabled
            if not config.retrieval.enable_semantic_search:
                logger.info("Semantic search disabled, returning empty results")
                return []
            
            # Generate query embedding
            if not self.embedding_model:
                logger.error("Embedding model not loaded, cannot perform semantic search")
                return []
            
            query_embedding = self.embedding_model.encode(query.query).tolist()
            
            # Perform semantic search with document filtering
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=query.top_k,
                similarity_threshold=query.similarity_threshold,
                document_filters=query.filters if hasattr(query, 'filters') else None
            )
            
            # Update retrieval method in results
            for result in results:
                result.retrieval_method = "semantic"
            
            processing_time = time.time() - start_time
            logger.info(f"Semantic retrieval completed in {processing_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            return []
    
    def retrieve_with_expansion(self, query: RetrievalQuery, expanded_queries: List[str]) -> List[RetrievalResult]:
        """
        Perform semantic retrieval with query expansion
        
        Args:
            query: Original RetrievalQuery object
            expanded_queries: List of expanded query variants
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            start_time = time.time()
            all_results = []
            seen_chunk_ids = set()
            
            # Search with original query
            original_results = self.retrieve(query)
            for result in original_results:
                if result.chunk_id not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result.chunk_id)
            
            # Search with expanded queries
            for expanded_query in expanded_queries[:3]:  # Limit to top 3 expansions
                try:
                    # Create temporary query object for expanded query
                    expanded_query_obj = RetrievalQuery(
                        query=expanded_query,
                        query_type=query.query_type,
                        filters=query.filters,
                        top_k=max(3, query.top_k // 2),  # Fewer results per expansion
                        similarity_threshold=query.similarity_threshold,
                        enable_reranking=query.enable_reranking,
                        enable_query_expansion=False  # Prevent recursive expansion
                    )
                    
                    expansion_results = self.retrieve(expanded_query_obj)
                    
                    # Add unique results
                    for result in expansion_results:
                        if result.chunk_id not in seen_chunk_ids:
                            # Reduce score for expansion results
                            result.semantic_score *= 0.8
                            result.final_score = result.semantic_score
                            result.retrieval_method = "semantic_expanded"
                            all_results.append(result)
                            seen_chunk_ids.add(result.chunk_id)
                            
                except Exception as e:
                    logger.warning(f"Error with expanded query '{expanded_query}': {str(e)}")
                    continue
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x.semantic_score, reverse=True)
            
            # Limit to requested top_k
            final_results = all_results[:query.top_k]
            
            processing_time = time.time() - start_time
            logger.info(f"Semantic retrieval with expansion completed in {processing_time:.3f}s, found {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval with expansion: {str(e)}")
            return self.retrieve(query)  # Fallback to basic retrieval
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Get chunks similar to a specific chunk
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Get the reference chunk
            reference_chunk = self.vector_store.get_chunk_by_id(chunk_id)
            if not reference_chunk:
                logger.warning(f"Chunk {chunk_id} not found")
                return []
            
            # Use the chunk content as query
            query = RetrievalQuery(
                query=reference_chunk.content,
                query_type=QueryType.SEMANTIC,
                top_k=top_k + 1,  # +1 to exclude the reference chunk itself
                similarity_threshold=0.3
            )
            
            # Perform semantic search
            results = self.retrieve(query)
            
            # Filter out the reference chunk itself
            similar_results = [r for r in results if r.chunk_id != chunk_id]
            
            logger.info(f"Found {len(similar_results)} chunks similar to {chunk_id}")
            return similar_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting similar chunks for {chunk_id}: {str(e)}")
            return []
    
    def get_document_chunks(self, document_id: str, limit: int = 100) -> List[RetrievalResult]:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document ID
            limit: Maximum number of chunks to return
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            results = self.vector_store.get_document_chunks(document_id, limit)
            
            # Update retrieval method
            for result in results:
                result.retrieval_method = "document_chunks"
            
            logger.info(f"Retrieved {len(results)} chunks for document {document_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting document chunks for {document_id}: {str(e)}")
            return []
    
    def test_connection(self) -> bool:
        """Test connection to vector store"""
        try:
            return self.vector_store.test_connection()
        except Exception as e:
            logger.error(f"Error testing semantic retriever connection: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic retriever"""
        try:
            vector_stats = self.vector_store.get_document_stats()
            
            return {
                "type": "semantic",
                "embedding_model": config.model.embedding_model,
                "embedding_device": config.model.embedding_device,
                "model_loaded": self.embedding_model is not None,
                "vector_store_stats": vector_stats,
                "enabled": config.retrieval.enable_semantic_search
            }
            
        except Exception as e:
            logger.error(f"Error getting semantic retriever stats: {str(e)}")
            return {
                "type": "semantic",
                "error": str(e),
                "enabled": config.retrieval.enable_semantic_search
            }

