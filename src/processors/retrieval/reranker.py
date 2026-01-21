"""
Reranker for RAG Retrieval Microservice
Uses Cross-Encoders to re-rank retrieval results for higher precision
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import time

from src.core.config import config
from src.core.document_models import RetrievalQuery, RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker using Cross-Encoder models"""

    def __init__(self):
        """Initialize reranker"""
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Cross-Encoder model with caching"""
        try:
            from sentence_transformers import CrossEncoder
            
            model_name = config.model.reranker_model
            logger.info(f"Loading reranker model: {model_name}")
            
            # Initialize model
            # We don't implement complex caching here as SentenceTransformers handles its own cache
            # But we could implement object caching if initialization is slow
            self.model = CrossEncoder(model_name)
            logger.info("Reranker model loaded successfully")
            
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking will be disabled")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading reranker model: {str(e)}")
            self.model = None

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = None) -> List[RetrievalResult]:
        """
        Rerank a list of retrieval results against a query
        
        Args:
            query: Search query
            results: List of retrieval results to rerank
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of retrieval results
        """
        try:
            if not self.model or not results:
                return results[:top_k] if top_k else results

            start_time = time.time()
            
            # Prepare pairs for cross-encoder
            pairs = [[query, result.content] for result in results]
            
            # Predict scores
            scores = self.model.predict(pairs)
            
            # Update results with new scores
            for result, score in zip(results, scores):
                # Cross-encoders return logits or unnormalized scores usually
                # We normalize them to 0-1 range approx for consistency if needed, 
                # or just use them as is for sorting. 
                # CrossEncoder default output depends on usage, but typically for Ranking it is a score.
                # Sigmoid is often applied for 0-1 probability.
                result.final_score = float(score)
                result.retrieval_method = "hybrid_reranked"
            
            # Sort by new score
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Update ranking positions
            for i, result in enumerate(results):
                result.ranking_position = i + 1
            
            # Limit to top_k
            final_k = top_k or config.retrieval.rerank_top_k
            reranked_results = results[:final_k]
            
            processing_time = time.time() - start_time
            logger.info(f"Reranking completed in {processing_time:.3f}s")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return results[:top_k] if top_k else results
