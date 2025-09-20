"""
Hybrid Retriever for RAG Retrieval Microservice
Combines semantic and keyword search with intelligent ranking
"""

import logging
from typing import List, Dict, Optional, Any
import time

from src.core.config import config
from src.core.document_models import RetrievalQuery, RetrievalResult, QueryType, QueryExpansion
from src.processors.retrieval.semantic_retriever import SemanticRetriever
from src.processors.retrieval.keyword_retriever import KeywordRetriever
from src.processors.retrieval.query_parser import QueryParser
from src.processors.retrieval.query_rephraser import QueryRephraser
from src.utils.logging_utils import retrieval_logger

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search"""
    
    def __init__(self):
        """Initialize hybrid retriever"""
        self.semantic_retriever = SemanticRetriever()
        self.keyword_retriever = KeywordRetriever()
        self.query_parser = QueryParser()
        self.query_rephraser = QueryRephraser()
        self.hybrid_weight = config.retrieval.hybrid_weight
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval combining semantic and keyword search
        
        Args:
            query: RetrievalQuery object
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            start_time = time.time()
            query_id = getattr(query, 'query_id', f"query_{int(start_time)}")

            # Log retrieval pipeline start
            retrieval_logger.log_pipeline_start("retrieval", query_id, query_text=query.query)

            # Check if hybrid search is enabled
            if not config.retrieval.enable_hybrid_search:
                logger.info("Hybrid search disabled, falling back to semantic search")
                retrieval_logger.log_retrieval_step("hybrid_check", query_id, "disabled, using semantic only")
                return self.semantic_retriever.retrieve(query)
            
            # Expand query if enabled using LLM-powered rephrasing
            expanded_queries = []
            logger.info(f"Query expansion enabled: {query.enable_query_expansion}")
            if query.enable_query_expansion:
                try:
                    # Get document summaries for context
                    document_summaries = []
                    if hasattr(query, 'document_ids') and query.document_ids:
                        for doc_id in query.document_ids:
                            try:
                                from src.storage.metadata_store import SQLiteStorage
                                store = SQLiteStorage()
                                summary_data = store.get_document_summary(doc_id)
                                if summary_data and summary_data.get('summary_text'):
                                    document_summaries.append(summary_data.get('summary_text', ''))
                            except Exception as e:
                                logger.warning(f"Could not get summary for document {doc_id}: {e}")

                    # Use LLM-powered QueryRephraser
                    llm_client = self.query_rephraser._get_llm_client()
                    logger.info(f"QueryRephraser: LLM available = {llm_client is not None}")
                    logger.info(f"QueryRephraser: Document summaries count = {len(document_summaries)}")

                    rephrased_results = self.query_rephraser.rephrase_query(
                        original_query=query.query,
                        document_summaries=document_summaries,
                        max_variations=3
                    )

                    logger.info(f"QueryRephraser: Got {len(rephrased_results)} rephrased results")

                    # Extract rephrased queries
                    for result in rephrased_results:
                        expanded_queries.append(result.rephrased_query)
                        logger.info(f"QueryRephraser: Added query: '{result.rephrased_query}'")

                    retrieval_logger.log_retrieval_step("query_expansion", query_id, f"expanded to {len(expanded_queries)} queries using LLM")
                except Exception as e:
                    logger.warning(f"LLM query expansion failed, falling back to basic expansion: {e}")
                    # Fallback to basic expansion
                    expansion = self.query_parser.expand_query(query)
                    expanded_queries = expansion.expanded_queries[1:]  # Exclude original query
                    retrieval_logger.log_retrieval_step("query_expansion", query_id, f"fallback expansion to {len(expanded_queries)} queries")
            else:
                retrieval_logger.log_retrieval_step("query_expansion", query_id, "disabled")

            # Perform semantic search
            semantic_results = []
            if config.retrieval.enable_semantic_search:
                if expanded_queries:
                    semantic_results = self.semantic_retriever.retrieve_with_expansion(query, expanded_queries)
                    retrieval_logger.log_retrieval_step("semantic_search", query_id, f"found {len(semantic_results)} results with expansion")
                else:
                    semantic_results = self.semantic_retriever.retrieve(query)
                    retrieval_logger.log_retrieval_step("semantic_search", query_id, f"found {len(semantic_results)} results")
            else:
                retrieval_logger.log_retrieval_step("semantic_search", query_id, "disabled")

            # Perform keyword search
            keyword_results = []
            if config.retrieval.enable_keyword_search:
                if expanded_queries:
                    keyword_results = self.keyword_retriever.retrieve_with_expansion(query, expanded_queries)
                    retrieval_logger.log_retrieval_step("keyword_search", query_id, f"found {len(keyword_results)} results with expansion")
                else:
                    keyword_results = self.keyword_retriever.retrieve(query)
                    retrieval_logger.log_retrieval_step("keyword_search", query_id, f"found {len(keyword_results)} results")
            else:
                retrieval_logger.log_retrieval_step("keyword_search", query_id, "disabled")
            
            # Combine and rank results
            combined_results = self._combine_results(semantic_results, keyword_results, query)
            retrieval_logger.log_retrieval_step("result_combination", query_id, f"combined {len(semantic_results)} semantic + {len(keyword_results)} keyword = {len(combined_results)} total")

            # Apply final ranking
            final_results = self._apply_hybrid_ranking(combined_results, query)
            retrieval_logger.log_retrieval_step("final_ranking", query_id, f"ranked {len(final_results)} final results")

            processing_time = time.time() - start_time
            logger.info(f"Hybrid retrieval completed in {processing_time:.3f}s, found {len(final_results)} results")

            # Log pipeline completion
            retrieval_logger.log_pipeline_end("retrieval", query_id, processing_time, "completed",
                                           semantic_results=len(semantic_results),
                                           keyword_results=len(keyword_results),
                                           final_results=len(final_results))
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            # Fallback to semantic search
            return self.semantic_retriever.retrieve(query)

    def retrieve_with_rephrasing(
        self,
        query: RetrievalQuery,
        document_summaries: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """
        Enhanced retrieval with query rephrasing and retry logic

        Args:
            query: Original retrieval query
            document_summaries: Optional document summaries for rephrasing context

        Returns:
            List of retrieval results
        """
        try:
            start_time = time.time()

            # Attempt 1: Original query
            logger.info(f"🔄 Retrieval attempt 1/3: Using original query")
            results = self.retrieve(query)

            if results:
                processing_time = time.time() - start_time
                logger.info(f"✅ Found {len(results)} results with original query in {processing_time:.3f}s")
                return results

            # If no results and we have document summaries, try rephrasing
            if document_summaries and self.query_rephraser:
                logger.info("🔄 No results found, attempting query rephrasing...")

                # Generate rephrased queries
                rephrased_queries = self.query_rephraser.rephrase_query(
                    query.query,
                    document_summaries,
                    max_variations=3
                )

                if not rephrased_queries:
                    logger.warning("❌ Query rephrasing failed or returned no variations")
                    return []

                # Try each rephrased query (Attempts 2-4, but limit to 3 total attempts)
                for i, rephrased_query_obj in enumerate(rephrased_queries[:2]):  # Max 2 additional attempts
                    attempt_num = i + 2
                    logger.info(f"🔄 Retrieval attempt {attempt_num}/3: '{rephrased_query_obj.rephrased_query}'")

                    # Create new query with rephrased text
                    rephrased_query = RetrievalQuery(
                        query=rephrased_query_obj.rephrased_query,
                        query_type=query.query_type,
                        filters=query.filters,
                        top_k=query.top_k,
                        similarity_threshold=query.similarity_threshold,
                        enable_reranking=query.enable_reranking,
                        enable_query_expansion=query.enable_query_expansion
                    )

                    # Try retrieval with rephrased query
                    results = self.retrieve(rephrased_query)

                    if results:
                        processing_time = time.time() - start_time
                        logger.info(f"✅ Found {len(results)} results with rephrased query (attempt {attempt_num}) in {processing_time:.3f}s")
                        logger.info(f"   Original: '{query.query}'")
                        logger.info(f"   Rephrased: '{rephrased_query_obj.rephrased_query}'")
                        return results

                logger.info("❌ All rephrasing attempts failed to find results")

            processing_time = time.time() - start_time
            logger.info(f"❌ No results found after {processing_time:.3f}s and all attempts")
            return []

        except Exception as e:
            logger.error(f"Error in enhanced retrieval with rephrasing: {str(e)}")
            # Fallback to original retrieval
            return self.retrieve(query)
    
    def _combine_results(
        self, 
        semantic_results: List[RetrievalResult], 
        keyword_results: List[RetrievalResult],
        query: RetrievalQuery
    ) -> List[RetrievalResult]:
        """Combine semantic and keyword results"""
        try:
            # Create a dictionary to store combined results
            combined_dict = {}
            
            # Add semantic results
            for result in semantic_results:
                chunk_id = result.chunk_id
                if chunk_id not in combined_dict:
                    combined_dict[chunk_id] = result
                else:
                    # Update existing result with semantic score
                    existing = combined_dict[chunk_id]
                    existing.semantic_score = result.semantic_score
                    existing.retrieval_method = "hybrid"
            
            # Add keyword results
            for result in keyword_results:
                chunk_id = result.chunk_id
                if chunk_id not in combined_dict:
                    combined_dict[chunk_id] = result
                else:
                    # Update existing result with keyword score
                    existing = combined_dict[chunk_id]
                    existing.keyword_score = result.keyword_score
                    existing.retrieval_method = "hybrid"
            
            # Convert to list
            combined_results = list(combined_dict.values())
            
            logger.debug(f"Combined {len(semantic_results)} semantic + {len(keyword_results)} keyword = {len(combined_results)} unique results")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            # Return semantic results as fallback
            return semantic_results
    
    def _apply_hybrid_ranking(self, results: List[RetrievalResult], query: RetrievalQuery) -> List[RetrievalResult]:
        """Apply hybrid ranking to combined results"""
        try:
            for result in results:
                # Calculate hybrid score
                semantic_score = result.semantic_score
                keyword_score = result.keyword_score
                
                # Apply hybrid weighting
                hybrid_score = (
                    self.hybrid_weight * semantic_score + 
                    (1.0 - self.hybrid_weight) * keyword_score
                )
                
                # Apply metadata boosts
                boosted_score = self._apply_metadata_boosts(hybrid_score, result, query)
                
                # Update final score
                result.final_score = boosted_score
                result.relevance_score = boosted_score
            
            # Sort by final score
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Update ranking positions
            for i, result in enumerate(results):
                result.ranking_position = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying hybrid ranking: {str(e)}")
            return results
    
    def _apply_metadata_boosts(self, base_score: float, result: RetrievalResult, query: RetrievalQuery) -> float:
        """Apply metadata-based score boosts"""
        try:
            boosted_score = base_score
            metadata = result.metadata
            
            # Freshness boost
            if config.retrieval.freshness_boost > 0:
                upload_date = metadata.get("uploaded_at")
                if upload_date:
                    try:
                        from datetime import datetime, timedelta
                        if isinstance(upload_date, str):
                            upload_date = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                        
                        days_old = (datetime.now() - upload_date).days
                        if days_old < 30:  # Recent documents
                            freshness_boost = config.retrieval.freshness_boost * (1.0 - days_old / 30.0)
                            boosted_score += freshness_boost
                    except Exception:
                        pass  # Ignore date parsing errors
            
            # Section relevance boost
            if config.retrieval.section_boost > 0:
                section = metadata.get("section", "").lower()
                query_terms = query.query.lower().split()
                
                # Check if query terms match section
                for term in query_terms:
                    if term in section:
                        boosted_score += config.retrieval.section_boost
                        break
            
            # Author authority boost
            if config.retrieval.author_boost > 0:
                author = metadata.get("author", "").lower()
                if author and len(author) > 3:  # Non-empty, meaningful author name
                    boosted_score += config.retrieval.author_boost
            
            # Document type boost
            if config.retrieval.document_type_boost > 0:
                doc_type = metadata.get("file_extension", "").lower()
                if doc_type in [".pdf", ".docx"]:  # Prefer structured documents
                    boosted_score += config.retrieval.document_type_boost
            
            # Ensure score doesn't exceed 1.0
            return min(1.0, boosted_score)
            
        except Exception as e:
            logger.warning(f"Error applying metadata boosts: {str(e)}")
            return base_score
    
    def retrieve_by_type(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve results based on query type
        
        Args:
            query: RetrievalQuery object
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            if query.query_type == QueryType.SEMANTIC:
                return self.semantic_retriever.retrieve(query)
            elif query.query_type == QueryType.KEYWORD:
                return self.keyword_retriever.retrieve(query)
            else:  # HYBRID or default
                return self.retrieve(query)
                
        except Exception as e:
            logger.error(f"Error in type-based retrieval: {str(e)}")
            return self.retrieve(query)  # Fallback to hybrid
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Get chunks similar to a specific chunk using hybrid approach
        
        Args:
            chunk_id: ID of the reference chunk
            top_k: Number of similar chunks to return
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Get semantic similar chunks
            semantic_similar = self.semantic_retriever.get_similar_chunks(chunk_id, top_k)
            
            # Get the reference chunk for keyword similarity
            reference_chunk = self.semantic_retriever.vector_store.get_chunk_by_id(chunk_id)
            if not reference_chunk:
                return semantic_similar
            
            # Create keyword query from reference chunk
            keyword_query = RetrievalQuery(
                query=reference_chunk.content,
                query_type=QueryType.KEYWORD,
                top_k=top_k,
                similarity_threshold=0.3
            )
            
            # Get keyword similar chunks
            keyword_similar = self.keyword_retriever.retrieve(keyword_query)
            
            # Combine and rank
            combined = self._combine_results(semantic_similar, keyword_similar, keyword_query)
            final_results = self._apply_hybrid_ranking(combined, keyword_query)
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting similar chunks for {chunk_id}: {str(e)}")
            return self.semantic_retriever.get_similar_chunks(chunk_id, top_k)
    
    def test_connection(self) -> bool:
        """Test connection to both retrievers"""
        try:
            semantic_ok = self.semantic_retriever.test_connection()
            keyword_ok = self.keyword_retriever.test_connection()
            
            return semantic_ok and keyword_ok
            
        except Exception as e:
            logger.error(f"Error testing hybrid retriever connection: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid retriever"""
        try:
            semantic_stats = self.semantic_retriever.get_stats()
            keyword_stats = self.keyword_retriever.get_stats()
            
            return {
                "type": "hybrid",
                "hybrid_weight": self.hybrid_weight,
                "semantic_retriever": semantic_stats,
                "keyword_retriever": keyword_stats,
                "enabled": config.retrieval.enable_hybrid_search
            }
            
        except Exception as e:
            logger.error(f"Error getting hybrid retriever stats: {str(e)}")
            return {
                "type": "hybrid",
                "error": str(e),
                "enabled": config.retrieval.enable_hybrid_search
            }

