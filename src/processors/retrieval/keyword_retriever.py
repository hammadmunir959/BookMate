"""
Keyword Retriever for RAG Retrieval Microservice
Handles keyword-based search using SQLite FTS5
"""

import logging
from typing import List, Dict, Optional, Any
import time

from src.core.config import config
from src.core.document_models import RetrievalQuery, RetrievalResult, QueryType
from src.storage.metadata_store import metadata_store

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """Keyword retriever using SQLite FTS5 full-text search"""
    
    def __init__(self):
        """Initialize keyword retriever"""
        self.metadata_store = metadata_store
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform keyword retrieval
        
        Args:
            query: RetrievalQuery object
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            start_time = time.time()
            
            # Check if keyword search is enabled
            if not config.retrieval.enable_keyword_search:
                logger.info("Keyword search disabled, returning empty results")
                return []
            
            # Perform keyword search
            search_results = self.metadata_store.keyword_search(
                query=query.query,
                limit=query.top_k,
                filters=query.filters
            )

            # Convert dictionaries to RetrievalResult objects
            results = []
            for chunk_dict in search_results:
                result = RetrievalResult(
                    chunk_id=chunk_dict.get('chunk_id', f"{chunk_dict['document_id']}_chunk"),
                    document_id=chunk_dict['document_id'],
                    content=chunk_dict.get('content', ''),  # Use actual chunk content
                    relevance_score=0.5,  # Default relevance for keyword matches
                    semantic_score=0.0,
                    keyword_score=1.0,
                    final_score=0.5,
                    citation="",
                    metadata={
                        'title': chunk_dict.get('title', ''),
                        'filename': chunk_dict.get('filename', ''),
                        'file_size': chunk_dict.get('file_size', 0),
                        'mime_type': chunk_dict.get('mime_type', ''),
                        'created_at': chunk_dict.get('created_at', ''),
                        'chunk_index': chunk_dict.get('chunk_index', 0),
                        'token_count': chunk_dict.get('token_count', 0),
                        'start_position': chunk_dict.get('start_position', 0),
                        'end_position': chunk_dict.get('end_position', 0)
                    },
                    ranking_position=len(results) + 1,
                    retrieval_method="keyword"
                )
                results.append(result)
            
            processing_time = time.time() - start_time
            logger.info(f"Keyword retrieval completed in {processing_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {str(e)}")
            return []
    
    def retrieve_with_expansion(self, query: RetrievalQuery, expanded_queries: List[str]) -> List[RetrievalResult]:
        """
        Perform keyword retrieval with query expansion
        
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
                            result.keyword_score *= 0.8
                            result.final_score = result.keyword_score
                            result.retrieval_method = "keyword_expanded"
                            all_results.append(result)
                            seen_chunk_ids.add(result.chunk_id)
                            
                except Exception as e:
                    logger.warning(f"Error with expanded query '{expanded_query}': {str(e)}")
                    continue
            
            # Sort by keyword score
            all_results.sort(key=lambda x: x.keyword_score, reverse=True)
            
            # Limit to requested top_k
            final_results = all_results[:query.top_k]
            
            processing_time = time.time() - start_time
            logger.info(f"Keyword retrieval with expansion completed in {processing_time:.3f}s, found {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval with expansion: {str(e)}")
            return self.retrieve(query)  # Fallback to basic retrieval
    
    def search_by_author(self, author: str, limit: int = 10) -> List[RetrievalResult]:
        """
        Search for chunks by author
        
        Args:
            author: Author name
            limit: Maximum number of results
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Create a query with author filter
            query = RetrievalQuery(
                query="",  # Empty query, will be filtered by author
                query_type=QueryType.KEYWORD,
                filters={"author": author},
                top_k=limit
            )
            
            # Use a generic search term to get results
            results = self.metadata_store.keyword_search(
                query="*",  # Match all content
                top_k=limit,
                filters={"author": author}
            )
            
            # Update retrieval method
            for result in results:
                result.retrieval_method = "author_search"
            
            logger.info(f"Found {len(results)} chunks by author: {author}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by author {author}: {str(e)}")
            return []
    
    def search_by_section(self, section: str, limit: int = 10) -> List[RetrievalResult]:
        """
        Search for chunks by section
        
        Args:
            section: Section name
            limit: Maximum number of results
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Create a query with section filter
            query = RetrievalQuery(
                query="",  # Empty query, will be filtered by section
                query_type=QueryType.KEYWORD,
                filters={"section": section},
                top_k=limit
            )
            
            # Use a generic search term to get results
            results = self.metadata_store.keyword_search(
                query="*",  # Match all content
                top_k=limit,
                filters={"section": section}
            )
            
            # Update retrieval method
            for result in results:
                result.retrieval_method = "section_search"
            
            logger.info(f"Found {len(results)} chunks in section: {section}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by section {section}: {str(e)}")
            return []
    
    def search_by_document_type(self, doc_type: str, limit: int = 10) -> List[RetrievalResult]:
        """
        Search for chunks by document type
        
        Args:
            doc_type: Document type (e.g., "pdf", "docx")
            limit: Maximum number of results
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Create a query with document type filter
            query = RetrievalQuery(
                query="",  # Empty query, will be filtered by document type
                query_type=QueryType.KEYWORD,
                filters={"document_type": doc_type},
                top_k=limit
            )
            
            # Use a generic search term to get results
            results = self.metadata_store.keyword_search(
                query="*",  # Match all content
                top_k=limit,
                filters={"document_type": doc_type}
            )
            
            # Update retrieval method
            for result in results:
                result.retrieval_method = "document_type_search"
            
            logger.info(f"Found {len(results)} chunks in document type: {doc_type}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by document type {doc_type}: {str(e)}")
            return []
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific chunk
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk metadata or None if not found
        """
        try:
            return self.metadata_store.get_chunk_metadata(chunk_id)
        except Exception as e:
            logger.error(f"Error getting chunk metadata for {chunk_id}: {str(e)}")
            return None
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata or None if not found
        """
        try:
            return self.metadata_store.get_document_metadata(document_id)
        except Exception as e:
            logger.error(f"Error getting document metadata for {document_id}: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to metadata store"""
        try:
            return self.metadata_store.test_connection()
        except Exception as e:
            logger.error(f"Error testing keyword retriever connection: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the keyword retriever"""
        try:
            metadata_stats = self.metadata_store.get_document_stats()
            
            return {
                "type": "keyword",
                "fts5_available": True,  # Assume FTS5 is available if we got this far
                "metadata_store_stats": metadata_stats,
                "enabled": config.retrieval.enable_keyword_search
            }
            
        except Exception as e:
            logger.error(f"Error getting keyword retriever stats: {str(e)}")
            return {
                "type": "keyword",
                "error": str(e),
                "enabled": config.retrieval.enable_keyword_search
            }

