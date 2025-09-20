"""
Retrieval API Router for Unified RAG Microservice
Handles query processing and document retrieval endpoints
"""

import logging
import time
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.config import config
from src.core.document_models import QueryType, RetrievalQuery, RetrievalResponse, RetrievalResult
from src.core.augmentation import augment
from src.storage.vector_store import chroma_db
from src.storage.metadata_store import sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for API
class RetrievalRequest(BaseModel):
    """Request model for document retrieval"""
    model_config = {"protected_namespaces": ()}

    query: str = Field(..., description="Search query", min_length=1, max_length=config.retrieval.max_query_length)
    query_type: Optional[str] = Field("hybrid", description="Query type: semantic, keyword, or hybrid")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    top_k: Optional[int] = Field(config.retrieval.top_k_retrieval, description="Number of results to return", ge=1, le=config.retrieval.top_k_max)
    similarity_threshold: Optional[float] = Field(config.retrieval.similarity_threshold, description="Minimum similarity score", ge=0.0, le=1.0)
    enable_reranking: Optional[bool] = Field(config.retrieval.enable_reranking, description="Enable result reranking")
    enable_query_expansion: Optional[bool] = Field(config.retrieval.enable_query_expansion, description="Enable query expansion")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    enable_augmentation: Optional[bool] = Field(True, description="Generate augmented prompt for generation")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation messages")


class RetrievalResponseModel(BaseModel):
    """Response model for retrieval results"""
    model_config = {"protected_namespaces": ()}

    success: bool
    query: str
    query_type: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    cache_hit: bool
    filters_applied: Dict[str, Any]
    query_hash: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    augmented_prompt: Optional[str] = None
    citations: Optional[List[str]] = None
    context_metadata: Optional[Dict[str, Any]] = None
    context: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None


@router.post("/retrieve", response_model=RetrievalResponseModel)
async def retrieve_documents(request: RetrievalRequest):
    """Retrieve documents based on query with optional augmentation"""
    start_time = time.time()

    try:
        logger.info(f"🔍 Starting retrieval for query: {request.query[:50]}...")

        # Validate query type
        try:
            query_type = QueryType(request.query_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid query_type: {request.query_type}. Must be one of: semantic, keyword, hybrid"
            )

        # Create retrieval query
        retrieval_query = RetrievalQuery(
            query=request.query,
            query_type=query_type,
            filters=request.filters or {},
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_reranking=request.enable_reranking,
            enable_query_expansion=request.enable_query_expansion,
            session_id=request.session_id,
            user_id=request.user_id
        )

        # Perform retrieval using the hybrid retriever
        from src.processors.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        retrieval_results = retriever.retrieve(retrieval_query)

        # Convert to response format
        results = []
        for result in retrieval_results:
            results.append({
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "content": result.content,
                "relevance_score": result.relevance_score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "final_score": result.final_score,
                "citation": {
                    "type": result.citation.type.value if result.citation else "chunk",
                    "page_number": getattr(result.citation, 'page_number', None) if result.citation else None,
                    "section": getattr(result.citation, 'section', None) if result.citation else None,
                    "heading": getattr(result.citation, 'heading', None) if result.citation else None,
                    "paragraph_number": getattr(result.citation, 'paragraph_number', None) if result.citation else None,
                    "reference_id": getattr(result.citation, 'reference_id', None) if result.citation else None,
                    "metadata": result.citation.metadata if result.citation else {}
                } if result.citation else {},
                "metadata": result.metadata,
                "ranking_position": result.ranking_position,
                "retrieval_method": result.retrieval_method
            })

        processing_time = time.time() - start_time
        
        response_dict = {
            "success": True,
            "query": request.query,
            "query_type": request.query_type,
            "results": results,
            "total_results": len(results),
            "processing_time": processing_time,
            "cache_hit": False,
            "filters_applied": request.filters or {},
            "query_hash": retrieval_query.query_hash,
            "session_id": request.session_id,
            "user_id": request.user_id
        }

        # Add augmentation if requested
        if request.enable_augmentation and len(retrieval_results) > 0:
            try:
                from src.core.augmentation import AugmentationConfigModel
                aug_config = AugmentationConfigModel(
                    max_context_tokens=config.generation.max_input_length,
                    min_score_threshold=config.retrieval.similarity_threshold,
                    top_k=min(request.top_k, config.retrieval.top_k_retrieval),
                    dedup_similarity=0.85,
                    citation_style=config.generation.citation_style
                )

                aug_result = augment(
                    query=request.query,
                    retrieved_results=retrieval_results,
                    conversation_history=request.conversation_history,
                    override_config=aug_config
                )

                response_dict.update({
                    "augmented_prompt": aug_result.augmented_prompt,
                    "citations": aug_result.citations,
                    "context_metadata": aug_result.context_metadata,
                    "context": [
                        {
                            "chunk_id": item.chunk_id,
                            "text": item.text,
                            "citation": item.citation,
                            "score": item.score,
                            "document_id": item.document_id,
                        }
                        for item in aug_result.context_items
                    ]
                })

            except Exception as e:
                logger.warning(f"Augmentation failed: {str(e)}")
                response_dict["augmented_prompt"] = request.query  # Fallback

        # No error message for successful retrieval

        logger.info(f"✅ Retrieval completed in {processing_time:.2f}s, found {len(results)} results")
        return RetrievalResponseModel(**response_dict)

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Retrieval failed: {str(e)}"
        logger.error(error_msg)
        processing_time = time.time() - start_time
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/retrieve/similar")
async def get_similar_chunks(
    chunk_id: str,
    top_k: Optional[int] = config.retrieval.top_k_retrieval
):
    """Get chunks similar to a specific chunk"""
    try:
        logger.info(f"🔍 Finding similar chunks for: {chunk_id}")

        # Get the original chunk
        original_chunks = chroma_db.get_document_chunks_by_ids([chunk_id])
        if not original_chunks:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")

        original_chunk = original_chunks[0]

        # Find similar chunks
        similar_results = chroma_db.search_similar(
            query_embedding=original_chunk.embedding,
            top_k=top_k + 1  # +1 to exclude the original
        )

        # Filter out the original chunk
        results = [r for r in similar_results if r.chunk_id != chunk_id][:top_k]

        # Convert to response format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "content": result.content,
                "relevance_score": result.relevance_score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "final_score": result.final_score,
                "citation": {
                    "type": result.citation.type.value if result.citation else "chunk",
                    "page_number": getattr(result.citation, 'page_number', None) if result.citation else None,
                    "section": getattr(result.citation, 'section', None) if result.citation else None,
                    "heading": getattr(result.citation, 'heading', None) if result.citation else None,
                    "paragraph_number": getattr(result.citation, 'paragraph_number', None) if result.citation else None,
                    "reference_id": getattr(result.citation, 'reference_id', None) if result.citation else None,
                    "metadata": result.citation.metadata if result.citation else {}
                } if result.citation else {},
                "metadata": result.metadata,
                "ranking_position": result.ranking_position,
                "retrieval_method": result.retrieval_method
            })

        return {
            "success": True,
            "chunk_id": chunk_id,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "processing_time": 0.1  # Simplified
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar chunks retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similar chunks retrieval failed: {str(e)}")
