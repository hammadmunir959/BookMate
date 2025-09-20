"""
Simplified RAG API Router - Two Endpoint Architecture
Combines document ingestion and unified query processing
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.service_manager import service_manager
from src.core.config import config
from src.processors.ingestion.ingestion_pipeline import DocumentProcessor
from src.processors.retrieval.hybrid_retriever import HybridRetriever
from src.processors.generation.text_generator import GenerationPipeline
from src.core.document_models import QueryType, RetrievalQuery, GenerationRequest
from src.storage.metadata_store import sqlite_storage

logger = logging.getLogger(__name__)

# Create main router
main_router = APIRouter()

# Enhanced logging utility
class EnhancedLogger:
    """Beautiful logging utility for RAG operations"""

    @staticmethod
    def log_step_start(step_name: str, details: Dict[str, Any] = None):
        """Log the start of a processing step"""
        details_str = f" - {', '.join([f'{k}: {v}' for k, v in (details or {}).items()])}" if details else ""
        logger.info(f"🚀 STARTING: {step_name}{details_str}")

    @staticmethod
    def log_step_success(step_name: str, duration: float, details: Dict[str, Any] = None):
        """Log successful completion of a step"""
        details_str = f" - {', '.join([f'{k}: {v}' for k, v in (details or {}).items()])}" if details else ""
        logger.info(f"✅ COMPLETED: {step_name} in {duration:.2f}s{details_str}")

    @staticmethod
    def log_step_error(step_name: str, error: str, duration: Optional[float] = None):
        """Log error in a processing step"""
        duration_str = f" after {duration:.2f}s" if duration else ""
        logger.error(f"❌ FAILED: {step_name}{duration_str} - {error}")

    @staticmethod
    def log_ingestion_progress(document_id: str, step: str, progress: int, total: int):
        """Log ingestion progress"""
        logger.info(f"📄 {document_id}: {step} - {progress}/{total} completed")

    @staticmethod
    def log_query_progress(query: str, step: str, details: Dict[str, Any] = None):
        """Log query processing progress"""
        details_str = f" - {', '.join([f'{k}: {v}' for k, v in (details or {}).items()])}" if details else ""
        logger.info(f"🔍 {query[:50]}...: {step}{details_str}")


# Pydantic Models for New API
class IngestionStepStatus(BaseModel):
    """Status of an ingestion step"""
    name: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ChunkInfo(BaseModel):
    """Information about a document chunk"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    summary: Optional[str] = None


class IngestionResponse(BaseModel):
    """Comprehensive response for document ingestion"""
    model_config = {"protected_namespaces": ()}

    success: bool
    document_id: str
    filename: str
    file_size: int
    mime_type: str
    total_chunks: int
    chunks: List[ChunkInfo]
    summary: Dict[str, Any]
    processing_stats: Dict[str, Any]
    steps_status: List[IngestionStepStatus]
    total_processing_time: float
    citation_mode: str
    error_message: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for unified query processing"""
    model_config = {"protected_namespaces": ()}

    query: str = Field(..., description="User query", min_length=1)
    document_id: Optional[str] = Field(None, description="Specific document ID to query")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    context: Optional[str] = Field(None, description="Additional context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation messages")
    max_tokens: Optional[int] = Field(2000, description="Maximum tokens for generation")
    temperature: Optional[float] = Field(0.1, description="Generation temperature")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(0.2, description="Minimum similarity threshold")


class CitationInfo(BaseModel):
    """Citation information for retrieved chunks"""
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    page_number: Optional[int] = None
    section: Optional[str] = None
    citation_text: str


class QueryResponse(BaseModel):
    """Response model for unified query processing"""
    model_config = {"protected_namespaces": ()}

    success: bool
    query: str
    answer: str
    citations: List[CitationInfo]
    total_chunks_retrieved: int
    processing_stats: Dict[str, Any]
    model_used: str
    token_count: Optional[int] = None
    confidence_score: Optional[float] = None
    processing_time: float
    error_message: Optional[str] = None

# Pydantic Models for New API
class IngestionRequest(BaseModel):
    """Request model for document ingestion"""
    model_config = {"protected_namespaces": ()}

    custom_metadata: Optional[str] = Field(None, description="JSON string of custom metadata")
    uploader_id: Optional[str] = Field(None, description="User identifier")
    force_reprocess: Optional[bool] = Field(False, description="Force reprocessing if document exists")


class IngestionStepStatus(BaseModel):
    """Status of an ingestion step"""
    name: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ChunkInfo(BaseModel):
    """Information about a document chunk"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    summary: Optional[str] = None


class IngestionResponse(BaseModel):
    """Comprehensive response for document ingestion"""
    model_config = {"protected_namespaces": ()}

    success: bool
    document_id: str
    filename: str
    file_size: int
    mime_type: str
    total_chunks: int
    chunks: List[ChunkInfo]
    summary: Dict[str, Any]
    processing_stats: Dict[str, Any]
    steps_status: List[IngestionStepStatus]
    total_processing_time: float
    citation_mode: str
    error_message: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for unified query processing"""
    model_config = {"protected_namespaces": ()}

    query: str = Field(..., description="User query", min_length=1)
    document_id: Optional[str] = Field(None, description="Specific document ID to query")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    context: Optional[str] = Field(None, description="Additional context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous conversation messages")
    max_tokens: Optional[int] = Field(2000, description="Maximum tokens for generation")
    temperature: Optional[float] = Field(0.1, description="Generation temperature")
    top_k: Optional[int] = Field(5, description="Number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(0.2, description="Minimum similarity threshold")


class CitationInfo(BaseModel):
    """Citation information for retrieved chunks"""
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    page_number: Optional[int] = None
    section: Optional[str] = None
    citation_text: str


class QueryResponse(BaseModel):
    """Response model for unified query processing"""
    model_config = {"protected_namespaces": ()}

    success: bool
    query: str
    answer: str
    citations: List[CitationInfo]
    total_chunks_retrieved: int
    processing_stats: Dict[str, Any]
    model_used: str
    token_count: Optional[int] = None
    confidence_score: Optional[float] = None
    processing_time: float
    error_message: Optional[str] = None

# Health check endpoint is now in main.py

# System stats endpoint
@main_router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        # Get database stats
        chroma_stats = {}
        sqlite_stats = {}

        try:
            chroma_db = service_manager.get_service('chroma_db')
            chroma_stats = chroma_db.get_document_stats()
        except Exception as e:
            logger.warning(f"ChromaDB stats failed: {str(e)}")

        try:
            sqlite_storage = service_manager.get_service('sqlite_storage')
            sqlite_stats = sqlite_storage.get_document_stats()
        except Exception as e:
            logger.warning(f"SQLite stats failed: {str(e)}")

        # Get cache stats
        cache_stats = {}
        try:
            cache_manager = service_manager.get_service('cache_manager')
            cache_stats = cache_manager.get_stats()
        except Exception as e:
            logger.warning(f"Cache stats failed: {str(e)}")

        return {
            "service": "unified-rag",
            "version": "1.0.0",
            "databases": {
                "chromadb": chroma_stats,
                "sqlite": sqlite_stats
            },
            "cache": cache_stats,
            "config": {
                "max_document_size_mb": config.ingestion.max_document_size_mb,
                "supported_extensions": config.ingestion.allowed_extensions,
                "chunk_size": config.ingestion.chunk_size,
                "top_k_retrieval": config.retrieval.top_k_retrieval,
                "max_input_length": config.generation.max_input_length
            }
        }

    except Exception as e:
        logger.error(f"Stats retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")

# Root endpoint
@main_router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Unified RAG Microservice",
        "version": "1.0.0",
        "description": "Complete RAG system combining ingestion, retrieval, and generation",
        "endpoints": {
            "ingestion": "/api/v1/ingest",
            "retrieval": "/api/v1/retrieve",
            "generation": "/api/v1/generate",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        },
        "supported_formats": config.ingestion.allowed_extensions,
        "max_file_size_mb": config.ingestion.max_document_size_mb
    }
