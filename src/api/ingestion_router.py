"""
Ingestion API Router for Unified RAG Microservice
Handles document upload and processing endpoints
"""

import logging
import time
import uuid
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.config import config
from src.core.document_models import IngestionResult
from src.processors.ingestion.document_validator import DocumentValidator
from src.processors.ingestion.text_extractor import EnhancedDocumentTextExtractor
from src.processors.ingestion.document_chunker import EnhancedDocumentChunker
from src.processors.ingestion.metadata_enricher import MetadataEnricher
from src.processors.ingestion.document_summarizer import DocumentSummarizer
from src.storage.vector_store import chroma_db
from src.storage.metadata_store import sqlite_storage
from src.utils.progress_tracker import IngestionStepTracker, register_tracker, unregister_tracker

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for API
class IngestionResponse(BaseModel):
    """Response model for document ingestion"""
    model_config = {"protected_namespaces": ()}

    success: bool
    document_id: str
    filename: str
    chunks_created: int
    pages: int
    citation_mode: str
    processing_time: float
    message: str
    steps_completed: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_metadata: Optional[str] = None,
    uploader_id: Optional[str] = None,
    force_reprocess: bool = False
):
    """
    Ingest and process a document for RAG

    - **file**: Document file to process (PDF, DOCX, HTML, TXT, MD, RTF)
    - **custom_metadata**: Optional JSON string of custom metadata
    - **uploader_id**: Optional user identifier
    - **force_reprocess**: Force reprocessing even if document exists
    """
    start_time = time.time()
    document_id = file.filename  # Use filename as document ID

    # Initialize progress tracker
    tracker = IngestionStepTracker(document_id, file.filename)
    register_tracker(tracker)

    try:
        logger.info(f"📁 Starting ingestion for document: {file.filename}")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check file size
        file_content = await file.read()
        if len(file_content) > config.ingestion.max_document_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.ingestion.max_document_size_mb}MB"
            )

        # Save uploaded file temporarily
        temp_file_path = f"/tmp/{document_id}_{time.time()}.tmp"
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        try:
            # Parse custom metadata if provided
            parsed_metadata = None
            if custom_metadata:
                import json
                try:
                    parsed_metadata = json.loads(custom_metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in custom_metadata: {custom_metadata}")

            # Step 1: Validate document
            tracker.start_step("validation")
            validator = DocumentValidator()
            is_valid, error_msg = validator.validate_file(temp_file_path, original_filename=file.filename)
            if not is_valid:
                tracker.fail_step("validation", error_msg)
                raise HTTPException(status_code=400, detail=f"Validation failed: {error_msg}")
            tracker.complete_step("validation", {"file_path": temp_file_path, "original_filename": file.filename})

            # Step 2: Extract text
            tracker.start_step("text_extraction")
            text_extractor = EnhancedDocumentTextExtractor()
            extracted_text, extracted_content = text_extractor.extract_text_with_structure(temp_file_path, original_filename=file.filename)
            if not extracted_text.strip():
                tracker.fail_step("text_extraction", "No text could be extracted")
                raise HTTPException(status_code=400, detail="No text could be extracted from document")
            tracker.complete_step("text_extraction", {
                "text_length": len(extracted_text),
                "structural_units": len(extracted_content.structural_units)
            })

            # Step 3: Chunk document
            tracker.start_step("chunking")
            chunker = EnhancedDocumentChunker()
            chunks = chunker.chunk_document_with_structure(extracted_content, document_id)
            if not chunks:
                tracker.fail_step("chunking", "No chunks produced")
                raise HTTPException(status_code=500, detail="Document chunking produced no chunks")
            tracker.complete_step("chunking", {"chunk_count": len(chunks)})

            # Step 4: Enrich metadata
            tracker.start_step("metadata_enrichment")
            metadata_enricher = MetadataEnricher()
            enriched_chunks = metadata_enricher.enrich_chunks(
                chunks, extracted_content.metadata, extracted_content.extraction_info
            )
            tracker.complete_step("metadata_enrichment", {"enriched_chunks": len(enriched_chunks)})

            # Step 5: Generate embeddings and store
            tracker.start_step("embedding_generation")
            # Import here to avoid circular imports
            from src.processors.ingestion.ingestion_pipeline import DocumentProcessor
            processor = DocumentProcessor()
            # Generate embeddings for chunks
            embedded_chunks = processor._generate_embeddings_with_persistence(
                enriched_chunks, document_id, temp_file_path, force_reprocess
            )

            # Store chunks in vector store
            chroma_db.add_chunks(embedded_chunks)

            # Store metadata in SQLite
            processor._store_in_sqlite(document_id, enriched_chunks[0].metadata if enriched_chunks else None, embedded_chunks, uploader_id)

            success, message = True, f"Successfully processed {len(embedded_chunks)} chunks"
            tracker.complete_step("embedding_generation", {"chunks_processed": len(embedded_chunks)})

            # Step 6: Generate summaries
            tracker.start_step("summarization")
            summarizer = DocumentSummarizer()
            summary_result = summarizer.summarize_document(document_id, enriched_chunks, force_regenerate=force_reprocess)
            if summary_result:
                # Store summaries - convert dictionaries to proper objects
                final_summary_dict = summary_result.get("final_summary")
                if final_summary_dict:
                    from src.core.document_models import DocumentSummary
                    doc_summary = DocumentSummary(
                        summary_id=str(uuid.uuid4()),
                        document_id=document_id,
                        summary_text=final_summary_dict.get('summary_text', ''),
                        chunk_count=len(summary_result.get('chunk_summaries', [])),
                        word_count=len(final_summary_dict.get('summary_text', '').split()),
                        processing_time=final_summary_dict.get('processing_time', 0)
                    )
                    sqlite_storage.store_document_summary(doc_summary)

                # Store chunk summaries
                chunk_summaries_list = summary_result.get("chunk_summaries", [])
                if chunk_summaries_list:
                    from src.core.document_models import ChunkSummary
                    chunk_summary_objects = []
                    for i, chunk_summary_dict in enumerate(chunk_summaries_list):
                        chunk_summary_obj = ChunkSummary(
                            chunk_summary_id=str(uuid.uuid4()),
                            document_id=document_id,
                            chunk_id=chunk_summary_dict.get('chunk_id', f'chunk_{i}'),
                            chunk_index=i,
                            summary_text=chunk_summary_dict.get('summary_text', ''),
                            word_count=len(chunk_summary_dict.get('summary_text', '').split()),
                            processing_time=chunk_summary_dict.get('processing_time', 0)
                        )
                        chunk_summary_objects.append(chunk_summary_obj)

                    if chunk_summary_objects:
                        sqlite_storage.store_chunk_summaries(chunk_summary_objects)
            tracker.complete_step("summarization", {"summary_generated": bool(summary_result)})

            # Mark as completed
            tracker.complete_ingestion({
                "chunks_created": len(enriched_chunks),
                "pages": len(extracted_content.structural_units),
                "citation_mode": "page"
            })

            processing_time = time.time() - start_time

            result = IngestionResult(
                success=True,
                document_id=document_id,
                filename=file.filename,
                chunks_created=len(enriched_chunks),
                pages=len(extracted_content.structural_units),
                citation_mode="page",
                processing_time=processing_time,
                message=f"Successfully processed {len(enriched_chunks)} chunks",
                steps_completed=[step.name for step in tracker.get_report().steps],
                errors=[],
                warnings=[]
            )

            logger.info(f"✅ Document ingestion completed: {file.filename} in {processing_time:.2f}s")
            unregister_tracker(document_id)
            return IngestionResponse(**result.__dict__)

        finally:
            # Clean up temporary file
            if Path(temp_file_path).exists():
                Path(temp_file_path).unlink()

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Ingestion failed: {str(e)}"
        logger.error(error_msg)
        tracker.fail_ingestion(error_msg)
        unregister_tracker(document_id)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/ingest/status/{document_id}")
async def get_ingestion_status(document_id: str):
    """Get processing status for a document"""
    try:
        # Get document metadata from SQLite
        doc_metadata = sqlite_storage.get_document_metadata(document_id)
        if doc_metadata:
            return {
                "document_id": document_id,
                "status": doc_metadata.get("status", "unknown"),
                "uploaded_at": doc_metadata.get("uploaded_at"),
                "processed_at": doc_metadata.get("processed_at"),
                "citation_mode": doc_metadata.get("citation_mode", "unknown"),
                "chunks_count": len(chroma_db.get_document_chunks(document_id))
            }
        else:
            return {"document_id": document_id, "status": "not_found"}
    except Exception as e:
        logger.error(f"Error getting status for {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document status")
