"""
Document Processor - Professional Ingestion Workflow
Integrates all components for citation-ready document processing
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import traceback

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_models import DocumentMetadata, DocumentChunk, DocumentSummary, ChunkSummary
from src.storage.vector_store import chroma_db, cache_manager
from src.storage.metadata_store import sqlite_storage
from src.processors.ingestion.text_extractor import EnhancedDocumentTextExtractor, ExtractedContent
from src.processors.ingestion.document_chunker import EnhancedDocumentChunker, ChunkingConfig
from src.processors.ingestion.metadata_enricher import MetadataEnricher
from src.processors.ingestion.citation_manager import CitationManager, CitationStyle
from src.processors.ingestion.document_validator import DocumentValidator
from src.utils.progress_tracker import IngestionStepTracker, register_tracker, unregister_tracker
from src.processors.ingestion.document_summarizer import DocumentSummarizer
from src.core.config import config
from src.utils.logging_utils import ingestion_logger, log_pipeline_step

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Professional document processor for BookMate RAG Agent
    Implements complete ingestion workflow with citation support
    """

    def __init__(self):
        """Initialize document processor with all components"""
        try:
            # Initialize components
            self.validator = DocumentValidator()
            self.text_extractor = EnhancedDocumentTextExtractor()
            self.chunker = EnhancedDocumentChunker()
            self.metadata_enricher = MetadataEnricher()
            self.citation_manager = CitationManager()
            self.summarizer = DocumentSummarizer(batch_size=10)

            # Ensure necessary directories exist
            self._ensure_directories()

            logger.info("DocumentProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DocumentProcessor: {str(e)}")
            raise

    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            config.database.chroma_db_path,
            config.cache.embedding_cache_path,
            config.cache.response_cache_path,
            Path(config.database.sqlite_db_path).parent,  # Get parent directory of SQLite file
            "./logs"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def process_document(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
        uploader_id: str = None,
        force_reprocess: bool = False,
        original_filename: str = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Process a single document through the complete BookMate ingestion pipeline

        Args:
            file_path: Path to the document file
            custom_metadata: Additional metadata to add
            uploader_id: ID of user who uploaded the document
            force_reprocess: Force reprocessing even if document hasn't changed
            original_filename: Original filename for proper document ID

        Returns:
            Tuple of (success, message, processing_results)
        """
        # Generate document ID and filename
        filename = original_filename or os.path.basename(file_path)
        document_id = filename  # Use filename as document ID for easier lookup
        
        # Initialize step tracker
        tracker = IngestionStepTracker(document_id, filename)
        register_tracker(tracker)
        
        processing_results = {
            'document_id': document_id,
            'filename': filename,
            'chunks_created': 0,
            'pages': 0,
            'citation_mode': 'page',
            'processing_time': 0,
            'steps_completed': [],
            'errors': [],
            'warnings': []
        }
        
        start_time = time.time()

        try:
            # Log pipeline start
            ingestion_logger.log_pipeline_start("ingestion", document_id, filename=filename, uploader_id=uploader_id)
            logger.info(f"🚀 Starting BookMate ingestion for document: {filename}")
            
            # Step 0: Force Reprocess - Clear existing data if requested
            if force_reprocess:
                step = tracker.start_step("force_reprocess_cleanup")
                logger.info(f"🗑️ Force reprocessing enabled - clearing existing data for: {filename}")
                self._clear_existing_document_data(document_id)
                tracker.complete_step("force_reprocess_cleanup", {"cleared_existing_data": True})
                processing_results['steps_completed'].append("force_reprocess_cleanup")
            
            # Step 1: Upload Document (already done by API)
            step = tracker.start_step("document_upload")
            logger.info(f"📁 Document uploaded: {filename}")
            tracker.complete_step("document_upload", {"filename": filename})
            processing_results['steps_completed'].append("document_upload")
            
            # Step 2: Validation
            step = tracker.start_step("validation")
            logger.info(f"🔍 Validating document: {filename}")
            is_valid, error_msg = self.validator.validate_file(file_path, original_filename=original_filename)
            if not is_valid:
                tracker.fail_step("validation", error_msg)
                tracker.fail_ingestion(f"Validation failed: {error_msg}")
                processing_results['errors'].append(f"Validation failed: {error_msg}")
                return False, f"Validation failed: {error_msg}", processing_results
            tracker.complete_step("validation", {"file_path": file_path})
            processing_results['steps_completed'].append("validation")
            ingestion_logger.log_ingestion_step("validation", document_id, "passed", file_size=os.path.getsize(file_path))

            # Step 3: Text Extraction with Structure
            step = tracker.start_step("text_extraction")
            logger.info(f"📄 Extracting text with structure from: {filename}")
            extracted_text, extracted_content = self.text_extractor.extract_text_with_structure(file_path, original_filename=original_filename)
            if not extracted_text.strip():
                tracker.fail_step("text_extraction", "No text could be extracted")
                tracker.fail_ingestion("No text could be extracted from document")
                processing_results['errors'].append("No text could be extracted from document")
                return False, "No text could be extracted from document", processing_results
            
            # Enhance metadata with custom data
            base_metadata = extracted_content.metadata
            enhanced_metadata = self._enhance_metadata(base_metadata, file_path, custom_metadata)
            
            tracker.complete_step("text_extraction", {
                "text_length": len(extracted_text),
                "structural_units": len(extracted_content.structural_units),
                "extraction_method": extracted_content.extraction_info.get('extraction_method', 'unknown')
            })
            processing_results['steps_completed'].append("text_extraction")
            ingestion_logger.log_ingestion_step("text_extraction", document_id, f"extracted {len(extracted_text)} chars, {len(extracted_content.structural_units)} units")

            # Step 4: Chunking with Structural Awareness
            step = tracker.start_step("chunking")
            logger.info(f"✂️ Chunking document with structural awareness: {filename}")
            chunks = self.chunker.chunk_document_with_structure(extracted_content, document_id)

            if not chunks:
                tracker.fail_step("chunking", "No chunks produced")
                tracker.fail_ingestion("Document chunking produced no chunks")
                processing_results['errors'].append("Document chunking produced no chunks")
                return False, "Document chunking produced no chunks", processing_results
            
            tracker.complete_step("chunking", {"chunk_count": len(chunks)})
            processing_results['steps_completed'].append("chunking")
            processing_results['chunks_created'] = len(chunks)
            ingestion_logger.log_ingestion_step("chunking", document_id, f"created {len(chunks)} chunks")
            
            # Step 5: Metadata Enrichment (Citation Info)
            step = tracker.start_step("metadata_enrichment")
            logger.info(f"🏷️ Enriching chunks with citation metadata: {filename}")
            enriched_chunks = self.metadata_enricher.enrich_chunks(chunks, enhanced_metadata, extracted_content.extraction_info)
            
            # Validate citation metadata
            citation_validation = self.metadata_enricher.validate_citation_metadata(enriched_chunks)
            if citation_validation['success_rate'] < 90:
                processing_results['warnings'].append(f"Citation metadata validation: {citation_validation['success_rate']:.1f}% success rate")
            
            tracker.complete_step("metadata_enrichment", {
                "enriched_chunks": len(enriched_chunks),
                "citation_success_rate": citation_validation['success_rate']
            })
            processing_results['steps_completed'].append("metadata_enrichment")
            ingestion_logger.log_ingestion_step("metadata_enrichment", document_id, f"enriched {len(enriched_chunks)} chunks, {citation_validation['success_rate']:.1f}% citation success")
            
            # Step 6: Embedding Generation
            step = tracker.start_step("embedding_generation")
            logger.info(f"🧠 Generating embeddings for {len(enriched_chunks)} chunks: {filename}")
            chunks_with_embeddings = self._generate_embeddings_with_persistence(enriched_chunks, document_id, file_path, force_reprocess)
            tracker.complete_step("embedding_generation", {
                "chunks_with_embeddings": len(chunks_with_embeddings),
                "total_chunks": len(enriched_chunks)
            })
            processing_results['steps_completed'].append("embedding_generation")
            ingestion_logger.log_ingestion_step("embedding_generation", document_id, f"generated embeddings for {len(chunks_with_embeddings)} chunks")
            
            # Step 7: Storage
            step = tracker.start_step("storage")
            logger.info(f"💾 Storing {len(chunks_with_embeddings)} chunks in storage: {filename}")
            
            # Store in ChromaDB
            chromadb_success = chroma_db.add_chunks(chunks_with_embeddings)
            if not chromadb_success:
                tracker.fail_step("storage", "Failed to store chunks in ChromaDB")
                tracker.fail_ingestion("Failed to store chunks in ChromaDB")
                processing_results['errors'].append("Failed to store chunks in ChromaDB")
                return False, "Failed to store chunks in ChromaDB", processing_results
            
            # Store in SQLite
            sqlite_success = self._store_in_sqlite(document_id, enhanced_metadata, chunks_with_embeddings, uploader_id)
            if not sqlite_success:
                processing_results['warnings'].append("Failed to store some metadata in SQLite")
            
            tracker.complete_step("storage", {
                "chromadb_success": chromadb_success,
                "sqlite_success": sqlite_success,
                "stored_chunks": len(chunks_with_embeddings)
            })
            processing_results['steps_completed'].append("storage")
            ingestion_logger.log_ingestion_step("storage", document_id, f"stored {len(chunks_with_embeddings)} chunks in ChromaDB & SQLite")
            
            # Step 8: Summarization
            step = tracker.start_step("summarization")
            logger.info(f"📝 Generating summaries: {filename}")
            try:
                summary_result = self.summarizer.summarize_document(document_id, chunks_with_embeddings, force_regenerate=force_reprocess)
                
                # Store summaries in SQLite
                if summary_result:
                    self._store_summaries_in_sqlite(document_id, summary_result)
                
                tracker.complete_step("summarization", {
                    "summary_generated": True,
                    "chunk_summaries": len(summary_result.get("chunk_summaries", [])),
                    "final_summary_length": len(summary_result.get("final_summary", {}).get("summary_text", ""))
                })
                processing_results['steps_completed'].append("summarization")
                ingestion_logger.log_ingestion_step("summarization", document_id, f"generated {len(summary_result.get('chunk_summaries', []))} chunk summaries")
                
            except Exception as e:
                logger.warning(f"Summary generation failed for document {document_id}: {str(e)}")
                tracker.complete_step("summarization", {"summary_generated": False, "error": str(e)})
                processing_results['warnings'].append(f"Summary generation failed: {str(e)}")
                ingestion_logger.log_ingestion_step("summarization", document_id, f"failed: {str(e)}")
            
            # Step 9: Completion
            step = tracker.start_step("completion")
            logger.info(f"✅ Completing ingestion: {filename}")
            
            # Determine citation mode
            citation_mode = self._determine_citation_mode(enhanced_metadata.file_extension)
            processing_results['citation_mode'] = citation_mode
            
            # Get page count
            page_count = enhanced_metadata.page_count or len(extracted_content.structural_units)
            processing_results['pages'] = page_count
            
            # Mark document as processed
            self._mark_document_processed(document_id, file_path)

            # Complete ingestion
            processing_time = time.time() - start_time
            processing_results['processing_time'] = processing_time
            
            message = f"Successfully processed document: {len(chunks_with_embeddings)} chunks created, {page_count} pages, citation mode: {citation_mode}"
            tracker.complete_ingestion(metadata=processing_results)
            
            logger.info(f"🎉 BookMate ingestion completed successfully for {filename} in {processing_time:.2f}s")

            # Log pipeline completion
            ingestion_logger.log_pipeline_end("ingestion", document_id, processing_time, "completed",
                                           chunks_created=len(chunks_with_embeddings), pages=page_count)

            unregister_tracker(document_id)
            return True, message, processing_results

        except Exception as e:
            error_msg = f"Error processing document {file_path}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            tracker.fail_ingestion(error_msg)
            processing_results['errors'].append(error_msg)
            processing_results['processing_time'] = time.time() - start_time
            
            unregister_tracker(document_id)
            return False, error_msg, processing_results

    def _enhance_metadata(
        self,
        base_metadata: DocumentMetadata,
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """Enhance metadata with additional information"""
        try:
            # Fix filename to use original name instead of temporary file name
            original_filename = os.path.basename(file_path)
            # Handle various temporary filename patterns
            if original_filename.endswith('.tmp') and '_' in original_filename:
                # Extract original filename from patterns like:
                # "DSM-5 Criteria for SCID-5 Disorders.pdf_1758272787.0150933.tmp"
                # "tmp_DSM-5 Criteria for SCID-5 Disorders.pdf_1758272787.0150933.tmp"
                parts = original_filename.split('_')
                if len(parts) > 1:
                    # Find the part that looks like a filename with extension
                    for i in range(len(parts)):
                        potential_filename = '_'.join(parts[i:])
                        if '.' in potential_filename and not potential_filename.startswith('tmp'):
                            original_filename = potential_filename
                            break
                    else:
                        # If no clear filename found, remove .tmp extension
                        original_filename = original_filename[:-4]
            
            base_metadata.filename = original_filename
            base_metadata.file_path = original_filename  # Use original filename as file path
            
            # Add custom metadata if provided
            if custom_metadata:
                for key, value in custom_metadata.items():
                    if hasattr(base_metadata, key):
                        setattr(base_metadata, key, value)
                    else:
                        base_metadata.custom_metadata[key] = value

            # Add processing timestamp
            base_metadata.custom_metadata['processed_at'] = datetime.now().isoformat()
            base_metadata.custom_metadata['processor_version'] = '2.0.0'
            base_metadata.custom_metadata['processor_type'] = 'BookMate'

            return base_metadata
        
        except Exception as e:
            logger.error(f"Error enhancing metadata: {str(e)}")
            return base_metadata

    def _generate_embeddings_with_persistence(self, chunks: List[DocumentChunk], document_id: str, file_path: str, force_reprocess: bool = False) -> List[DocumentChunk]:
        """Generate embeddings for document chunks with persistence checking"""
        try:
            # Import here to avoid dependency issues
            from sentence_transformers import SentenceTransformer

            # Load model (cached persistently)
            model_name = config.model.embedding_model
            model_cache_key = f"embedding_model_name_{model_name}"

            cached_model_name = cache_manager.get(model_cache_key)
            if cached_model_name is None:
                logger.info(f"Loading embedding model: {model_name}")
                model = SentenceTransformer(model_name)
                # Cache only model name, not the object
                cache_manager.set(model_cache_key, model_name, ttl_hours=24*7)
                logger.info("Embedding model cached for future use")
            else:
                # Load model from cached name
                model = SentenceTransformer(model_name)

            # Prepare cache key for embeddings (used in both cache check and storage)
            file_hash = self._get_file_hash(file_path)
            embedding_cache_key = f"embeddings_{document_id}_{file_hash}"
            
            # Check for existing embeddings in cache (skip if force reprocessing)
            if not force_reprocess:
                cached_embeddings = cache_manager.get(embedding_cache_key)
                if cached_embeddings and len(cached_embeddings) == len(chunks):
                    logger.info(f"Using cached embeddings for {len(chunks)} chunks")
                    # Attach cached embeddings to chunks
                    for chunk, embedding in zip(chunks, cached_embeddings):
                        chunk.embedding = embedding
                    return chunks
            else:
                logger.info(f"Force reprocessing enabled - generating new embeddings for {len(chunks)} chunks")

            # Generate new embeddings
            logger.info(f"Generating new embeddings for {len(chunks)} chunks")
            texts = [chunk.content for chunk in chunks]
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)

            # Convert to list format and attach to chunks
            embeddings_list = [embedding.tolist() for embedding in embeddings]

            for chunk, embedding in zip(chunks, embeddings_list):
                chunk.embedding = embedding

            # Cache embeddings for future use
            cache_manager.set(embedding_cache_key, embeddings_list, ttl_hours=24*30)  # Cache for 30 days
            logger.info(f"Embeddings generated and cached for document {document_id}")

            return chunks

        except ImportError as e:
            logger.error(f"SentenceTransformers not installed: {str(e)}")
            logger.warning("Skipping embedding generation - chunks will be stored without embeddings")
            return chunks
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            logger.warning("Skipping embedding generation - chunks will be stored without embeddings")
            return chunks

    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content for change detection"""
        try:
            import hashlib
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {str(e)}")
            # Fallback to file path and modification time
            path = Path(file_path)
            return hashlib.sha256(f"{file_path}{path.stat().st_mtime}".encode()).hexdigest()[:16]

    def _store_in_sqlite(
        self, 
        document_id: str, 
        metadata: DocumentMetadata, 
        chunks: List[DocumentChunk],
        uploader_id: str = None
    ) -> bool:
        """Store document metadata and citation information in SQLite"""
        try:
            # Store document metadata
            citation_mode = self._determine_citation_mode(metadata.file_extension)
            success = sqlite_storage.store_document_metadata(document_id, metadata, uploader_id, citation_mode)
            
            if success:
                # Store chunks content
                chunks_success = sqlite_storage.store_chunks(document_id, chunks)
                if not chunks_success:
                    logger.warning(f"Failed to store chunks in SQLite for {document_id}")

                # Store citation metadata
                sqlite_storage.store_citation_metadata(document_id, chunks)
                logger.info(f"Stored document metadata, chunks, and citation info in SQLite for {document_id}")

            return success
            
        except Exception as e:
            logger.error(f"Error storing in SQLite: {str(e)}")
            return False
    
    def _store_summaries_in_sqlite(self, document_id: str, summary_result: Dict[str, Any]):
        """Store summaries in SQLite"""
        try:
            # Store document summary
            if 'final_summary' in summary_result and summary_result['final_summary']:
                doc_summary = DocumentSummary(
                    summary_id=str(uuid.uuid4()),
                    document_id=document_id,
                    summary_text=summary_result['final_summary'].get('summary_text', ''),
                    chunk_count=len(summary_result.get('chunk_summaries', [])),
                    word_count=len(summary_result['final_summary'].get('summary_text', '').split()),
                    processing_time=summary_result['final_summary'].get('processing_time', 0)
                )
                sqlite_storage.store_document_summary(doc_summary)
            
            # Store chunk summaries
            if 'chunk_summaries' in summary_result and summary_result['chunk_summaries']:
                chunk_summaries = []
                for i, chunk_summary in enumerate(summary_result['chunk_summaries']):
                    chunk_summary_obj = ChunkSummary(
                        chunk_summary_id=str(uuid.uuid4()),
                        document_id=document_id,
                        chunk_id=chunk_summary.get('chunk_id', f'chunk_{i}'),
                        chunk_index=i,
                        summary_text=chunk_summary.get('summary_text', ''),
                        word_count=len(chunk_summary.get('summary_text', '').split()),
                        processing_time=chunk_summary.get('processing_time', 0)
                    )
                    chunk_summaries.append(chunk_summary_obj)
                
                sqlite_storage.store_chunk_summaries(chunk_summaries)
            
            logger.info(f"Stored summaries in SQLite for {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing summaries in SQLite: {str(e)}")
    
    def _determine_citation_mode(self, file_extension: str) -> str:
        """Determine citation mode based on file extension"""
        extension = file_extension.lower()
        if extension == '.pdf':
            return 'page'
        elif extension in ['.docx', '.html', '.md']:
            return 'heading'
        elif extension in ['.txt', '.rtf']:
            return 'paragraph'
        else:
            return 'chunk'
    
    def _mark_document_processed(self, document_id: str, file_path: str):
        """Mark document as processed with timestamp"""
        try:
            file_mtime = Path(file_path).stat().st_mtime
            cache_key = f"doc_processed_{document_id}"

            cache_manager.set(
                cache_key,
                {
                    'document_id': document_id,
                    'file_path': file_path,
                    'file_mtime': file_mtime,
                    'processed_at': datetime.now().isoformat(),
                    'processor_type': 'BookMate'
                },
                ttl_hours=config.cache.cache_ttl_hours
            )
        except Exception as e:
            logger.warning(f"Error marking document as processed: {str(e)}")
    
    def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status for a document"""
        try:
            # Check SQLite for document metadata
            doc_metadata = sqlite_storage.get_document_metadata(document_id)
            if doc_metadata:
                return {
                    'document_id': document_id,
                    'status': doc_metadata.get('status', 'unknown'),
                    'uploaded_at': doc_metadata.get('uploaded_at'),
                    'processed_at': doc_metadata.get('processed_at'),
                    'citation_mode': doc_metadata.get('citation_mode', 'unknown')
                }

            return None
        except Exception as e:
            logger.error(f"Error getting document status: {str(e)}")
            return None

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and all its chunks from the system"""
        try:
            # Remove from ChromaDB
            chromadb_success = chroma_db.delete_document(document_id)
            
            # Remove from SQLite
            sqlite_success = sqlite_storage.delete_document(document_id)
            
            if chromadb_success or sqlite_success:
                # Remove from cache
                cache_manager.set(f"doc_processed_{document_id}", None, ttl_hours=0)
                cache_manager.set(f"doc_status_{document_id}", None, ttl_hours=0)

                logger.info(f"Successfully removed document: {document_id}")
                return True
            else:
                logger.error(f"Failed to remove document from storage: {document_id}")
                return False

        except Exception as e:
            logger.error(f"Error removing document {document_id}: {str(e)}")
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        try:
            # Get ChromaDB stats
            chromadb_stats = chroma_db.get_document_stats()
            
            # Get SQLite stats
            sqlite_stats = sqlite_storage.get_document_stats()

            return {
                'chromadb': chromadb_stats,
                'sqlite': sqlite_stats,
                'database_path': config.database.chroma_db_path,
                'sqlite_path': config.database.sqlite_db_path,
                'supported_extensions': self.validator.allowed_extensions,
                'max_document_size_mb': config.ingestion.max_document_size_mb,
                'chunk_size': config.ingestion.chunk_size,
                'chunk_overlap': config.ingestion.chunk_overlap,
                'processor_type': 'BookMate'
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {}

    def _clear_existing_document_data(self, document_id: str) -> bool:
        """Clear all existing data for a document (for force reprocessing)"""
        try:
            logger.info(f"🗑️ Clearing existing data for document: {document_id}")
            
            # Remove from ChromaDB (chunks and embeddings)
            chromadb_success = chroma_db.delete_document(document_id)
            if chromadb_success:
                logger.info(f"✅ Removed document chunks and embeddings from ChromaDB: {document_id}")
            else:
                logger.info(f"ℹ️ No existing data found in ChromaDB for: {document_id}")
            
            # Remove from SQLite (metadata, summaries, citations)
            sqlite_success = sqlite_storage.delete_document(document_id)
            if sqlite_success:
                logger.info(f"✅ Removed document metadata and summaries from SQLite: {document_id}")
            else:
                logger.info(f"ℹ️ No existing data found in SQLite for: {document_id}")
            
            # Clear cache entries
            cache_keys_to_clear = [
                f"doc_processed_{document_id}",
                f"doc_status_{document_id}",
                f"embeddings_{document_id}",
                f"summary_{document_id}",
                f"chunks_{document_id}"
            ]
            
            cleared_cache_count = 0
            for cache_key in cache_keys_to_clear:
                if cache_manager.get(cache_key) is not None:
                    cache_manager.set(cache_key, None, ttl_hours=0)
                    cleared_cache_count += 1
            
            if cleared_cache_count > 0:
                logger.info(f"✅ Cleared {cleared_cache_count} cache entries for: {document_id}")
            
            # Clear embedding cache files
            try:
                import glob
                embedding_cache_pattern = f"*{document_id}*"
                cache_files = glob.glob(os.path.join(config.cache.embedding_cache_path, embedding_cache_pattern))
                for cache_file in cache_files:
                    try:
                        os.remove(cache_file)
                        logger.debug(f"Removed embedding cache file: {cache_file}")
                    except OSError:
                        pass
            except Exception as e:
                logger.warning(f"Error clearing embedding cache files: {str(e)}")
            
            logger.info(f"✅ Successfully cleared all existing data for document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing existing data for document {document_id}: {str(e)}")
            return False


# Standalone testing function
def test_document_processor():
    """Test the document processor with sample data"""
    try:
        # Create a test text file
        test_file = "/tmp/test_document_processor.txt"
        test_content = """Introduction

This is the first paragraph of the document. It contains some important information about the topic.

Methodology

This section describes the methodology used in the research. It includes various approaches and techniques.

Results

The results show significant findings. The data analysis reveals important patterns and trends.

Conclusion

In conclusion, the research provides valuable insights into the topic. Future work should focus on extending these findings."""
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test processor
        processor = DocumentProcessor()
        success, message, results = processor.process_document(test_file, uploader_id="test_user")
        
        print("Document Processor Test Results:")
        print(f"Success: {success}")
        print(f"Message: {message}")
        print(f"Results: {results}")
        
        # Cleanup
        os.remove(test_file)
        print("Test completed successfully")
        
        return success
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test
    test_document_processor()