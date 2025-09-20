"""
Database operations for RAG Agent
Handles ChromaDB setup, document storage, and retrieval operations
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import numpy as np

from src.core.document_models import DocumentChunk, DocumentMetadata, RetrievalResult, CacheEntry
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import config
from src.core.service_manager import service_manager

logger = logging.getLogger(__name__)


class ChromaDatabase:
    """ChromaDB wrapper for document storage and retrieval"""

    def __init__(self, persist_directory: str = None):
        """Initialize ChromaDB client"""
        self.persist_directory = persist_directory or config.database.chroma_db_path
        self.collection_name = config.database.chroma_collection_name

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

        logger.info(f"ChromaDB initialized at {self.persist_directory}")

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one with robust error handling"""
        try:
            # First try to get the collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")

            # Validate collection is accessible
            try:
                count = collection.count()
                logger.debug(f"Collection validation successful: {count} documents")
            except Exception as validation_error:
                logger.warning(f"Collection exists but validation failed: {validation_error}")
                # Try to recreate the collection
                logger.info("Attempting to recreate collection due to validation failure")
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info("Deleted corrupted collection")
                except Exception:
                    pass  # Collection might not exist

                collection = self._create_new_collection()

        except (ValueError, Exception) as e:
            # Collection doesn't exist, create it
            error_msg = str(e).lower()
            if ("does not exist" in error_msg or
                "not found" in error_msg or
                "collection" in error_msg and ("not" in error_msg or "invalid" in error_msg)):
                logger.info(f"Collection {self.collection_name} not found, creating new one")
                collection = self._create_new_collection()
            else:
                logger.error(f"Unexpected error accessing collection: {str(e)}")
                raise

        return collection

    def _create_new_collection(self) -> chromadb.Collection:
        """Create a new collection with proper metadata"""
        from datetime import datetime
        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "RAG Agent document collection",
                "created_at": str(datetime.now()),
                "version": "1.0",
                "embedding_model": config.model.embedding_model
            }
        )
        logger.info(f"Created new collection: {self.collection_name}")
        return collection

    def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the database"""
        try:
            if not chunks:
                logger.warning("No chunks provided to add")
                return False

            # Ensure we have a valid collection (re-get if needed)
            try:
                # Test if collection is accessible
                self.collection.count()
            except Exception:
                # Collection might be invalid, get a fresh one
                logger.info("Collection reference invalid, getting fresh collection")
                self.collection = self._get_or_create_collection()

            # Separate chunks with and without embeddings
            chunks_with_embeddings = []
            chunks_without_embeddings = []

            for chunk in chunks:
                if chunk.embedding is not None:
                    chunks_with_embeddings.append(chunk)
                else:
                    chunks_without_embeddings.append(chunk)
                    logger.debug(f"Chunk {chunk.chunk_id} has no embedding, will store without embedding for BM25 fallback")

            # Add chunks with embeddings first
            if chunks_with_embeddings:
                ids = []
                documents = []
                embeddings = []
                metadatas = []

                for chunk in chunks_with_embeddings:
                    ids.append(chunk.chunk_id)
                    documents.append(chunk.content)

                    # Convert numpy array to list if needed
                    if isinstance(chunk.embedding, np.ndarray):
                        embedding_list = chunk.embedding.tolist()
                    else:
                        embedding_list = chunk.embedding
                    embeddings.append(embedding_list)

                    # Prepare metadata
                    metadata = self._prepare_chunk_metadata(chunk)
                    metadatas.append(metadata)

                # Add to ChromaDB
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(chunks_with_embeddings)} chunks with embeddings")

            # Add chunks without embeddings (for BM25 fallback)
            if chunks_without_embeddings:
                ids = []
                documents = []
                metadatas = []

                for chunk in chunks_without_embeddings:
                    ids.append(chunk.chunk_id)
                    documents.append(chunk.content)
                    # Prepare metadata
                    metadata = self._prepare_chunk_metadata(chunk)
                    metadatas.append(metadata)

                # Add to ChromaDB without embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(chunks_without_embeddings)} chunks without embeddings (BM25 fallback)")

            if not chunks_with_embeddings and not chunks_without_embeddings:
                logger.warning("No valid chunks found")
                return False

            logger.info(f"Added {len(chunks_with_embeddings) + len(chunks_without_embeddings)} chunks to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {str(e)}")
            return False

    def _prepare_chunk_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Prepare metadata dictionary for a chunk"""
        metadata = {
            "document_id": chunk.document_id,
            "chunk_index": str(chunk.chunk_index),
            "start_position": str(chunk.start_position),
            "end_position": str(chunk.end_position),
            "token_count": str(chunk.token_count),
            "created_at": chunk.created_at.isoformat() if chunk.created_at else "",
        }

        # Add document metadata if available
        if chunk.metadata:
            metadata.update({
                "filename": chunk.metadata.filename,
                "file_path": chunk.metadata.file_path,
                "file_size": str(chunk.metadata.file_size),
                "file_extension": chunk.metadata.file_extension,
                "mime_type": chunk.metadata.mime_type,
                "title": chunk.metadata.title or "",
                "author": chunk.metadata.author or "",
                "language": chunk.metadata.language or "en",
                "page_count": str(chunk.metadata.page_count) if chunk.metadata.page_count else "0",
                "creation_date": chunk.metadata.creation_date.isoformat() if chunk.metadata.creation_date else "",
                "modification_date": chunk.metadata.modification_date.isoformat() if chunk.metadata.modification_date else "",
                "source_url": chunk.metadata.source_url or "",
                "tags": json.dumps(chunk.metadata.tags) if chunk.metadata.tags else "[]",
                "custom_metadata": json.dumps(chunk.metadata.custom_metadata) if chunk.metadata.custom_metadata else "{}"
            })

        return metadata

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = None,
        similarity_threshold: float = None,
        document_filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar chunks using embedding"""
        try:
            if top_k is None:
                top_k = config.retrieval.top_k_retrieval
            if similarity_threshold is None:
                similarity_threshold = config.retrieval.similarity_threshold

            # Prepare ChromaDB query with optional filtering
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ['documents', 'metadatas', 'distances']
            }

            # Add document filtering if provided
            if document_filters:
                where_clause = {}
                if 'document_id' in document_filters:
                    # Legacy single document filter
                    where_clause['document_id'] = document_filters['document_id']
                elif 'document_ids' in document_filters:
                    # New multiple document filter - ChromaDB supports IN operator
                    doc_ids = document_filters['document_ids']
                    if doc_ids:
                        where_clause['document_id'] = {"$in": doc_ids}

                if where_clause:
                    query_params['where'] = where_clause
                    logger.info(f"Applying document filters: {where_clause}")

            # Query ChromaDB
            results = self.collection.query(**query_params)

            retrieval_results = []

            if results['documents'] and len(results['documents']) > 0 and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    # ChromaDB uses cosine distance by default (0 = identical, 2 = opposite)
                    # Convert to similarity score (0 = no similarity, 1 = identical)
                    similarity_score = max(0, 1 - (distance / 2))

                    # Apply similarity threshold (lower threshold for better recall)
                    if similarity_score < 0.01:  # Very low threshold to allow more results through
                        continue

                    # Reconstruct chunk from metadata
                    chunk = DocumentChunk(
                        chunk_id=results['ids'][0][i],
                        document_id=metadata.get('document_id', ''),
                        content=doc,
                        chunk_index=int(metadata.get('chunk_index', 0)),
                        start_position=int(metadata.get('start_position', 0)),
                        end_position=int(metadata.get('end_position', 0)),
                        token_count=int(metadata.get('token_count', 0)),
                        metadata=DocumentMetadata(
                            filename=metadata.get('filename', ''),
                            file_path='',  # Not stored in metadata
                            file_size=0,   # Not stored in metadata
                            file_extension=metadata.get('file_extension', ''),
                            mime_type='',  # Not stored in metadata
                            title=metadata.get('title'),
                            author=metadata.get('author'),
                            language=metadata.get('language'),
                            source_url=metadata.get('source_url'),
                            tags=json.loads(metadata.get('tags', '[]'))
                        )
                    )

                    retrieval_result = RetrievalResult(
                        chunk_id=results['ids'][0][i],
                        document_id=metadata.get('document_id', ''),
                        content=doc,
                        relevance_score=similarity_score,
                        semantic_score=similarity_score,
                        keyword_score=0.0,
                        final_score=similarity_score,
                        citation=None,
                        metadata=metadata,
                        ranking_position=i,
                        retrieval_method="semantic"
                    )

                    retrieval_results.append(retrieval_result)

            logger.info(f"Retrieved {len(retrieval_results)} chunks with similarity >= 0.01")
            return retrieval_results

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            # First try to find by document_id
            results = self.collection.get(
                where={"document_id": document_id},
                include=['metadatas']
            )

            # If not found by document_id, try to find by filename (exact match)
            if not results['ids']:
                logger.info(f"No chunks found for document_id {document_id}, trying filename search")
                results = self.collection.get(
                    where={"filename": document_id},
                    include=['metadatas']
                )

            # If still not found, try to find by partial filename match
            if not results['ids']:
                logger.info(f"No exact filename match, trying partial match for {document_id}")
                # Get all documents and filter by filename
                all_results = self.collection.get(include=['metadatas'])
                matching_ids = []
                
                for i, metadata in enumerate(all_results['metadatas']):
                    filename = metadata.get('filename', '')
                    if document_id.lower() in filename.lower():
                        matching_ids.append(all_results['ids'][i])
                
                if matching_ids:
                    # Delete all matching chunks
                    self.collection.delete(ids=matching_ids)
                    logger.info(f"Deleted {len(matching_ids)} chunks for documents matching {document_id}")
                    return True

            if not results['ids']:
                logger.warning(f"No chunks found for document {document_id}")
                return False

            # Delete chunks
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    def get_document_stats(self, document_id: str = None) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            if document_id:
                # Stats for specific document - try document_id first, then filename
                results = self.collection.get(
                    where={"document_id": document_id},
                    include=['metadatas']
                )
                
                # If not found by document_id, try exact filename match
                if not results['ids']:
                    results = self.collection.get(
                        where={"filename": document_id},
                        include=['metadatas']
                    )
                
                # If still not found, try partial filename match
                if not results['ids']:
                    all_results = self.collection.get(include=['metadatas'])
                    matching_metadatas = []
                    
                    for i, metadata in enumerate(all_results['metadatas']):
                        filename = metadata.get('filename', '')
                        if document_id.lower() in filename.lower():
                            matching_metadatas.append(metadata)
                    
                    if matching_metadatas:
                        return {
                            "document_id": document_id,
                            "chunk_count": len(matching_metadatas),
                            "total_tokens": sum(int(m.get('token_count', 0)) for m in matching_metadatas)
                        }
                
                if not results['ids']:
                    return {}
                
                return {
                    "document_id": document_id,
                    "chunk_count": len(results['ids']),
                    "total_tokens": sum(int(m.get('token_count', 0)) for m in results['metadatas'])
                }
            else:
                # Overall stats
                results = self.collection.get(include=['metadatas'])
                doc_ids = set(m.get('document_id', '') for m in results['metadatas'])

                return {
                    "total_chunks": len(results['ids']),
                    "total_documents": len(doc_ids),
                    "total_tokens": sum(int(m.get('token_count', 0)) for m in results['metadatas'])
                }

        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {}

    def get_document_chunks(self, document_id: str) -> List[Any]:
        """Get all chunks for a specific document"""
        try:
            # Get all results and filter manually to ensure we find everything
            all_results = self.collection.get(include=['metadatas', 'documents', 'embeddings'])
            matching_indices = []
            
            for i, metadata in enumerate(all_results['metadatas']):
                # Check if this chunk belongs to the document
                try:
                    doc_id = metadata.get('document_id', '')
                    filename = metadata.get('filename', '')
                    
                    # Safe string comparisons
                    if (doc_id == document_id or 
                        filename == document_id or
                        (isinstance(doc_id, str) and document_id in doc_id) or
                        (isinstance(filename, str) and document_id in filename)):
                        matching_indices.append(i)
                except Exception as e:
                    logger.warning(f"Error checking metadata for chunk {i}: {str(e)}")
                    continue
            
            if not matching_indices:
                logger.warning(f"No chunks found for document {document_id}")
                return []
            
            # Filter results to only matching chunks
            results = {
                'ids': [all_results['ids'][i] for i in matching_indices],
                'metadatas': [all_results['metadatas'][i] for i in matching_indices],
                'documents': [all_results['documents'][i] for i in matching_indices],
                'embeddings': [all_results['embeddings'][i] for i in matching_indices] if all_results.get('embeddings') is not None else None
            }
            
            if not results['ids']:
                logger.warning(f"No chunks found for document {document_id}")
                return []
            
            # Convert to DocumentChunk objects
            from src.core.document_models import DocumentChunk, DocumentMetadata
            chunks = []
            
            for i, chunk_id in enumerate(results['ids']):
                try:
                    metadata_dict = results['metadatas'][i]
                    content = results['documents'][i]
                    embedding = results['embeddings'][i] if results.get('embeddings') is not None else None
                    
                    # Create DocumentMetadata object
                    doc_metadata = DocumentMetadata(
                        filename=metadata_dict.get('filename', ''),
                        file_path=metadata_dict.get('file_path', ''),
                        file_size=metadata_dict.get('file_size', 0),
                        file_extension=metadata_dict.get('file_extension', ''),
                        mime_type=metadata_dict.get('mime_type', ''),
                        title=metadata_dict.get('title', ''),
                        author=metadata_dict.get('author', ''),
                        creation_date=metadata_dict.get('creation_date'),
                        modification_date=metadata_dict.get('modification_date'),
                        page_count=metadata_dict.get('page_count', 0),
                        custom_metadata=metadata_dict.get('custom_metadata', {})
                    )
                    
                    # Create DocumentChunk object
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=metadata_dict.get('document_id', document_id),
                        content=content,
                        chunk_index=metadata_dict.get('chunk_index', i),
                        start_position=metadata_dict.get('start_position', 0),
                        end_position=metadata_dict.get('end_position', len(content)),
                        token_count=metadata_dict.get('token_count', 0),
                        metadata=doc_metadata
                    )
                    
                    # Add embedding if available
                    try:
                        if embedding is not None:
                            # Convert to list if it's a numpy array or similar
                            if hasattr(embedding, 'tolist'):
                                chunk.embedding = embedding.tolist()
                            elif isinstance(embedding, (list, tuple)):
                                chunk.embedding = list(embedding)
                            else:
                                chunk.embedding = embedding
                    except Exception as embed_error:
                        logger.warning(f"Could not set embedding for chunk {chunk_id}: {embed_error}")
                    
                    chunks.append(chunk)
                    
                except Exception as e:
                    logger.error(f"Error creating chunk {i} ({chunk_id}): {str(e)}")
                    continue
            
            # Sort chunks by chunk_index
            chunks.sort(key=lambda x: x.chunk_index)
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks for {document_id}: {str(e)}")
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection"""
        try:
            results = self.collection.get(include=['metadatas'])
            
            # Group chunks by document
            documents = {}
            for i, metadata in enumerate(results['metadatas']):
                doc_id = metadata.get('document_id', '')
                filename = metadata.get('filename', '')
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': filename,
                        'chunk_count': 0,
                        'total_tokens': 0
                    }
                
                documents[doc_id]['chunk_count'] += 1
                documents[doc_id]['total_tokens'] += int(metadata.get('token_count', 0))
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def reset_collection(self) -> bool:
        """Reset/clear the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False

    def repair_collection(self) -> bool:
        """Attempt to repair a corrupted collection"""
        try:
            logger.info(f"Attempting to repair collection: {self.collection_name}")

            # Step 1: Backup current state
            try:
                current_count = self.collection.count()
                logger.info(f"Current collection has {current_count} documents")
            except Exception:
                logger.warning("Could not get current collection count")

            # Step 2: Try to reset and recreate
            reset_success = self.reset_collection()
            if reset_success:
                logger.info("Collection repair successful - collection reset and recreated")
                return True
            else:
                logger.error("Collection repair failed - could not reset collection")
                return False

        except Exception as e:
            logger.error(f"Unexpected error during collection repair: {str(e)}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector store"""
        try:
            status = {
                'healthy': False,
                'collection_exists': False,
                'document_count': 0,
                'last_error': None,
                'collection_name': self.collection_name
            }

            # Test collection access
            try:
                count = self.collection.count()
                status['healthy'] = True
                status['collection_exists'] = True
                status['document_count'] = count
                logger.debug(f"Vector store health check passed: {count} documents")
            except Exception as e:
                status['last_error'] = str(e)
                logger.warning(f"Vector store health check failed: {str(e)}")

                # Try to repair
                if "does not exist" in str(e).lower():
                    logger.info("Collection missing, attempting repair...")
                    if self.repair_collection():
                        status['healthy'] = True
                        status['collection_exists'] = True
                        logger.info("Collection repair successful")
                    else:
                        logger.error("Collection repair failed")

            return status

        except Exception as e:
            logger.error(f"Error getting vector store health status: {str(e)}")
            return {
                'healthy': False,
                'collection_exists': False,
                'document_count': 0,
                'last_error': str(e),
                'collection_name': self.collection_name
            }


class CacheManager:
    """Simple file-based cache for embeddings and responses"""

    def __init__(self, cache_dir: str = None):
        """Initialize cache manager"""
        self.cache_dir = Path(cache_dir or config.cache.embedding_cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key"""
        # Create a safe filename from key
        safe_key = "".join(c for c in key if c.isalnum() or c in ('_', '-')).strip()
        if len(safe_key) > 100:
            safe_key = safe_key[:100]
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            # Validate required fields
            if not all(key in data for key in ['key', 'value', 'created_at', 'ttl_hours']):
                raise ValueError("Missing required cache entry fields")

            # Parse datetime from ISO format
            from datetime import datetime
            created_at = datetime.fromisoformat(data['created_at'])
            
            # Create cache entry with parsed datetime
            cache_entry = CacheEntry(
                key=data['key'],
                value=data['value'],
                created_at=created_at,
                ttl_hours=data['ttl_hours']
            )

            if cache_entry.is_expired:
                cache_path.unlink()  # Delete expired cache
                logger.debug(f"Removed expired cache file: {cache_path.name}")
                return None

            return cache_entry.value

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Corrupted cache file for key {key}: {str(e)}")
            # Delete corrupted cache file
            try:
                cache_path.unlink()
                logger.debug(f"Removed corrupted cache file: {cache_path.name}")
            except OSError:
                pass  # File may already be deleted or inaccessible
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading cache for key {key}: {str(e)}")
            return None

    def set(self, key: str, value: Any, ttl_hours: int = None) -> bool:
        """Set cached value"""
        if ttl_hours is None:
            ttl_hours = config.cache.cache_ttl_hours

        cache_entry = CacheEntry(
            key=key,
            value=value,
            ttl_hours=ttl_hours
        )

        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "key": cache_entry.key,
                    "value": cache_entry.value,
                    "created_at": cache_entry.created_at.isoformat(),
                    "ttl_hours": cache_entry.ttl_hours
                }, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing cache for key {key}: {str(e)}")
            return False

    def clear_expired(self) -> int:
        """Clear expired cache entries"""
        cleared_count = 0
        corrupted_count = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                # Validate required fields
                if not all(key in data for key in ['key', 'value', 'created_at', 'ttl_hours']):
                    raise ValueError("Missing required cache entry fields")

                # Parse datetime from ISO format
                from datetime import datetime
                created_at = datetime.fromisoformat(data['created_at'])
                
                # Create cache entry with parsed datetime
                cache_entry = CacheEntry(
                    key=data['key'],
                    value=data['value'],
                    created_at=created_at,
                    ttl_hours=data['ttl_hours']
                )

                if cache_entry.is_expired:
                    cache_file.unlink()
                    cleared_count += 1
                    logger.debug(f"Removed expired cache file: {cache_file.name}")

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                logger.warning(f"Corrupted cache file {cache_file.name}: {str(e)}")
                # Delete corrupted cache files
                try:
                    cache_file.unlink()
                    corrupted_count += 1
                    logger.debug(f"Removed corrupted cache file: {cache_file.name}")
                except OSError as unlink_error:
                    logger.error(f"Failed to delete corrupted cache file {cache_file}: {unlink_error}")
            except Exception as e:
                logger.error(f"Unexpected error processing cache file {cache_file}: {str(e)}")
                # Don't delete files with unexpected errors to avoid data loss
                continue

        logger.info(f"Cleared {cleared_count} expired cache entries and {corrupted_count} corrupted files")
        return cleared_count + corrupted_count

    def clear_all(self) -> int:
        """Clear ALL cache entries"""
        cleared_count = 0
        
        try:
            # Delete all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {str(e)}")
            
            logger.info(f"Cleared ALL {cleared_count} cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing all cache entries: {str(e)}")
            return 0


# Global instances
chroma_db = ChromaDatabase()
cache_manager = CacheManager()
