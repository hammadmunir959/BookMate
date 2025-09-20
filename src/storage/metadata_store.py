"""
SQLite Storage Layer for BookMate RAG Agent
Stores document-level metadata, summaries, and citation information
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_models import DocumentMetadata, DocumentSummary, ChunkSummary
from src.core.config import config
from src.core.service_manager import service_manager

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """SQLite storage for document registry and summaries"""
    
    def __init__(self, db_path: str = None):
        """Initialize SQLite storage"""
        self.db_path = db_path or config.database.sqlite_db_path
        self._ensure_db_directory()
        self._initialize_database()
        logger.info(f"SQLiteStorage initialized at {self.db_path}")
    
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        if db_dir != Path('.'):  # Only create directory if it's not the current directory
            db_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Documents table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT,
                        file_size INTEGER,
                        file_extension TEXT,
                        mime_type TEXT,
                        title TEXT,
                        author TEXT,
                        page_count INTEGER,
                        language TEXT DEFAULT 'en',
                        uploader_id TEXT,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed_at TIMESTAMP,
                        status TEXT DEFAULT 'processing',
                        metadata_json TEXT,
                        citation_mode TEXT DEFAULT 'page',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Document summaries table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS document_summaries (
                        summary_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        summary_type TEXT DEFAULT 'comprehensive',
                        chunk_count INTEGER,
                        word_count INTEGER,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id)
                    )
                ''')

                # Chunks table for storing chunk content
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER,
                        token_count INTEGER,
                        start_position INTEGER,
                        end_position INTEGER,
                        metadata TEXT,
                        summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id)
                    )
                ''')

                # Chunk summaries table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chunk_summaries (
                        chunk_summary_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        chunk_index INTEGER,
                        summary_text TEXT NOT NULL,
                        word_count INTEGER,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id)
                    )
                ''')
                
                # Citation metadata table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS citation_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        citation_anchor TEXT,
                        citation_type TEXT,
                        page_number INTEGER,
                        heading TEXT,
                        paragraph_index INTEGER,
                        section_title TEXT,
                        source_location TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents (uploaded_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_summaries_document_id ON document_summaries (document_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_summaries_document_id ON chunk_summaries (document_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_citation_document_id ON citation_metadata (document_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_citation_chunk_id ON citation_metadata (chunk_id)')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def store_document_metadata(
        self, 
        document_id: str, 
        metadata: DocumentMetadata,
        uploader_id: str = None,
        citation_mode: str = "page"
    ) -> bool:
        """
        Store document metadata
        
        Args:
            document_id: Unique document identifier
            metadata: Document metadata
            uploader_id: ID of user who uploaded the document
            citation_mode: Citation mode (page, heading, paragraph, chunk)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare metadata JSON
                metadata_dict = {
                    'title': metadata.title,
                    'author': metadata.author,
                    'creation_date': metadata.creation_date.isoformat() if metadata.creation_date else None,
                    'modification_date': metadata.modification_date.isoformat() if metadata.modification_date else None,
                    'source_url': metadata.source_url,
                    'tags': metadata.tags,
                    'custom_metadata': metadata.custom_metadata
                }
                
                cursor.execute('''
                    INSERT OR REPLACE INTO documents (
                        document_id, filename, file_path, file_size, file_extension,
                        mime_type, title, author, page_count, language, uploader_id,
                        metadata_json, citation_mode, processed_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    document_id,
                    metadata.filename,
                    metadata.file_path,
                    metadata.file_size,
                    metadata.file_extension,
                    metadata.mime_type,
                    metadata.title,
                    metadata.author,
                    metadata.page_count,
                    metadata.language,
                    uploader_id,
                    json.dumps(metadata_dict),
                    citation_mode,
                    datetime.now().isoformat(),
                    'completed'
                ))
                
                conn.commit()
                logger.info(f"Stored document metadata for {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            return False
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document metadata dictionary or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM documents WHERE document_id = ?
                ''', (document_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Convert row to dictionary
                columns = [description[0] for description in cursor.description]
                doc_data = dict(zip(columns, row))
                
                # Parse metadata JSON
                if doc_data.get('metadata_json'):
                    try:
                        doc_data['metadata'] = json.loads(doc_data['metadata_json'])
                    except json.JSONDecodeError:
                        doc_data['metadata'] = {}
                
                return doc_data
                
        except Exception as e:
            logger.error(f"Error getting document metadata: {str(e)}")
            return None
    
    def store_document_summary(self, summary: DocumentSummary) -> bool:
        """
        Store document summary
        
        Args:
            summary: Document summary object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO document_summaries (
                        summary_id, document_id, summary_text, summary_type,
                        chunk_count, word_count, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    summary.summary_id,
                    summary.document_id,
                    summary.summary_text,
                    summary.summary_type,
                    summary.chunk_count,
                    summary.word_count,
                    summary.processing_time
                ))
                
                conn.commit()
                logger.info(f"Stored document summary for {summary.document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing document summary: {str(e)}")
            return False
    
    def get_document_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document summary
        
        Args:
            document_id: Document identifier
            
        Returns:
            Summary data dictionary or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM document_summaries WHERE document_id = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (document_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
                
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            return None
    
    def store_chunk_summaries(self, chunk_summaries: List[ChunkSummary]) -> bool:
        """
        Store chunk summaries
        
        Args:
            chunk_summaries: List of chunk summary objects
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for summary in chunk_summaries:
                    cursor.execute('''
                        INSERT OR REPLACE INTO chunk_summaries (
                            chunk_summary_id, document_id, chunk_id, chunk_index,
                            summary_text, word_count, processing_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        summary.chunk_summary_id,
                        summary.document_id,
                        summary.chunk_id,
                        summary.chunk_index,
                        summary.summary_text,
                        summary.word_count,
                        summary.processing_time
                    ))
                
                conn.commit()
                logger.info(f"Stored {len(chunk_summaries)} chunk summaries")
                return True
                
        except Exception as e:
            logger.error(f"Error storing chunk summaries: {str(e)}")
            return False
    
    def get_chunk_summaries(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get chunk summaries for a document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunk summary dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM chunk_summaries WHERE document_id = ?
                    ORDER BY chunk_index
                ''', (document_id,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting chunk summaries: {str(e)}")
            return []
    
    def store_chunks(self, document_id: str, chunks: List[Any]) -> bool:
        """
        Store document chunks in SQLite

        Args:
            document_id: Document identifier
            chunks: List of document chunks

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for chunk in chunks:
                    # Handle both DocumentChunk objects and dictionaries for compatibility
                    if hasattr(chunk, 'chunk_id'):  # DocumentChunk object
                        chunk_id = chunk.chunk_id
                        content = chunk.content
                        chunk_index = chunk.chunk_index
                        token_count = chunk.token_count
                        start_position = chunk.start_position
                        end_position = chunk.end_position
                        metadata = chunk.metadata if chunk.metadata else {}
                        summary = chunk.summary if chunk.summary else ''
                    else:  # Dictionary (fallback)
                        chunk_id = chunk.get('chunk_id', '')
                        content = chunk.get('content', '')
                        chunk_index = chunk.get('chunk_index', 0)
                        token_count = chunk.get('token_count', 0)
                        start_position = chunk.get('start_position', 0)
                        end_position = chunk.get('end_position', 0)
                        metadata = chunk.get('metadata', {})
                        summary = chunk.get('summary', '')

                    # Convert DocumentMetadata objects to dictionaries for JSON serialization
                    if metadata and hasattr(metadata, '__dataclass_fields__'):  # DocumentMetadata object
                        metadata = {
                            'filename': metadata.filename,
                            'file_path': metadata.file_path,
                            'file_size': metadata.file_size,
                            'file_extension': metadata.file_extension,
                            'mime_type': metadata.mime_type,
                            'title': metadata.title,
                            'author': metadata.author,
                            'creation_date': metadata.creation_date.isoformat() if metadata.creation_date else None,
                            'modification_date': metadata.modification_date.isoformat() if metadata.modification_date else None,
                            'page_count': metadata.page_count,
                            'language': metadata.language,
                            'source_url': metadata.source_url,
                            'tags': metadata.tags,
                            'custom_metadata': metadata.custom_metadata,
                        }

                    cursor.execute('''
                        INSERT OR REPLACE INTO chunks (
                            chunk_id, document_id, content, chunk_index,
                            token_count, start_position, end_position, metadata, summary
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        chunk_id,
                        document_id,
                        content,
                        chunk_index,
                        token_count,
                        start_position,
                        end_position,
                        json.dumps(metadata),
                        summary
                    ))

                conn.commit()
                logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
                return True

        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            return False

    def store_citation_metadata(self, document_id: str, chunks: List[Any]) -> bool:
        """
        Store citation metadata for chunks
        
        Args:
            document_id: Document identifier
            chunks: List of document chunks with citation metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for chunk in chunks:
                    if (hasattr(chunk, 'metadata') and 
                        chunk.metadata and 
                        chunk.metadata.custom_metadata):
                        
                        custom_meta = chunk.metadata.custom_metadata
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO citation_metadata (
                                document_id, chunk_id, citation_anchor, citation_type,
                                page_number, heading, paragraph_index, section_title,
                                source_location
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            document_id,
                            chunk.chunk_id,
                            custom_meta.get('citation_anchor'),
                            custom_meta.get('citation_type'),
                            custom_meta.get('page_number'),
                            custom_meta.get('heading'),
                            custom_meta.get('paragraph_index'),
                            custom_meta.get('section_title'),
                            custom_meta.get('source_location')
                        ))
                
                conn.commit()
                logger.info(f"Stored citation metadata for {len(chunks)} chunks")
                return True
                
        except Exception as e:
            logger.error(f"Error storing citation metadata: {str(e)}")
            return False
    
    def get_citation_metadata(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get citation metadata for a document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of citation metadata dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM citation_metadata WHERE document_id = ?
                    ORDER BY chunk_id
                ''', (document_id,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting citation metadata: {str(e)}")
            return []
    
    def get_citation_metadata_by_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get citation metadata for a specific chunk
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Citation metadata dictionary or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM citation_metadata WHERE chunk_id = ?
                ''', (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting citation metadata for chunk {chunk_id}: {str(e)}")
            return None
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List documents with pagination
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT document_id, filename, file_size, file_extension,
                           mime_type, title, author, page_count, status,
                           uploaded_at, processed_at, citation_mode
                    FROM documents
                    ORDER BY uploaded_at DESC
                    LIMIT ? OFFSET ?
                ''', (limit, offset))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def keyword_search(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform keyword search on document content and chunks with improved fallback

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching document metadata
        """
        try:
            # Simple keyword search using LIKE operator
            # In a production system, you'd want to use FTS (Full Text Search)
            search_terms = query.lower().split()

            # Create search condition for each term in chunks content
            chunk_conditions = []
            chunk_params = []
            for term in search_terms:
                chunk_conditions.append("LOWER(content) LIKE ?")
                chunk_params.append(f"%{term}%")

            chunk_where_clause = " AND ".join(chunk_conditions) if chunk_conditions else "1=1"

            # Add document filtering if provided
            if filters:
                if 'document_id' in filters:
                    # Legacy single document filter
                    chunk_where_clause = f"({chunk_where_clause}) AND c.document_id = ?"
                    chunk_params.append(filters['document_id'])
                    logger.info(f"Applying document filter: document_id = {filters['document_id']}")
                elif 'document_ids' in filters:
                    # New multiple document filter
                    doc_ids = filters['document_ids']
                    if doc_ids:
                        placeholders = ', '.join(['?' for _ in doc_ids])
                        chunk_where_clause = f"({chunk_where_clause}) AND c.document_id IN ({placeholders})"
                        chunk_params.extend(doc_ids)
                        logger.info(f"Applying document filters: document_ids = {doc_ids}")

            # Search chunks first for content matches
            chunk_sql = f"""
                SELECT DISTINCT c.chunk_id, c.document_id, c.content, c.chunk_index, c.token_count,
                               c.start_position, c.end_position, d.filename, d.title, d.file_size,
                               d.mime_type, d.created_at
                FROM chunks c
                JOIN documents d ON c.document_id = d.document_id
                WHERE {chunk_where_clause}
                ORDER BY d.created_at DESC
                LIMIT ?
            """
            chunk_params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(chunk_sql, chunk_params)
                chunk_results = cursor.fetchall()

                # Convert to list of dictionaries
                columns = ['chunk_id', 'document_id', 'content', 'chunk_index', 'token_count',
                          'start_position', 'end_position', 'filename', 'title', 'file_size', 'mime_type', 'created_at']
                chunks = []
                for row in chunk_results:
                    chunk_dict = dict(zip(columns, row))
                    chunks.append(chunk_dict)

                # If no chunk matches, try more flexible search
                if not chunks and search_terms:
                    logger.info("No exact chunk matches, trying flexible search...")

                    # Try OR search instead of AND
                    chunk_conditions_or = []
                    chunk_params_or = []
                    for term in search_terms:
                        chunk_conditions_or.append("LOWER(content) LIKE ?")
                        chunk_params_or.append(f"%{term}%")

                    chunk_where_clause_or = " OR ".join(chunk_conditions_or) if chunk_conditions_or else "1=1"

                    # Add document filtering
                    if filters:
                        if 'document_id' in filters:
                            chunk_where_clause_or = f"({chunk_where_clause_or}) AND c.document_id = ?"
                            chunk_params_or.append(filters['document_id'])
                        elif 'document_ids' in filters:
                            doc_ids = filters['document_ids']
                            if doc_ids:
                                placeholders = ', '.join(['?' for _ in doc_ids])
                                chunk_where_clause_or = f"({chunk_where_clause_or}) AND c.document_id IN ({placeholders})"
                                chunk_params_or.extend(doc_ids)

                    chunk_sql_or = f"""
                        SELECT DISTINCT c.chunk_id, c.document_id, c.content, c.chunk_index, c.token_count,
                                       c.start_position, c.end_position, d.filename, d.title, d.file_size,
                                       d.mime_type, d.created_at
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.document_id
                        WHERE {chunk_where_clause_or}
                        ORDER BY d.created_at DESC
                        LIMIT ?
                    """
                    chunk_params_or.append(limit)

                    cursor.execute(chunk_sql_or, chunk_params_or)
                    chunk_results_or = cursor.fetchall()

                    for row in chunk_results_or:
                        chunk_dict = dict(zip(columns, row))
                        chunks.append(chunk_dict)

                    if chunk_results_or:
                        logger.info(f"Flexible search found {len(chunk_results_or)} chunks")

                # If still no matches, fallback to document metadata search
                if not chunks:
                    # Create search condition for document metadata
                    doc_conditions = []
                    doc_params = []
                    for term in search_terms:
                        doc_conditions.append("LOWER(title) LIKE ? OR LOWER(filename) LIKE ?")
                        doc_params.extend([f"%{term}%", f"%{term}%"])

                    doc_where_clause = " OR ".join(doc_conditions) if doc_conditions else "1=1"

                    # Add document filtering if provided
                    if filters:
                        if 'document_id' in filters:
                            doc_where_clause = f"({doc_where_clause}) AND document_id = ?"
                            doc_params.append(filters['document_id'])
                        elif 'document_ids' in filters:
                            doc_ids = filters['document_ids']
                            if doc_ids:
                                placeholders = ', '.join(['?' for _ in doc_ids])
                                doc_where_clause = f"({doc_where_clause}) AND document_id IN ({placeholders})"
                                doc_params.extend(doc_ids)

                    doc_sql = f"""
                        SELECT document_id, filename, title, file_size, mime_type, created_at
                        FROM documents
                        WHERE {doc_where_clause}
                        ORDER BY created_at DESC
                        LIMIT ?
                    """
                    doc_params.append(limit)

                    cursor.execute(doc_sql, doc_params)
                    doc_results = cursor.fetchall()

                    # Convert document results to chunk-like results for consistency
                    for row in doc_results:
                        doc_columns = ['document_id', 'filename', 'title', 'file_size', 'mime_type', 'created_at']
                        doc_dict = dict(zip(doc_columns, row))
                        # Create a chunk-like dict from document metadata
                        chunk_dict = {
                            'chunk_id': f"{doc_dict['document_id']}_doc",
                            'document_id': doc_dict['document_id'],
                            'content': f"Document: {doc_dict.get('title', doc_dict.get('filename', 'Unknown'))}\n{doc_dict.get('filename', '')}",
                            'chunk_index': 0,
                            'token_count': 10,  # Estimate
                            'start_position': 0,
                            'end_position': len(doc_dict.get('filename', '')),
                            'filename': doc_dict.get('filename', ''),
                            'title': doc_dict.get('title', ''),
                            'file_size': doc_dict.get('file_size', 0),
                            'mime_type': doc_dict.get('mime_type', ''),
                            'created_at': doc_dict.get('created_at', '')
                        }
                        chunks.append(chunk_dict)

                    if doc_results:
                        logger.info(f"Document metadata search found {len(doc_results)} documents")

            # If we have filters but no documents found, return documents from filters anyway
            # This handles cases where we want to return a specific document even if search terms don't match
            if not chunks and filters and 'document_ids' in filters and filters['document_ids']:
                logger.info("No search matches but document filters provided, returning filtered documents")
                doc_ids = filters['document_ids']

                doc_sql = f"""
                    SELECT document_id, filename, title, file_size, mime_type, created_at
                    FROM documents
                    WHERE document_id IN ({', '.join(['?' for _ in doc_ids])})
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                params = doc_ids + [limit]

                cursor.execute(doc_sql, params)
                fallback_results = cursor.fetchall()

                # Convert fallback results to chunk-like results
                for row in fallback_results:
                    doc_columns = ['document_id', 'filename', 'title', 'file_size', 'mime_type', 'created_at']
                    doc_dict = dict(zip(doc_columns, row))
                    # Create a chunk-like dict from document metadata
                    chunk_dict = {
                        'chunk_id': f"{doc_dict['document_id']}_doc",
                        'document_id': doc_dict['document_id'],
                        'content': f"Document: {doc_dict.get('title', doc_dict.get('filename', 'Unknown'))}\n{doc_dict.get('filename', '')}",
                        'chunk_index': 0,
                        'token_count': 10,  # Estimate
                        'start_position': 0,
                        'end_position': len(doc_dict.get('filename', '')),
                        'filename': doc_dict.get('filename', ''),
                        'title': doc_dict.get('title', ''),
                        'file_size': doc_dict.get('file_size', 0),
                        'mime_type': doc_dict.get('mime_type', ''),
                        'created_at': doc_dict.get('created_at', '')
                    }
                    chunks.append(chunk_dict)

                logger.info(f"Fallback returned {len(fallback_results)} documents from filters")

            logger.info(f"Keyword search found {len(chunks)} chunks for query: {query}")
            return chunks

        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document

        Args:
            document_id: Document identifier

        Returns:
            List of chunk dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT chunk_id, document_id, content, chunk_index,
                           token_count, start_position, end_position, metadata, summary
                    FROM chunks
                    WHERE document_id = ?
                    ORDER BY chunk_index
                ''', (document_id,))

                rows = cursor.fetchall()
                chunks = []

                for row in rows:
                    chunk_data = {
                        'chunk_id': row[0],
                        'document_id': row[1],
                        'content': row[2],
                        'chunk_index': row[3],
                        'token_count': row[4],
                        'start_position': row[5],
                        'end_position': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {},
                        'summary': row[8]
                    }
                    chunks.append(chunk_data)

                logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
                return chunks

        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get document statistics

        Returns:
            Dictionary with document statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total documents
                cursor.execute('SELECT COUNT(*) FROM documents')
                total_documents = cursor.fetchone()[0]

                # Documents by status
                cursor.execute('SELECT status, COUNT(*) FROM documents GROUP BY status')
                status_counts = dict(cursor.fetchall())

                # Documents by file type
                cursor.execute('SELECT file_extension, COUNT(*) FROM documents GROUP BY file_extension')
                file_type_counts = dict(cursor.fetchall())

                # Total summaries
                cursor.execute('SELECT COUNT(*) FROM document_summaries')
                total_summaries = cursor.fetchone()[0]

                # Total chunk summaries
                cursor.execute('SELECT COUNT(*) FROM chunk_summaries')
                total_chunk_summaries = cursor.fetchone()[0]

                return {
                    'total_documents': total_documents,
                    'status_counts': status_counts,
                    'file_type_counts': file_type_counts,
                    'total_summaries': total_summaries,
                    'total_chunk_summaries': total_chunk_summaries,
                    'database_path': self.db_path
                }

        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document and all related data
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete in order to respect foreign key constraints
                cursor.execute('DELETE FROM citation_metadata WHERE document_id = ?', (document_id,))
                cursor.execute('DELETE FROM chunk_summaries WHERE document_id = ?', (document_id,))
                cursor.execute('DELETE FROM document_summaries WHERE document_id = ?', (document_id,))
                cursor.execute('DELETE FROM documents WHERE document_id = ?', (document_id,))
                
                conn.commit()
                logger.info(f"Deleted document {document_id} and all related data")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def update_document_status(self, document_id: str, status: str) -> bool:
        """
        Update document processing status
        
        Args:
            document_id: Document identifier
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE documents 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?
                ''', (status, document_id))
                
                conn.commit()
                logger.info(f"Updated document {document_id} status to {status}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            return False


# Global SQLite storage instance
sqlite_storage = SQLiteStorage()


# Standalone testing function
def test_sqlite_storage():
    """Test the SQLite storage with sample data"""
    try:
        from src.core.document_models import DocumentMetadata, DocumentSummary, ChunkSummary
        
        # Create test storage
        test_db_path = "/tmp/test_bookmate.db"
        storage = SQLiteStorage(test_db_path)
        
        # Test document metadata
        doc_metadata = DocumentMetadata(
            filename="test_document.pdf",
            file_path="/path/to/test_document.pdf",
            file_size=1024,
            file_extension=".pdf",
            mime_type="application/pdf",
            title="Test Document",
            author="Test Author",
            page_count=5
        )
        
        document_id = "test_doc_123"
        success = storage.store_document_metadata(document_id, doc_metadata, "user123")
        print(f"Document metadata stored: {success}")
        
        # Test document summary
        doc_summary = DocumentSummary(
            summary_id="summary_123",
            document_id=document_id,
            summary_text="This is a test document summary.",
            chunk_count=10,
            word_count=20,
            processing_time=1.5
        )
        
        success = storage.store_document_summary(doc_summary)
        print(f"Document summary stored: {success}")
        
        # Test chunk summaries
        chunk_summaries = [
            ChunkSummary(
                chunk_summary_id="chunk_sum_1",
                document_id=document_id,
                chunk_id="chunk_1",
                chunk_index=0,
                summary_text="First chunk summary",
                word_count=5
            ),
            ChunkSummary(
                chunk_summary_id="chunk_sum_2",
                document_id=document_id,
                chunk_id="chunk_2",
                chunk_index=1,
                summary_text="Second chunk summary",
                word_count=5
            )
        ]
        
        success = storage.store_chunk_summaries(chunk_summaries)
        print(f"Chunk summaries stored: {success}")
        
        # Test retrieval
        retrieved_metadata = storage.get_document_metadata(document_id)
        print(f"Retrieved metadata: {retrieved_metadata is not None}")
        
        retrieved_summary = storage.get_document_summary(document_id)
        print(f"Retrieved summary: {retrieved_summary is not None}")
        
        retrieved_chunk_summaries = storage.get_chunk_summaries(document_id)
        print(f"Retrieved chunk summaries: {len(retrieved_chunk_summaries)}")
        
        # Test stats
        stats = storage.get_document_stats()
        print(f"Document stats: {stats}")
        
        # Cleanup
        os.remove(test_db_path)
        print("Test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


# Global instance
metadata_store = SQLiteStorage()


if __name__ == "__main__":
    # Run test
    test_sqlite_storage()
