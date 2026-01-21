"""
Metadata Enricher for BookMate RAG Agent
Adds citation-ready metadata to document chunks for source attribution
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_models import DocumentChunk, DocumentMetadata
from src.utils.llm_client import llm_client
import json

logger = logging.getLogger(__name__)


@dataclass
class CitationMetadata:
    """Citation metadata for source attribution"""
    
    page_number: Optional[int] = None
    heading: Optional[str] = None
    paragraph_index: Optional[int] = None
    section_title: Optional[str] = None
    citation_anchor: str = ""  # Primary citation reference
    citation_type: str = "page"  # "page", "heading", "paragraph", "chunk"
    source_location: str = ""  # Human-readable location


class MetadataEnricher:
    """Enriches document chunks with citation-ready metadata"""
    
    def __init__(self):
        """Initialize metadata enricher"""
        self.citation_rules = {
            'pdf': 'page',
            'docx': 'heading',
            'html': 'heading', 
            'txt': 'paragraph',
            'md': 'heading'
        }
        logger.info("MetadataEnricher initialized")
    
    def enrich_chunks(
        self, 
        chunks: List[DocumentChunk], 
        document_metadata: DocumentMetadata,
        structural_info: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Enrich chunks with citation metadata
        
        Args:
            chunks: List of document chunks
            document_metadata: Document metadata
            structural_info: Structural information (pages, headings, etc.)
            
        Returns:
            List of enriched chunks with citation metadata
        """
        try:
            logger.info(f"Enriching {len(chunks)} chunks with citation metadata")
            
            # Determine citation strategy based on file type
            file_extension = document_metadata.file_extension.lower()
            citation_strategy = self.citation_rules.get(file_extension, 'chunk')
            
            enriched_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Create citation metadata
                    citation_meta = self._create_citation_metadata(
                        chunk, 
                        document_metadata,
                        structural_info,
                        citation_strategy,
                        i
                    )
                    
                    # Enrich chunk with citation metadata
                    enriched_chunk = self._enrich_single_chunk(
                        chunk, 
                        citation_meta, 
                        document_metadata
                    )
                    
                    enriched_chunks.append(enriched_chunk)
                    
                except Exception as e:
                    logger.error(f"Error enriching chunk {i}: {str(e)}")
                    # Add fallback metadata
                    fallback_meta = CitationMetadata(
                        citation_anchor=f"chunk_{i}",
                        citation_type="chunk",
                        source_location=f"Chunk #{i+1}"
                    )
                    enriched_chunk = self._enrich_single_chunk(
                        chunk, 
                        fallback_meta, 
                        document_metadata
                    )
                    enriched_chunks.append(enriched_chunk)
            
            logger.info(f"Successfully enriched {len(enriched_chunks)} chunks")
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"Error in metadata enrichment: {str(e)}")
            raise
    
    def _create_citation_metadata(
        self,
        chunk: DocumentChunk,
        document_metadata: DocumentMetadata,
        structural_info: Optional[Dict[str, Any]],
        citation_strategy: str,
        chunk_index: int
    ) -> CitationMetadata:
        """Create citation metadata for a chunk"""
        
        citation_meta = CitationMetadata()
        
        try:
            # Extract structural information from chunk metadata if available
            chunk_meta = chunk.metadata.__dict__ if chunk.metadata else {}
            
            # Set page number (primary citation anchor for PDFs)
            if 'page_number' in chunk_meta:
                citation_meta.page_number = int(chunk_meta['page_number'])
                citation_meta.citation_anchor = f"p.{citation_meta.page_number}"
                citation_meta.citation_type = "page"
                citation_meta.source_location = f"Page {citation_meta.page_number}"
            
            # Set heading information
            elif 'heading' in chunk_meta and chunk_meta['heading']:
                citation_meta.heading = chunk_meta['heading']
                citation_meta.citation_anchor = f"'{citation_meta.heading}'"
                citation_meta.citation_type = "heading"
                citation_meta.source_location = f"Section '{citation_meta.heading}'"
            
            # Set paragraph information
            elif 'paragraph_index' in chunk_meta:
                citation_meta.paragraph_index = int(chunk_meta['paragraph_index'])
                citation_meta.citation_anchor = f"para.{citation_meta.paragraph_index}"
                citation_meta.citation_type = "paragraph"
                citation_meta.source_location = f"Paragraph {citation_meta.paragraph_index + 1}"
            
            # Fallback to chunk-based citation
            else:
                citation_meta.citation_anchor = f"chunk_{chunk_index}"
                citation_meta.citation_type = "chunk"
                citation_meta.source_location = f"Chunk #{chunk_index + 1}"
            
            # Add section title if available
            if 'section_title' in chunk_meta:
                citation_meta.section_title = chunk_meta['section_title']
            
            # Enhance source location with document name
            doc_name = document_metadata.filename or "Document"
            citation_meta.source_location = f"{doc_name} ({citation_meta.source_location})"
            
            return citation_meta
            
        except Exception as e:
            logger.warning(f"Error creating citation metadata: {str(e)}")
            # Return fallback metadata
            return CitationMetadata(
                citation_anchor=f"chunk_{chunk_index}",
                citation_type="chunk",
                source_location=f"Chunk #{chunk_index + 1}"
            )
    
    def _enrich_single_chunk(
        self,
        chunk: DocumentChunk,
        citation_meta: CitationMetadata,
        document_metadata: DocumentMetadata
    ) -> DocumentChunk:
        """Enrich a single chunk with citation metadata"""
        
        try:
            # Create enriched metadata
            enriched_metadata = DocumentMetadata(
                filename=document_metadata.filename,
                file_path=document_metadata.file_path,
                file_size=document_metadata.file_size,
                file_extension=document_metadata.file_extension,
                mime_type=document_metadata.mime_type,
                title=document_metadata.title,
                author=document_metadata.author,
                creation_date=document_metadata.creation_date,
                modification_date=document_metadata.modification_date,
                page_count=document_metadata.page_count,
                language=document_metadata.language,
                source_url=document_metadata.source_url,
                tags=document_metadata.tags.copy(),
                custom_metadata=document_metadata.custom_metadata.copy()
            )
            
            # Add citation-specific metadata
            enriched_metadata.custom_metadata.update({
                'citation_anchor': citation_meta.citation_anchor,
                'citation_type': citation_meta.citation_type,
                'source_location': citation_meta.source_location,
                'page_number': citation_meta.page_number,
                'heading': citation_meta.heading,
                'paragraph_index': citation_meta.paragraph_index,
                'section_title': citation_meta.section_title,
                'enriched_at': datetime.now().isoformat(),
                'enricher_version': '1.0.0'
            })
            
            # Create enriched chunk
            enriched_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                start_position=chunk.start_position,
                end_position=chunk.end_position,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
                metadata=enriched_metadata,
                summary=chunk.summary,
                keywords=chunk.keywords.copy(),
                created_at=chunk.created_at
            )
            
            return enriched_chunk
            
        except Exception as e:
            logger.error(f"Error enriching single chunk: {str(e)}")
            # Return original chunk with minimal enrichment
            if chunk.metadata:
                chunk.metadata.custom_metadata.update({
                    'citation_anchor': citation_meta.citation_anchor,
                    'citation_type': citation_meta.citation_type,
                    'source_location': citation_meta.source_location,
                    'enriched_at': datetime.now().isoformat()
                })
            return chunk
    
    def get_citation_format(
        self, 
        chunk: DocumentChunk, 
        format_type: str = "default"
    ) -> str:
        """
        Generate citation string for a chunk
        
        Args:
            chunk: Document chunk with citation metadata
            format_type: Citation format type ("default", "academic", "simple")
            
        Returns:
            Formatted citation string
        """
        try:
            if not chunk.metadata or not chunk.metadata.custom_metadata:
                return f"({chunk.document_id}, chunk {chunk.chunk_index})"
            
            custom_meta = chunk.metadata.custom_metadata
            doc_name = chunk.metadata.filename or chunk.document_id
            citation_anchor = custom_meta.get('citation_anchor', f"chunk_{chunk.chunk_index}")
            citation_type = custom_meta.get('citation_type', 'chunk')
            
            if format_type == "academic":
                if citation_type == "page":
                    return f"({doc_name}, p. {custom_meta.get('page_number', '?')})"
                elif citation_type == "heading":
                    return f"({doc_name}, {citation_anchor})"
                else:
                    return f"({doc_name}, {citation_anchor})"
            
            elif format_type == "simple":
                return f"({doc_name}, {citation_anchor})"
            
            else:  # default
                source_location = custom_meta.get('source_location', citation_anchor)
                return f"({source_location})"
                
        except Exception as e:
            logger.error(f"Error generating citation format: {str(e)}")
            return f"({chunk.document_id}, chunk {chunk.chunk_index})"
    
    def validate_citation_metadata(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate citation metadata across chunks
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'total_chunks': len(chunks),
                'valid_citations': 0,
                'missing_citations': 0,
                'citation_types': {},
                'errors': []
            }
            
            for chunk in chunks:
                try:
                    if (chunk.metadata and 
                        chunk.metadata.custom_metadata and 
                        'citation_anchor' in chunk.metadata.custom_metadata):
                        
                        validation_results['valid_citations'] += 1
                        
                        citation_type = chunk.metadata.custom_metadata.get('citation_type', 'unknown')
                        validation_results['citation_types'][citation_type] = (
                            validation_results['citation_types'].get(citation_type, 0) + 1
                        )
                    else:
                        validation_results['missing_citations'] += 1
                        validation_results['errors'].append(
                            f"Chunk {chunk.chunk_id} missing citation metadata"
                        )
                        
                except Exception as e:
                    validation_results['errors'].append(
                        f"Error validating chunk {chunk.chunk_id}: {str(e)}"
                    )
            
            validation_results['success_rate'] = (
                validation_results['valid_citations'] / validation_results['total_chunks'] * 100
                if validation_results['total_chunks'] > 0 else 0
            )
            
            logger.info(f"Citation validation completed: {validation_results['success_rate']:.1f}% success rate")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in citation validation: {str(e)}")
            return {
                'total_chunks': len(chunks),
                'valid_citations': 0,
                'missing_citations': len(chunks),
                'errors': [f"Validation failed: {str(e)}"]
            }

    def enrich_document_with_llm(self, text_content: str, metadata: DocumentMetadata) -> DocumentMetadata:
        """
        Enrich document metadata using LLM extraction
        
        Args:
            text_content: Document text content (will be truncated)
            metadata: Current document metadata
            
        Returns:
            Enriched DocumentMetadata
        """
        try:
            # Truncate text for LLM (first 3000 chars is usually enough for metadata)
            context_text = text_content[:3000]
            
            prompt = f"""
            Analyze the following document text and extract rich metadata in JSON format.
            
            TEXT START:
            {context_text}
            ...
            TEXT END
            
            Return ONLY a valid JSON object with the following keys:
            - title: Improved title if applicable, else original
            - author: Extracted author(s)
            - summary: Brief 2-sentence summary
            - keywords: List of 5-10 key topics/concepts
            - entities: List of important named entities (people, organizations, products)
            - language: ISO language code (e.g., 'en', 'es')
            - category: Technical Category (e.g., 'Finance', 'Engineering', 'Medical', 'General')
            
            Do not include any explanation, only the JSON.
            """
            
            response = llm_client.generate_text(
                prompt=prompt,
                max_tokens=600,
                temperature=0.1,
                system_prompt="You are a librarian AI specialized in metadata extraction.",
                expect_json=True
            )
            
            if response:
                # Clean up response if it contains markdown code blocks
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                    
                try:
                    rich_data = json.loads(response)
                    
                    # Update metadata
                    if rich_data.get('title') and not metadata.title:
                        metadata.title = rich_data['title']
                    
                    if rich_data.get('author') and not metadata.author:
                        metadata.author = rich_data['author']
                    
                    if rich_data.get('language') and not metadata.language:
                        metadata.language = rich_data['language']
                        
                    # Store rich metadata in custom_metadata for SQLite storage
                    metadata.custom_metadata['rich_metadata'] = rich_data
                    metadata.custom_metadata['summary'] = rich_data.get('summary')
                    metadata.custom_metadata['keywords'] = rich_data.get('keywords')
                    
                    # Update tags with keywords
                    if rich_data.get('keywords'):
                        current_tags = set(metadata.tags)
                        current_tags.update(rich_data['keywords'])
                        metadata.tags = list(current_tags)
                        
                    logger.info(f"Successfully extracted rich metadata for {metadata.filename}")
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response for metadata extraction")
            
            return metadata

        except Exception as e:
            logger.error(f"Error in LLM metadata enrichment: {str(e)}")
            return metadata


# Standalone testing function
def test_metadata_enricher():
    """Test the metadata enricher with sample data"""
    try:
        from src.core.document_models import DocumentMetadata, DocumentChunk
        
        # Create test metadata
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
        
        # Create test chunks
        test_chunks = [
            DocumentChunk(
                chunk_id="test_chunk_1",
                document_id="test_doc",
                content="This is the first chunk of the document.",
                chunk_index=0,
                start_position=0,
                end_position=50,
                token_count=10,
                metadata=DocumentMetadata(
                    filename="test_document.pdf",
                    file_path="/path/to/test_document.pdf",
                    file_size=1024,
                    file_extension=".pdf",
                    mime_type="application/pdf",
                    custom_metadata={'page_number': 1}
                )
            ),
            DocumentChunk(
                chunk_id="test_chunk_2",
                document_id="test_doc",
                content="This is the second chunk of the document.",
                chunk_index=1,
                start_position=50,
                end_position=100,
                token_count=10,
                metadata=DocumentMetadata(
                    filename="test_document.pdf",
                    file_path="/path/to/test_document.pdf",
                    file_size=1024,
                    file_extension=".pdf",
                    mime_type="application/pdf",
                    custom_metadata={'page_number': 2}
                )
            )
        ]
        
        # Test enricher
        enricher = MetadataEnricher()
        enriched_chunks = enricher.enrich_chunks(test_chunks, doc_metadata)
        
        print("Metadata Enricher Test Results:")
        print(f"Original chunks: {len(test_chunks)}")
        print(f"Enriched chunks: {len(enriched_chunks)}")
        
        for chunk in enriched_chunks:
            if chunk.metadata and chunk.metadata.custom_metadata:
                citation_anchor = chunk.metadata.custom_metadata.get('citation_anchor', 'N/A')
                citation_type = chunk.metadata.custom_metadata.get('citation_type', 'N/A')
                print(f"Chunk {chunk.chunk_index}: {citation_anchor} ({citation_type})")
        
        # Test validation
        validation_results = enricher.validate_citation_metadata(enriched_chunks)
        print(f"Validation success rate: {validation_results['success_rate']:.1f}%")
        
        return enriched_chunks
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run test
    test_metadata_enricher()
