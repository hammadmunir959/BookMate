"""
Citation Manager for BookMate RAG Agent
Handles citation rules, formatting, and source attribution
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_models import DocumentChunk

logger = logging.getLogger(__name__)


class CitationStyle(Enum):
    """Citation style options"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    SIMPLE = "simple"
    ACADEMIC = "academic"


@dataclass
class CitationRule:
    """Citation rule for different document types"""
    
    file_extension: str
    primary_citation_type: str  # "page", "heading", "paragraph", "chunk"
    fallback_citation_type: str
    format_template: str
    description: str


class CitationManager:
    """Manages citation rules and formatting for different document types"""
    
    def __init__(self):
        """Initialize citation manager with default rules"""
        self.citation_rules = {
            '.pdf': CitationRule(
                file_extension='.pdf',
                primary_citation_type='page',
                fallback_citation_type='chunk',
                format_template='{document_name} (p. {page_number})',
                description='PDF documents cited by page number'
            ),
            '.docx': CitationRule(
                file_extension='.docx',
                primary_citation_type='heading',
                fallback_citation_type='paragraph',
                format_template='{document_name} ({section_title})',
                description='DOCX documents cited by section heading'
            ),
            '.html': CitationRule(
                file_extension='.html',
                primary_citation_type='heading',
                fallback_citation_type='chunk',
                format_template='{document_name} ({section_title})',
                description='HTML documents cited by heading'
            ),
            '.txt': CitationRule(
                file_extension='.txt',
                primary_citation_type='paragraph',
                fallback_citation_type='chunk',
                format_template='{document_name} (para. {paragraph_index})',
                description='Text documents cited by paragraph'
            ),
            '.md': CitationRule(
                file_extension='.md',
                primary_citation_type='heading',
                fallback_citation_type='paragraph',
                format_template='{document_name} ({section_title})',
                description='Markdown documents cited by heading'
            )
        }
        
        self.citation_styles = {
            CitationStyle.APA: {
                'format': '({author}, {year}, p. {page})',
                'description': 'APA style citation'
            },
            CitationStyle.MLA: {
                'format': '({author} {page})',
                'description': 'MLA style citation'
            },
            CitationStyle.CHICAGO: {
                'format': '({author} {year}, {page})',
                'description': 'Chicago style citation'
            },
            CitationStyle.HARVARD: {
                'format': '({author} {year}: {page})',
                'description': 'Harvard style citation'
            },
            CitationStyle.SIMPLE: {
                'format': '({document_name}, {location})',
                'description': 'Simple citation format'
            },
            CitationStyle.ACADEMIC: {
                'format': '({document_name}, {location})',
                'description': 'Academic citation format'
            }
        }
        
        logger.info("CitationManager initialized with citation rules")
    
    def get_citation_rule(self, file_extension: str) -> CitationRule:
        """
        Get citation rule for a file extension
        
        Args:
            file_extension: File extension (e.g., '.pdf', '.docx')
            
        Returns:
            Citation rule for the file type
        """
        extension = file_extension.lower()
        return self.citation_rules.get(extension, self.citation_rules['.txt'])
    
    def generate_citation(
        self, 
        chunk: DocumentChunk, 
        style: CitationStyle = CitationStyle.SIMPLE,
        include_author: bool = False
    ) -> str:
        """
        Generate citation for a document chunk
        
        Args:
            chunk: Document chunk with metadata
            style: Citation style to use
            include_author: Whether to include author information
            
        Returns:
            Formatted citation string
        """
        try:
            if not chunk.metadata or not chunk.metadata.custom_metadata:
                return f"({chunk.document_id}, chunk {chunk.chunk_index})"
            
            custom_meta = chunk.metadata.custom_metadata
            doc_name = chunk.metadata.filename or chunk.document_id
            file_extension = chunk.metadata.file_extension or '.txt'
            
            # Get citation rule for this file type
            rule = self.get_citation_rule(file_extension)
            
            # Determine citation type and location
            citation_type, location = self._determine_citation_location(chunk, rule)
            
            # Generate citation based on style
            if style == CitationStyle.SIMPLE:
                return self._generate_simple_citation(doc_name, location, citation_type)
            elif style == CitationStyle.ACADEMIC:
                return self._generate_academic_citation(chunk, doc_name, location, citation_type, include_author)
            else:
                return self._generate_style_citation(chunk, style, doc_name, location, citation_type, include_author)
                
        except Exception as e:
            logger.error(f"Error generating citation: {str(e)}")
            return f"({chunk.document_id}, chunk {chunk.chunk_index})"
    
    def _determine_citation_location(self, chunk: DocumentChunk, rule: CitationRule) -> Tuple[str, str]:
        """
        Determine the citation location for a chunk
        
        Args:
            chunk: Document chunk
            rule: Citation rule for the document type
            
        Returns:
            Tuple of (citation_type, location_string)
        """
        try:
            custom_meta = chunk.metadata.custom_metadata
            
            # Try primary citation type first
            if rule.primary_citation_type == 'page' and 'page_number' in custom_meta:
                return 'page', str(custom_meta['page_number'])
            
            elif rule.primary_citation_type == 'heading' and 'section_heading' in custom_meta:
                return 'heading', custom_meta['section_heading']
            
            elif rule.primary_citation_type == 'paragraph' and 'paragraph_index' in custom_meta:
                return 'paragraph', str(custom_meta['paragraph_index'] + 1)
            
            # Try fallback citation type
            if rule.fallback_citation_type == 'page' and 'page_number' in custom_meta:
                return 'page', str(custom_meta['page_number'])
            
            elif rule.fallback_citation_type == 'heading' and 'section_heading' in custom_meta:
                return 'heading', custom_meta['section_heading']
            
            elif rule.fallback_citation_type == 'paragraph' and 'paragraph_index' in custom_meta:
                return 'paragraph', str(custom_meta['paragraph_index'] + 1)
            
            # Final fallback to chunk
            return 'chunk', f"chunk {chunk.chunk_index + 1}"
            
        except Exception as e:
            logger.error(f"Error determining citation location: {str(e)}")
            return 'chunk', f"chunk {chunk.chunk_index + 1}"
    
    def _generate_simple_citation(self, doc_name: str, location: str, citation_type: str) -> str:
        """Generate simple citation format"""
        if citation_type == 'page':
            return f"({doc_name}, p. {location})"
        elif citation_type == 'heading':
            return f"({doc_name}, {location})"
        elif citation_type == 'paragraph':
            return f"({doc_name}, para. {location})"
        else:
            return f"({doc_name}, {location})"
    
    def _generate_academic_citation(
        self, 
        chunk: DocumentChunk, 
        doc_name: str, 
        location: str, 
        citation_type: str,
        include_author: bool
    ) -> str:
        """Generate academic citation format"""
        try:
            author = ""
            if include_author and chunk.metadata and chunk.metadata.author:
                author = f"{chunk.metadata.author}, "
            
            if citation_type == 'page':
                return f"({author}{doc_name}, p. {location})"
            elif citation_type == 'heading':
                return f"({author}{doc_name}, {location})"
            elif citation_type == 'paragraph':
                return f"({author}{doc_name}, para. {location})"
            else:
                return f"({author}{doc_name}, {location})"
                
        except Exception as e:
            logger.error(f"Error generating academic citation: {str(e)}")
            return f"({doc_name}, {location})"
    
    def _generate_style_citation(
        self, 
        chunk: DocumentChunk, 
        style: CitationStyle, 
        doc_name: str, 
        location: str, 
        citation_type: str,
        include_author: bool
    ) -> str:
        """Generate citation in specific academic style"""
        try:
            style_config = self.citation_styles.get(style, self.citation_styles[CitationStyle.SIMPLE])
            format_template = style_config['format']
            
            # Extract year from creation date if available
            year = ""
            if chunk.metadata and chunk.metadata.creation_date:
                year = str(chunk.metadata.creation_date.year)
            
            # Get author
            author = ""
            if include_author and chunk.metadata and chunk.metadata.author:
                author = chunk.metadata.author
            
            # Format page/location
            page = location
            if citation_type == 'page':
                page = f"p. {location}"
            elif citation_type == 'paragraph':
                page = f"para. {location}"
            
            # Replace placeholders
            citation = format_template.format(
                author=author,
                year=year,
                page=page,
                document_name=doc_name,
                location=location
            )
            
            return citation
            
        except Exception as e:
            logger.error(f"Error generating style citation: {str(e)}")
            return f"({doc_name}, {location})"
    
    def format_citation_in_text(
        self, 
        text: str, 
        chunks: List[DocumentChunk],
        style: CitationStyle = CitationStyle.SIMPLE
    ) -> str:
        """
        Format citations within text content
        
        Args:
            text: Text content that may contain citations
            chunks: List of chunks that were used to generate the text
            style: Citation style to use
            
        Returns:
            Text with formatted citations
        """
        try:
            if not chunks:
                return text
            
            # Find citation placeholders in text
            citation_pattern = r'\[citation:(\d+)\]'
            matches = re.findall(citation_pattern, text)
            
            for match in matches:
                try:
                    chunk_index = int(match)
                    if 0 <= chunk_index < len(chunks):
                        chunk = chunks[chunk_index]
                        citation = self.generate_citation(chunk, style)
                        text = text.replace(f'[citation:{match}]', citation)
                except (ValueError, IndexError):
                    continue
            
            return text
            
        except Exception as e:
            logger.error(f"Error formatting citations in text: {str(e)}")
            return text
    
    def get_citation_summary(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get citation summary for a list of chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Citation summary information
        """
        try:
            summary = {
                'total_chunks': len(chunks),
                'citation_types': {},
                'documents': {},
                'citation_coverage': 0
            }
            
            cited_chunks = 0
            
            for chunk in chunks:
                if chunk.metadata and chunk.metadata.custom_metadata:
                    custom_meta = chunk.metadata.custom_metadata
                    citation_type = custom_meta.get('citation_type', 'unknown')
                    
                    # Count citation types
                    summary['citation_types'][citation_type] = (
                        summary['citation_types'].get(citation_type, 0) + 1
                    )
                    
                    # Count documents
                    doc_name = chunk.metadata.filename or chunk.document_id
                    if doc_name not in summary['documents']:
                        summary['documents'][doc_name] = {
                            'chunk_count': 0,
                            'citation_types': set()
                        }
                    
                    summary['documents'][doc_name]['chunk_count'] += 1
                    summary['documents'][doc_name]['citation_types'].add(citation_type)
                    
                    cited_chunks += 1
            
            # Calculate citation coverage
            if summary['total_chunks'] > 0:
                summary['citation_coverage'] = (cited_chunks / summary['total_chunks']) * 100
            
            # Convert sets to lists for JSON serialization
            for doc_name in summary['documents']:
                summary['documents'][doc_name]['citation_types'] = list(
                    summary['documents'][doc_name]['citation_types']
                )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating citation summary: {str(e)}")
            return {
                'total_chunks': len(chunks),
                'citation_types': {},
                'documents': {},
                'citation_coverage': 0,
                'error': str(e)
            }
    
    def validate_citations(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate citations across chunks
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'total_chunks': len(chunks),
                'valid_citations': 0,
                'invalid_citations': 0,
                'missing_citations': 0,
                'errors': [],
                'warnings': []
            }
            
            for i, chunk in enumerate(chunks):
                try:
                    if not chunk.metadata or not chunk.metadata.custom_metadata:
                        validation_results['missing_citations'] += 1
                        validation_results['errors'].append(f"Chunk {i} missing metadata")
                        continue
                    
                    custom_meta = chunk.metadata.custom_metadata
                    
                    # Check for required citation fields
                    required_fields = ['citation_anchor', 'citation_type', 'source_location']
                    missing_fields = [field for field in required_fields if field not in custom_meta]
                    
                    if missing_fields:
                        validation_results['invalid_citations'] += 1
                        validation_results['errors'].append(
                            f"Chunk {i} missing citation fields: {missing_fields}"
                        )
                    else:
                        validation_results['valid_citations'] += 1
                        
                        # Check for potential issues
                        citation_type = custom_meta.get('citation_type', '')
                        if citation_type == 'page' and 'page_number' not in custom_meta:
                            validation_results['warnings'].append(
                                f"Chunk {i} marked as page citation but missing page_number"
                            )
                        elif citation_type == 'heading' and 'section_heading' not in custom_meta:
                            validation_results['warnings'].append(
                                f"Chunk {i} marked as heading citation but missing section_heading"
                            )
                
                except Exception as e:
                    validation_results['errors'].append(f"Error validating chunk {i}: {str(e)}")
            
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
                'invalid_citations': len(chunks),
                'missing_citations': 0,
                'errors': [f"Validation failed: {str(e)}"]
            }


# Standalone testing function
def test_citation_manager():
    """Test the citation manager with sample data"""
    try:
        from src.core.document_models import DocumentMetadata, DocumentChunk
        
        # Create test chunks with different citation types
        test_chunks = [
            DocumentChunk(
                chunk_id="test_chunk_1",
                document_id="test_doc",
                content="This is the first chunk of a PDF document.",
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
                    author="Test Author",
                    custom_metadata={
                        'citation_anchor': 'p.1',
                        'citation_type': 'page',
                        'source_location': 'test_document.pdf (Page 1)',
                        'page_number': 1
                    }
                )
            ),
            DocumentChunk(
                chunk_id="test_chunk_2",
                document_id="test_doc",
                content="This is the second chunk of a DOCX document.",
                chunk_index=1,
                start_position=50,
                end_position=100,
                token_count=10,
                metadata=DocumentMetadata(
                    filename="test_document.docx",
                    file_path="/path/to/test_document.docx",
                    file_size=2048,
                    file_extension=".docx",
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    author="Test Author",
                    custom_metadata={
                        'citation_anchor': 'Methodology',
                        'citation_type': 'heading',
                        'source_location': 'test_document.docx (Section Methodology)',
                        'section_heading': 'Methodology'
                    }
                )
            )
        ]
        
        # Test citation manager
        citation_manager = CitationManager()
        
        print("Citation Manager Test Results:")
        
        # Test citation generation
        for i, chunk in enumerate(test_chunks):
            citation = citation_manager.generate_citation(chunk, CitationStyle.SIMPLE)
            print(f"Chunk {i} citation: {citation}")
        
        # Test academic citation
        academic_citation = citation_manager.generate_citation(test_chunks[0], CitationStyle.ACADEMIC, include_author=True)
        print(f"Academic citation: {academic_citation}")
        
        # Test citation summary
        summary = citation_manager.get_citation_summary(test_chunks)
        print(f"Citation summary: {summary}")
        
        # Test validation
        validation_results = citation_manager.validate_citations(test_chunks)
        print(f"Validation success rate: {validation_results['success_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test
    test_citation_manager()

