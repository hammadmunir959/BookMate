"""
Enhanced Document Chunker for BookMate RAG Agent
Splits text into chunks while respecting structural boundaries for citation
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_models import DocumentChunk, DocumentMetadata
from src.processors.ingestion.text_extractor import StructuralUnit, ExtractedContent
from src.core.config import config

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class ChunkingConfig:
    """Configuration for enhanced document chunking"""

    chunk_size: int = None  # Target chunk size in tokens
    chunk_overlap: int = None  # Overlap between chunks in tokens
    min_chunk_size: int = None
    max_chunk_size: int = None
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    preserve_pages: bool = True
    preserve_sections: bool = True
    use_smart_splitting: bool = True
    respect_structural_boundaries: bool = True

    def __post_init__(self):
        if self.chunk_size is None:
            self.chunk_size = config.ingestion.chunk_size
        if self.chunk_overlap is None:
            self.chunk_overlap = config.ingestion.chunk_overlap
        if self.min_chunk_size is None:
            self.min_chunk_size = config.ingestion.min_chunk_size
        if self.max_chunk_size is None:
            self.max_chunk_size = config.ingestion.max_chunk_size


class EnhancedDocumentChunker:
    """Enhanced document chunker that respects structural boundaries"""

    def __init__(self, config: ChunkingConfig = None):
        """Initialize enhanced chunker with configuration"""
        self.config = config or ChunkingConfig()

        # Initialize token counter (simple word-based approximation)
        self.tokenizer = SimpleTokenizer()

        logger.info(f"EnhancedDocumentChunker initialized with chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")

    def chunk_document_with_structure(
        self,
        extracted_content: ExtractedContent,
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Chunk a document with structural awareness

        Args:
            extracted_content: Extracted content with structural information
            document_id: Unique document identifier

        Returns:
            List of DocumentChunk objects with structural metadata
        """
        try:
            logger.info(f"Chunking document {document_id} with {len(extracted_content.structural_units)} structural units")

            chunks = []
            chunk_index = 0
            current_position = 0

            # Process each structural unit
            for unit in extracted_content.structural_units:
                if not unit.text.strip():
                    continue

                # Chunk this structural unit
                unit_chunks = self._chunk_structural_unit(
                    unit,
                    extracted_content.metadata,
                    document_id,
                    chunk_index,
                    current_position
                )

                chunks.extend(unit_chunks)
                chunk_index += len(unit_chunks)
                current_position += len(unit.text)

            logger.info(f"Created {len(chunks)} chunks from document {document_id}")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking document with structure: {str(e)}")
            raise

    def _chunk_structural_unit(
        self,
        unit: StructuralUnit,
        document_metadata: DocumentMetadata,
        document_id: str,
        start_chunk_index: int,
        start_position: int
    ) -> List[DocumentChunk]:
        """Chunk a single structural unit"""
        try:
            chunks = []

            # Check if unit is small enough to be a single chunk
            tokens = self.tokenizer.tokenize(unit.text)
            total_tokens = len(tokens)

            if total_tokens <= self.config.max_chunk_size:
                # Create single chunk for this unit
                chunk = self._create_chunk_from_unit(
                    unit,
                    document_metadata,
                    document_id,
                    start_chunk_index,
                    start_position,
                    unit.text,
                    total_tokens
                )
                chunks.append(chunk)
                return chunks

            # Need to split into multiple chunks
            if self.config.respect_structural_boundaries:
                # Try to split by sentences first
                chunks = self._chunk_by_sentences(
                    unit, document_metadata, document_id, 
                    start_chunk_index, start_position
                )
            else:
                # Use traditional chunking
                chunks = self._chunk_by_tokens(
                    unit, document_metadata, document_id,
                    start_chunk_index, start_position
                )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking structural unit: {str(e)}")
            # Fallback: create single chunk
            return [self._create_chunk_from_unit(
                unit, document_metadata, document_id,
                start_chunk_index, start_position,
                unit.text, len(unit.text.split())
            )]

    def _chunk_by_sentences(
        self,
        unit: StructuralUnit,
        document_metadata: DocumentMetadata,
        document_id: str,
        start_chunk_index: int,
        start_position: int
    ) -> List[DocumentChunk]:
        """Chunk by sentences while respecting boundaries"""
        try:
            chunks = []
            sentences = sent_tokenize(unit.text)
            
            current_chunk_text = ""
            current_chunk_tokens = 0
            chunk_index = start_chunk_index
            current_position = start_position
            
            for sentence in sentences:
                sentence_tokens = self.tokenizer.tokenize(sentence)
                
                # Check if adding this sentence would exceed chunk size
                if (current_chunk_tokens + len(sentence_tokens) > self.config.chunk_size and 
                    current_chunk_text.strip()):
                    
                    # Create chunk from current text
                    chunk = self._create_chunk_from_unit(
                        unit, document_metadata, document_id,
                        chunk_index, current_position,
                        current_chunk_text.strip(), current_chunk_tokens
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk_text)
                    current_chunk_text = overlap_text + " " + sentence
                    current_chunk_tokens = len(self.tokenizer.tokenize(current_chunk_text))
                    chunk_index += 1
                    current_position += len(current_chunk_text)
                else:
                    # Add sentence to current chunk
                    if current_chunk_text:
                        current_chunk_text += " " + sentence
                    else:
                        current_chunk_text = sentence
                    current_chunk_tokens += len(sentence_tokens)
            
            # Add final chunk if there's remaining text
            if current_chunk_text.strip():
                chunk = self._create_chunk_from_unit(
                    unit, document_metadata, document_id,
                    chunk_index, current_position,
                    current_chunk_text.strip(), current_chunk_tokens
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking by sentences: {str(e)}")
            # Fallback to token-based chunking
            return self._chunk_by_tokens(unit, document_metadata, document_id, start_chunk_index, start_position)

    def _chunk_by_tokens(
        self,
        unit: StructuralUnit,
        document_metadata: DocumentMetadata,
        document_id: str,
        start_chunk_index: int,
        start_position: int
    ) -> List[DocumentChunk]:
        """Chunk by tokens with overlap"""
        try:
            chunks = []
            tokens = self.tokenizer.tokenize(unit.text)
            total_tokens = len(tokens)
            
            current_position = 0
            chunk_index = start_chunk_index

            while current_position < total_tokens:
                # Calculate chunk boundaries
                chunk_start = max(0, current_position - self.config.chunk_overlap)
                chunk_end = min(total_tokens, chunk_start + self.config.chunk_size)

                # Ensure minimum chunk size
                if chunk_end - chunk_start < self.config.min_chunk_size and chunk_end < total_tokens:
                    chunk_end = min(total_tokens, chunk_start + self.config.min_chunk_size)

                # Extract chunk text
                chunk_tokens = tokens[chunk_start:chunk_end]
                chunk_text = self.tokenizer.detokenize(chunk_tokens)

                # Create chunk
                chunk = self._create_chunk_from_unit(
                    unit, document_metadata, document_id,
                    chunk_index, start_position + chunk_start,
                    chunk_text, len(chunk_tokens)
                )
                chunks.append(chunk)

                # Move to next chunk
                current_position = chunk_end
                chunk_index += 1

                # Prevent infinite loops
                if chunk_end <= chunk_start:
                    break

            return chunks

        except Exception as e:
            logger.error(f"Error chunking by tokens: {str(e)}")
            raise

    def _create_chunk_from_unit(
        self,
        unit: StructuralUnit,
        document_metadata: DocumentMetadata,
        document_id: str,
        chunk_index: int,
        start_position: int,
        chunk_text: str,
        token_count: int
    ) -> DocumentChunk:
        """Create a DocumentChunk from a structural unit"""
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
            
            # Add structural metadata
            enriched_metadata.custom_metadata.update({
                'unit_type': unit.unit_type,
                'unit_index': unit.unit_index,
                'heading': unit.heading,
                'level': unit.level,
                'chunked_at': logger.name,  # Will be replaced with actual timestamp
                'chunker_version': '1.0.0'
            })
            
            # Add unit-specific metadata
            enriched_metadata.custom_metadata.update(unit.metadata)
            
            # Set page number for PDFs
            if unit.unit_type == "page":
                enriched_metadata.custom_metadata['page_number'] = unit.unit_index
            
            # Set paragraph index for text files
            elif unit.unit_type == "paragraph":
                enriched_metadata.custom_metadata['paragraph_index'] = unit.unit_index
            
            # Set section information for structured documents
            elif unit.unit_type == "section":
                enriched_metadata.custom_metadata['section_index'] = unit.unit_index
                enriched_metadata.custom_metadata['section_heading'] = unit.heading
                enriched_metadata.custom_metadata['section_level'] = unit.level
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                content=chunk_text,
                chunk_index=chunk_index,
                start_position=start_position,
                end_position=start_position + len(chunk_text),
                token_count=token_count,
                metadata=enriched_metadata
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating chunk from unit: {str(e)}")
            raise

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of the current chunk"""
        try:
            words = text.split()
            if len(words) <= self.config.chunk_overlap:
                return text
            
            # Get last few words for overlap
            overlap_words = words[-self.config.chunk_overlap:]
            return " ".join(overlap_words)
            
        except Exception as e:
            logger.error(f"Error getting overlap text: {str(e)}")
            return ""

    def validate_chunk_boundaries(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate that chunks respect structural boundaries
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'total_chunks': len(chunks),
                'boundary_violations': 0,
                'page_boundary_violations': 0,
                'section_boundary_violations': 0,
                'paragraph_boundary_violations': 0,
                'errors': []
            }
            
            for i, chunk in enumerate(chunks):
                try:
                    if not chunk.metadata or not chunk.metadata.custom_metadata:
                        validation_results['errors'].append(f"Chunk {i} missing metadata")
                        continue
                    
                    custom_meta = chunk.metadata.custom_metadata
                    unit_type = custom_meta.get('unit_type', 'unknown')
                    
                    # Check for boundary violations
                    if i > 0:
                        prev_chunk = chunks[i-1]
                        if (prev_chunk.metadata and 
                            prev_chunk.metadata.custom_metadata):
                            
                            prev_unit_type = prev_chunk.metadata.custom_metadata.get('unit_type', 'unknown')
                            
                            # Check page boundary violations
                            if (unit_type == "page" and prev_unit_type == "page" and
                                custom_meta.get('page_number') != prev_chunk.metadata.custom_metadata.get('page_number')):
                                validation_results['page_boundary_violations'] += 1
                                validation_results['boundary_violations'] += 1
                            
                            # Check section boundary violations
                            elif (unit_type == "section" and prev_unit_type == "section" and
                                  custom_meta.get('section_index') != prev_chunk.metadata.custom_metadata.get('section_index')):
                                validation_results['section_boundary_violations'] += 1
                                validation_results['boundary_violations'] += 1
                            
                            # Check paragraph boundary violations
                            elif (unit_type == "paragraph" and prev_unit_type == "paragraph" and
                                  custom_meta.get('paragraph_index') != prev_chunk.metadata.custom_metadata.get('paragraph_index')):
                                validation_results['paragraph_boundary_violations'] += 1
                                validation_results['boundary_violations'] += 1
                
                except Exception as e:
                    validation_results['errors'].append(f"Error validating chunk {i}: {str(e)}")
            
            validation_results['boundary_compliance_rate'] = (
                (validation_results['total_chunks'] - validation_results['boundary_violations']) / 
                validation_results['total_chunks'] * 100
                if validation_results['total_chunks'] > 0 else 0
            )
            
            logger.info(f"Chunk boundary validation completed: {validation_results['boundary_compliance_rate']:.1f}% compliance")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in chunk boundary validation: {str(e)}")
            return {
                'total_chunks': len(chunks),
                'boundary_violations': len(chunks),
                'errors': [f"Validation failed: {str(e)}"]
            }


class SimpleTokenizer:
    """Simple tokenizer for counting tokens and basic text processing"""

    def __init__(self):
        # Common contractions and abbreviations
        self.contractions = {
            "can't": "can n't",
            "won't": "will n't",
            "shan't": "shall n't",
            "don't": "do n't",
            "doesn't": "does n't",
            "didn't": "did n't",
            "isn't": "is n't",
            "aren't": "are n't",
            "wasn't": "was n't",
            "weren't": "were n't",
            "haven't": "have n't",
            "hasn't": "has n't",
            "hadn't": "had n't",
            "i'm": "i 'm",
            "you're": "you 're",
            "he's": "he 's",
            "she's": "she 's",
            "it's": "it 's",
            "we're": "we 're",
            "they're": "they 're",
            "i've": "i 've",
            "you've": "you 've",
            "we've": "we 've",
            "they've": "they 've",
            "i'd": "i 'd",
            "you'd": "you 'd",
            "he'd": "he 'd",
            "she'd": "she 'd",
            "we'd": "we 'd",
            "they'd": "they 'd",
            "i'll": "i 'll",
            "you'll": "you 'll",
            "he'll": "he 'll",
            "she'll": "she 'll",
            "we'll": "we 'll",
            "they'll": "they 'll",
        }

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/tokens"""
        if not text:
            return []

        # Normalize text
        text = text.lower()

        # Handle contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Simple word tokenization
        words = word_tokenize(text)

        return words

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text"""
        if not tokens:
            return ""

        text = " ".join(tokens)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([\'"])\s+', r'\1', text)
        text = re.sub(r'\s+([\'"])', r'\1', text)

        return text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenize(text))


# Standalone testing function
def test_enhanced_chunker():
    """Test the enhanced chunker with sample data"""
    try:
        from src.core.document_models import DocumentMetadata
        from src.processors.ingestion.text_extractor import StructuralUnit, ExtractedContent
        
        # Create test metadata
        doc_metadata = DocumentMetadata(
            filename="test_document.pdf",
            file_path="/path/to/test_document.pdf",
            file_size=1024,
            file_extension=".pdf",
            mime_type="application/pdf",
            title="Test Document",
            author="Test Author",
            page_count=2
        )
        
        # Create test structural units
        structural_units = [
            StructuralUnit(
                unit_type="page",
                unit_index=1,
                text="This is the first page of the document. It contains important information about the topic. The content is structured and well-organized.",
                metadata={'page_number': 1, 'text_length': 150}
            ),
            StructuralUnit(
                unit_type="page",
                unit_index=2,
                text="This is the second page of the document. It continues the discussion from the first page. The information is comprehensive and detailed.",
                metadata={'page_number': 2, 'text_length': 140}
            )
        ]
        
        # Create extracted content
        extracted_content = ExtractedContent(
            full_text="This is the first page... This is the second page...",
            structural_units=structural_units,
            metadata=doc_metadata,
            extraction_info={'total_pages': 2}
        )
        
        # Test chunker
        chunker = EnhancedDocumentChunker()
        chunks = chunker.chunk_document_with_structure(extracted_content, "test_doc")
        
        print("Enhanced Chunker Test Results:")
        print(f"Structural units: {len(structural_units)}")
        print(f"Chunks created: {len(chunks)}")
        
        for chunk in chunks:
            if chunk.metadata and chunk.metadata.custom_metadata:
                unit_type = chunk.metadata.custom_metadata.get('unit_type', 'unknown')
                unit_index = chunk.metadata.custom_metadata.get('unit_index', 'unknown')
                print(f"Chunk {chunk.chunk_index}: {unit_type} {unit_index} - {chunk.content[:50]}...")
        
        # Test validation
        validation_results = chunker.validate_chunk_boundaries(chunks)
        print(f"Boundary compliance rate: {validation_results['boundary_compliance_rate']:.1f}%")
        
        return chunks
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run test
    test_enhanced_chunker()

