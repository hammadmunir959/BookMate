"""
Enhanced Text Extractor for BookMate RAG Agent
Extracts text while preserving structural information for citation
"""

import os
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_models import DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class StructuralUnit:
    """Represents a structural unit of text (page, section, paragraph)"""
    
    unit_type: str  # "page", "section", "paragraph"
    unit_index: int
    text: str
    heading: Optional[str] = None
    level: int = 0  # For hierarchical structures
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedContent:
    """Container for extracted text with structural information"""
    
    full_text: str
    structural_units: List[StructuralUnit]
    metadata: DocumentMetadata
    extraction_info: Dict[str, Any] = field(default_factory=dict)


class EnhancedTextExtractor(ABC):
    """Abstract base class for enhanced text extractors"""

    @abstractmethod
    def extract_text_with_structure(self, file_path: str) -> Tuple[str, Optional[ExtractedContent]]:
        """
        Extract text with structural information

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (extracted_text, extracted_content_with_structure)
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions"""
        pass


class EnhancedPDFExtractor(EnhancedTextExtractor):
    """Enhanced PDF extractor that preserves page structure"""

    def __init__(self):
        try:
            import PyPDF2
            self.pdf_lib = 'PyPDF2'
        except ImportError:
            try:
                import pypdf
                self.pdf_lib = 'pypdf'
            except ImportError:
                try:
                    import fitz  # PyMuPDF
                    self.pdf_lib = 'fitz'
                except ImportError:
                    raise ImportError("No PDF library found. Install PyPDF2, pypdf, or PyMuPDF")

    def get_supported_extensions(self) -> list[str]:
        return ['.pdf']

    def extract_text_with_structure(self, file_path: str) -> Tuple[str, Optional[ExtractedContent]]:
        """Extract text from PDF with page structure"""
        try:
            text_parts = []
            structural_units = []
            metadata = None

            if self.pdf_lib == 'PyPDF2':
                text, structural_units, metadata = self._extract_with_pypdf2(file_path)
            elif self.pdf_lib == 'pypdf':
                text, structural_units, metadata = self._extract_with_pypdf(file_path)
            elif self.pdf_lib == 'fitz':
                text, structural_units, metadata = self._extract_with_fitz(file_path)

            if not text.strip():
                return "", None

            # Create extracted content
            extracted_content = ExtractedContent(
                full_text=text,
                structural_units=structural_units,
                metadata=metadata,
                extraction_info={
                    'extraction_method': self.pdf_lib,
                    'total_pages': len(structural_units),
                    'total_units': len(structural_units)
                }
            )

            return text, extracted_content

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            return "", None

    def _extract_with_pypdf2(self, file_path: str) -> Tuple[str, List[StructuralUnit], DocumentMetadata]:
        """Extract using PyPDF2 with structure"""
        import PyPDF2

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Extract metadata
            pdf_metadata = pdf_reader.metadata
            metadata = DocumentMetadata(
                filename=Path(file_path).name,
                file_path=file_path,
                file_size=Path(file_path).stat().st_size,
                file_extension='.pdf',
                mime_type='application/pdf',
                title=pdf_metadata.title if pdf_metadata and pdf_metadata.title else None,
                author=pdf_metadata.author if pdf_metadata and pdf_metadata.author else None,
                page_count=len(pdf_reader.pages)
            )

            # Extract text with page structure
            text_parts = []
            structural_units = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
                    
                    # Create structural unit for page
                    structural_unit = StructuralUnit(
                        unit_type="page",
                        unit_index=page_num,
                        text=page_text,
                        metadata={
                            'page_number': page_num,
                            'text_length': len(page_text),
                            'word_count': len(page_text.split())
                        }
                    )
                    structural_units.append(structural_unit)

            return '\n'.join(text_parts), structural_units, metadata

    def _extract_with_pypdf(self, file_path: str) -> Tuple[str, List[StructuralUnit], DocumentMetadata]:
        """Extract using pypdf with structure"""
        from pypdf import PdfReader

        reader = PdfReader(file_path)

        # Extract metadata
        pdf_metadata = reader.metadata
        metadata = DocumentMetadata(
            filename=Path(file_path).name,
            file_path=file_path,
            file_size=Path(file_path).stat().st_size,
            file_extension='.pdf',
            mime_type='application/pdf',
            title=pdf_metadata.title if pdf_metadata and pdf_metadata.title else None,
            author=pdf_metadata.author if pdf_metadata and pdf_metadata.author else None,
            page_count=len(reader.pages)
        )

        # Extract text with page structure
        text_parts = []
        structural_units = []
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text_parts.append(page_text)
                
                # Create structural unit for page
                structural_unit = StructuralUnit(
                    unit_type="page",
                    unit_index=page_num,
                    text=page_text,
                    metadata={
                        'page_number': page_num,
                        'text_length': len(page_text),
                        'word_count': len(page_text.split())
                    }
                )
                structural_units.append(structural_unit)

        return '\n'.join(text_parts), structural_units, metadata

    def _extract_with_fitz(self, file_path: str) -> Tuple[str, List[StructuralUnit], DocumentMetadata]:
        """Extract using PyMuPDF with structure"""
        import fitz

        doc = fitz.open(file_path)

        # Extract metadata
        pdf_metadata = doc.metadata
        metadata = DocumentMetadata(
            filename=Path(file_path).name,
            file_path=file_path,
            file_size=Path(file_path).stat().st_size,
            file_extension='.pdf',
            mime_type='application/pdf',
            title=pdf_metadata.get('title'),
            author=pdf_metadata.get('author'),
            page_count=doc.page_count
        )

        # Extract text with page structure
        text_parts = []
        structural_units = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
                
                # Create structural unit for page
                structural_unit = StructuralUnit(
                    unit_type="page",
                    unit_index=page_num + 1,
                    text=page_text,
                    metadata={
                        'page_number': page_num + 1,
                        'text_length': len(page_text),
                        'word_count': len(page_text.split())
                    }
                )
                structural_units.append(structural_unit)

        doc.close()
        return '\n'.join(text_parts), structural_units, metadata


class EnhancedDocxExtractor(EnhancedTextExtractor):
    """Enhanced DOCX extractor that preserves heading structure"""

    def __init__(self):
        try:
            import docx
            self.docx_available = True
        except ImportError:
            self.docx_available = False
            logger.warning("python-docx not installed. DOCX extraction will not work.")

    def get_supported_extensions(self) -> list[str]:
        return ['.docx']

    def extract_text_with_structure(self, file_path: str) -> Tuple[str, Optional[ExtractedContent]]:
        """Extract text from DOCX with heading structure"""
        if not self.docx_available:
            raise ImportError("python-docx is required for DOCX extraction")

        try:
            import docx

            doc = docx.Document(file_path)

            # Extract metadata
            core_props = doc.core_properties
            metadata = DocumentMetadata(
                filename=Path(file_path).name,
                file_path=file_path,
                file_size=Path(file_path).stat().st_size,
                file_extension='.docx',
                mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                title=core_props.title if core_props.title else None,
                author=core_props.author if core_props.author else None,
                creation_date=core_props.created if core_props.created else None,
                modification_date=core_props.modified if core_props.modified else None
            )

            # Extract text with structure
            text_parts = []
            structural_units = []
            current_section = None
            section_index = 0
            paragraph_index = 0

            for para in doc.paragraphs:
                para_text = para.text.strip()
                if not para_text:
                    continue

                # Check if paragraph is a heading
                if para.style.name.startswith('Heading'):
                    # Save previous section if exists
                    if current_section:
                        structural_units.append(current_section)
                    
                    # Start new section
                    heading_level = int(para.style.name.split()[-1]) if para.style.name.split()[-1].isdigit() else 1
                    current_section = StructuralUnit(
                        unit_type="section",
                        unit_index=section_index,
                        text=para_text,
                        heading=para_text,
                        level=heading_level,
                        metadata={
                            'heading_level': heading_level,
                            'paragraph_count': 0,
                            'text_length': len(para_text)
                        }
                    )
                    section_index += 1
                    paragraph_index = 0
                else:
                    # Add paragraph to current section
                    if current_section:
                        current_section.text += '\n' + para_text
                        current_section.metadata['paragraph_count'] += 1
                        current_section.metadata['text_length'] += len(para_text)
                    else:
                        # Create a default section for paragraphs without headings
                        current_section = StructuralUnit(
                            unit_type="section",
                            unit_index=section_index,
                            text=para_text,
                            heading="Introduction",
                            level=1,
                            metadata={
                                'heading_level': 1,
                                'paragraph_count': 1,
                                'text_length': len(para_text)
                            }
                        )
                        section_index += 1

                    paragraph_index += 1

                text_parts.append(para_text)

            # Add final section
            if current_section:
                structural_units.append(current_section)

            full_text = '\n'.join(text_parts)

            # Create extracted content
            extracted_content = ExtractedContent(
                full_text=full_text,
                structural_units=structural_units,
                metadata=metadata,
                extraction_info={
                    'extraction_method': 'python-docx',
                    'total_sections': len(structural_units),
                    'total_paragraphs': len(text_parts)
                }
            )

            return full_text, extracted_content

        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return "", None


class EnhancedHTMLExtractor(EnhancedTextExtractor):
    """Enhanced HTML extractor that preserves heading structure"""

    def __init__(self):
        try:
            from bs4 import BeautifulSoup
            self.bs4_available = True
        except ImportError:
            self.bs4_available = False
            logger.warning("BeautifulSoup4 not installed. HTML extraction will be basic.")

    def get_supported_extensions(self) -> list[str]:
        return ['.html', '.htm']

    def extract_text_with_structure(self, file_path: str) -> Tuple[str, Optional[ExtractedContent]]:
        """Extract text from HTML with heading structure"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                html_content = file.read()

            if self.bs4_available:
                text, structural_units, title = self._extract_with_bs4(html_content)
            else:
                text, structural_units, title = self._extract_basic_html(html_content)

            # Extract metadata
            metadata = DocumentMetadata(
                filename=Path(file_path).name,
                file_path=file_path,
                file_size=Path(file_path).stat().st_size,
                file_extension=Path(file_path).suffix.lower(),
                mime_type='text/html',
                title=title
            )

            # Create extracted content
            extracted_content = ExtractedContent(
                full_text=text,
                structural_units=structural_units,
                metadata=metadata,
                extraction_info={
                    'extraction_method': 'BeautifulSoup' if self.bs4_available else 'basic',
                    'total_sections': len(structural_units)
                }
            )

            return text, extracted_content

        except Exception as e:
            logger.error(f"Error extracting text from HTML {file_path}: {str(e)}")
            return "", None

    def _extract_with_bs4(self, html_content: str) -> Tuple[str, List[StructuralUnit], Optional[str]]:
        """Extract text using BeautifulSoup with structure"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract title
        title = None
        if soup.title:
            title = soup.title.string

        # Extract text with structure
        text_parts = []
        structural_units = []
        section_index = 0

        # Find all headings and content
        current_section = None
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
            if element.name.startswith('h'):
                # Save previous section
                if current_section:
                    structural_units.append(current_section)
                
                # Start new section
                heading_level = int(element.name[1])
                heading_text = element.get_text().strip()
                
                current_section = StructuralUnit(
                    unit_type="section",
                    unit_index=section_index,
                    text=heading_text,
                    heading=heading_text,
                    level=heading_level,
                    metadata={
                        'heading_level': heading_level,
                        'tag': element.name,
                        'text_length': len(heading_text)
                    }
                )
                section_index += 1
                text_parts.append(heading_text)
                
            elif element.name in ['p', 'div'] and current_section:
                # Add content to current section
                content_text = element.get_text().strip()
                if content_text:
                    current_section.text += '\n' + content_text
                    current_section.metadata['text_length'] += len(content_text)
                    text_parts.append(content_text)

        # Add final section
        if current_section:
            structural_units.append(current_section)

        # Clean up whitespace
        text = '\n'.join(text_parts)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        return text, structural_units, title

    def _extract_basic_html(self, html_content: str) -> Tuple[str, List[StructuralUnit], Optional[str]]:
        """Basic HTML text extraction without BeautifulSoup"""
        import re

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)

        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        title = title_match.group(1) if title_match else None

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Create a single structural unit
        structural_unit = StructuralUnit(
            unit_type="section",
            unit_index=0,
            text=text,
            heading="Content",
            level=1,
            metadata={
                'heading_level': 1,
                'text_length': len(text)
            }
        )

        return text, [structural_unit], title


class EnhancedTextExtractor(EnhancedTextExtractor):
    """Enhanced plain text extractor that preserves paragraph structure"""

    def get_supported_extensions(self) -> list[str]:
        return ['.txt', '.md', '.rtf']

    def extract_text_with_structure(self, file_path: str) -> Tuple[str, Optional[ExtractedContent]]:
        """Extract text from plain text file with paragraph structure"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

            # Extract metadata
            metadata = DocumentMetadata(
                filename=Path(file_path).name,
                file_path=file_path,
                file_size=Path(file_path).stat().st_size,
                file_extension=Path(file_path).suffix.lower(),
                mime_type='text/plain'
            )

            # Split into paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Create structural units for paragraphs
            structural_units = []
            for i, para in enumerate(paragraphs):
                structural_unit = StructuralUnit(
                    unit_type="paragraph",
                    unit_index=i,
                    text=para,
                    metadata={
                        'paragraph_index': i,
                        'text_length': len(para),
                        'word_count': len(para.split())
                    }
                )
                structural_units.append(structural_unit)

            # Create extracted content
            extracted_content = ExtractedContent(
                full_text=content,
                structural_units=structural_units,
                metadata=metadata,
                extraction_info={
                    'extraction_method': 'plain_text',
                    'total_paragraphs': len(structural_units)
                }
            )

            return content, extracted_content

        except Exception as e:
            logger.error(f"Error extracting text from text file {file_path}: {str(e)}")
            return "", None


class EnhancedDocumentTextExtractor:
    """Main enhanced text extractor that delegates to specific extractors"""

    def __init__(self):
        self.extractors = {}
        
        # Initialize PDF extractor if available
        try:
            self.extractors['.pdf'] = EnhancedPDFExtractor()
        except ImportError:
            logger.warning("PDF libraries not installed. .pdf files will not be supported.")
        
        # Initialize DOCX extractor if available
        try:
            self.extractors['.docx'] = EnhancedDocxExtractor()
        except ImportError:
            logger.warning("python-docx not installed. .docx files will not be supported.")
        
        # Initialize HTML extractor
        self.extractors['.html'] = EnhancedHTMLExtractor()
        self.extractors['.htm'] = EnhancedHTMLExtractor()
        
        # Initialize text extractors
        self.extractors['.txt'] = EnhancedTextExtractor()
        self.extractors['.md'] = EnhancedTextExtractor()
        self.extractors['.rtf'] = EnhancedTextExtractor()

        # Initialize DOC extractor if available
        try:
            import docx  # This will work for both .doc and .docx
            self.extractors['.doc'] = EnhancedDocxExtractor()
        except ImportError:
            logger.warning("python-docx not installed. .doc files will not be supported.")

    def extract_text_with_structure(self, file_path: str, original_filename: Optional[str] = None) -> Tuple[str, Optional[ExtractedContent]]:
        """
        Extract text from any supported document format with structure

        Args:
            file_path: Path to the document file
            original_filename: Original filename (used for extension validation when file_path is temp file)

        Returns:
            Tuple of (extracted_text, extracted_content_with_structure)
        """
        path = Path(file_path)

        # Use original filename for extension if provided
        if original_filename:
            from pathlib import Path as PathLib
            original_path = PathLib(original_filename)
            extension = original_path.suffix.lower()
        else:
            extension = path.suffix.lower()

        if extension not in self.extractors:
            raise ValueError(f"Unsupported file extension: {extension}")

        extractor = self.extractors[extension]
        return extractor.extract_text_with_structure(file_path)

    def get_supported_extensions(self) -> list[str]:
        """Get all supported file extensions"""
        return list(self.extractors.keys())


# Standalone testing function
def test_enhanced_text_extractor():
    """Test the enhanced text extractor with sample data"""
    try:
        # Create a test text file
        test_file = "/tmp/test_document.txt"
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
        
        # Test extractor
        extractor = EnhancedDocumentTextExtractor()
        text, extracted_content = extractor.extract_text_with_structure(test_file)
        
        print("Enhanced Text Extractor Test Results:")
        print(f"Full text length: {len(text)}")
        print(f"Structural units: {len(extracted_content.structural_units)}")
        
        for unit in extracted_content.structural_units:
            print(f"Unit {unit.unit_index}: {unit.unit_type} - {unit.text[:50]}...")
        
        # Cleanup
        os.remove(test_file)
        print("Test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test
    test_enhanced_text_extractor()

