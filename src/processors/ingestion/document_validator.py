"""
Document validation for RAG Agent
Validates file type, size, and content before processing
"""

import os
import magic
from pathlib import Path
from typing import Optional, Tuple, List
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import config

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Validates documents before ingestion"""

    def __init__(self):
        """Initialize validator with configuration"""
        self.max_size_bytes = config.ingestion.max_document_size_mb * 1024 * 1024
        self.allowed_extensions = set(config.ingestion.allowed_extensions)

        # MIME type mappings
        self.mime_to_extension = {
            'application/pdf': '.pdf',
            'text/html': '.html',
            'text/plain': '.txt',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/rtf': '.rtf',
            'text/markdown': '.md',
        }

        self.extension_to_mime = {v: k for k, v in self.mime_to_extension.items()}

    def validate_file(self, file_path: str, original_filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate a file for ingestion with comprehensive checks

        Args:
            file_path: Path to the file to validate
            original_filename: Original filename (used for extension validation when file_path is temp file)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                return False, f"File does not exist: {file_path}"

            # Check if it's a file (not directory)
            if not path.is_file():
                return False, f"Path is not a file: {file_path}"

            # Check file accessibility (read permissions)
            if not path.stat().st_mode & 0o400:  # Check read permission
                return False, f"File is not readable: {file_path}"

            # Check file size with detailed reporting
            file_size = path.stat().st_size
            max_size_mb = self.max_size_bytes / (1024 * 1024)

            if file_size == 0:
                return False, "File is empty (0 bytes)"

            if file_size > self.max_size_bytes:
                return False, f"File size {file_size / (1024*1024):.2f}MB exceeds maximum allowed size of {max_size_mb:.1f}MB"

            # Get file extension and validate
            # Use original filename if provided (for temp files), otherwise use file path
            if original_filename:
                from pathlib import Path as PathLib
                original_path = PathLib(original_filename)
                file_extension = original_path.suffix.lower()
            else:
                file_extension = path.suffix.lower()

            # Check extension against allowed list
            if file_extension not in self.allowed_extensions:
                return False, f"File extension '{file_extension}' not supported. Supported extensions: {', '.join(sorted(self.allowed_extensions))}"

            # Enhanced MIME type validation
            mime_valid, mime_error = self._validate_mime_type(path, file_extension)
            if not mime_valid:
                logger.warning(f"MIME type validation failed for {file_path}: {mime_error}")
                # Continue processing but log the warning

            # File type specific validation
            type_valid, type_error = self._validate_file_type(path, file_extension)
            if not type_valid:
                return False, type_error

            # Check for potential file corruption or issues
            corruption_valid, corruption_error = self._validate_file_integrity(path, file_extension)
            if not corruption_valid:
                return False, corruption_error

            logger.info(f"File validation passed for {file_path} ({file_size} bytes, {file_extension})")
            return True, None

        except PermissionError:
            return False, f"Permission denied accessing file: {file_path}"
        except OSError as e:
            return False, f"OS error accessing file {file_path}: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error validating file {file_path}: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def _validate_pdf(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Validate PDF file"""
        try:
            # Try to read first few bytes to check PDF header
            with open(path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    return False, "Invalid PDF file format"

            return True, None
        except Exception as e:
            return False, f"PDF validation error: {str(e)}"

    def _validate_word_document(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Validate Word document"""
        try:
            # Basic validation - check file isn't corrupted
            with open(path, 'rb') as f:
                header = f.read(512)  # Read first 512 bytes

            # DOC files start with specific bytes
            if path.suffix.lower() == '.doc':
                if len(header) < 8:
                    return False, "DOC file too small"

            return True, None
        except Exception as e:
            return False, f"Word document validation error: {str(e)}"

    def _validate_html(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Validate HTML file"""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024).lower()  # Read first 1KB

            # Check for basic HTML structure
            if '<html' not in content and '<!doctype html' not in content:
                logger.warning("HTML file doesn't contain standard HTML tags")

            return True, None
        except Exception as e:
            return False, f"HTML validation error: {str(e)}"

    def _validate_text(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Validate text file"""
        try:
            # Try to read as text
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)

            if not content.strip():
                return False, "Text file appears to be empty or contains only whitespace"

            return True, None
        except Exception as e:
            return False, f"Text file validation error: {str(e)}"

    def get_file_info(self, file_path: str) -> Optional[dict]:
        """Get basic file information"""
        try:
            path = Path(file_path)
            stat = path.stat()

            return {
                'filename': path.name,
                'extension': path.suffix.lower(),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified_time': stat.st_mtime,
                'created_time': stat.st_ctime,
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return None

    def _validate_mime_type(self, path: Path, file_extension: str) -> Tuple[bool, Optional[str]]:
        """Validate MIME type matches expected type for extension"""
        try:
            mime_type = magic.from_file(str(path), mime=True)

            if not mime_type:
                return False, "Could not detect MIME type"

            expected_mime = self.extension_to_mime.get(file_extension)

            if expected_mime:
                if mime_type == expected_mime:
                    return True, None
                elif file_extension in ['.txt', '.md'] and mime_type.startswith('text/'):
                    # Allow text subtypes for text files
                    return True, None
                elif file_extension == '.html' and mime_type in ['text/html', 'application/xhtml+xml']:
                    # Allow HTML variants
                    return True, None
                else:
                    return False, f"MIME type mismatch: expected {expected_mime}, detected {mime_type}"

            return True, None  # No expected MIME type defined, accept any

        except Exception as e:
            return False, f"MIME type detection failed: {str(e)}"

    def _validate_file_type(self, path: Path, file_extension: str) -> Tuple[bool, Optional[str]]:
        """Validate file type specific requirements"""
        if file_extension == '.pdf':
            return self._validate_pdf(path)
        elif file_extension in ['.doc', '.docx']:
            return self._validate_word_document(path)
        elif file_extension == '.html':
            return self._validate_html(path)
        elif file_extension in ['.txt', '.md']:
            return self._validate_text(path)
        else:
            return True, None  # No specific validation for this type

    def _validate_file_integrity(self, path: Path, file_extension: str) -> Tuple[bool, Optional[str]]:
        """Check for file corruption or integrity issues"""
        try:
            # Basic integrity check - ensure file is readable and has reasonable content
            with open(path, 'rb') as f:
                # Try to read first 1KB
                first_kb = f.read(1024)

                if not first_kb:
                    return False, "File appears to be empty or unreadable"

                # Check for binary vs text files
                if file_extension in ['.txt', '.md', '.html']:
                    try:
                        first_kb.decode('utf-8')
                    except UnicodeDecodeError:
                        return False, "Text file contains invalid UTF-8 characters"

                # File type specific integrity checks
                if file_extension == '.pdf':
                    if not first_kb.startswith(b'%PDF-'):
                        return False, "Invalid PDF file format"
                elif file_extension == '.html':
                    content_str = first_kb.decode('utf-8', errors='ignore').lower()
                    if '<html' not in content_str and '<!doctype html' not in content_str:
                        logger.warning("HTML file doesn't contain standard HTML tags")

            return True, None

        except Exception as e:
            return False, f"File integrity check failed: {str(e)}"

    def validate_batch(self, file_paths: List[str]) -> List[Tuple[str, bool, Optional[str]]]:
        """Validate multiple files"""
        results = []

        for file_path in file_paths:
            is_valid, error = self.validate_file(file_path)
            results.append((file_path, is_valid, error))

        return results
