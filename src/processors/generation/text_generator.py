"""
Generation Pipeline for RAG Generation Microservice
Handles LLM-powered text generation with citation management
"""

import logging
import time
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.core.config import config
from src.core.document_models import (
    GenerationRequest, GenerationResult, Citation, CitationType,
    GenerationType, ModelProvider
)
from src.utils.llm_client import llm_client
from src.core.service_manager import service_manager
from src.utils.logging_utils import generation_logger

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Handles the complete generation workflow"""

    def __init__(self):
        self._llm_client = None
        self._cache_manager = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = llm_client
        return self._llm_client

    @property
    def cache_manager(self):
        if self._cache_manager is None:
            self._cache_manager = service_manager.get_service('cache_manager')
        return self._cache_manager
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResult:
        """
        Generate text response using the provided augmented prompt
        
        Args:
            request: GenerationRequest with prompt and parameters
            
        Returns:
            GenerationResult with answer and metadata
        """
        start_time = time.time()
        request_id = request.request_hash

        try:
            # Log generation pipeline start
            generation_logger.log_pipeline_start("generation", request_id, prompt_length=len(request.augmented_prompt))

            logger.info(f"🚀 Starting generation for request: {request.request_hash}")

            # Check cache first
            cached_result = self._get_cached_result(request.request_hash)
            if cached_result:
                logger.info(f"✅ Using cached result for request: {request.request_hash}")
                generation_logger.log_generation_step("cache_check", request_id, "HIT - using cached result")
                processing_time = time.time() - start_time
                generation_logger.log_pipeline_end("generation", request_id, processing_time, "completed",
                                                cached=True)
                return cached_result
            else:
                generation_logger.log_generation_step("cache_check", request_id, "MISS - generating new response")
            
            # Prepare the prompt for LLM
            final_prompt = self._prepare_prompt(request)
            generation_logger.log_generation_step("prompt_preparation", request_id, f"prepared {len(final_prompt)} char prompt")

            # Generate response using LLM
            llm_response = await self._call_llm(final_prompt, request)
            if not llm_response:
                generation_logger.log_generation_step("llm_call", request_id, "failed - empty response")
                raise Exception("LLM returned empty response")

            generation_logger.log_generation_step("llm_call", request_id, f"generated {len(llm_response)} char response")

            # Process and format the response
            answer, citations = self._process_response(llm_response, request)
            generation_logger.log_generation_step("response_processing", request_id, f"processed response with {len(citations)} citations")

            # Remove duplicate content from the answer
            answer = self._deduplicate_answer(answer)

            # Add markdown formatting to improve readability
            try:
                answer = self._add_markdown_formatting(answer)
            except Exception as e:
                logger.warning(f"Markdown formatting failed: {str(e)}")
                # Continue with original answer
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = GenerationResult(
                success=True,
                answer=answer,
                citations_used=citations,
                model_used=self._get_model_name(request),
                processing_time=processing_time,
                session_id=request.session_id,
                user_id=request.user_id
            )
            
            # Cache the result
            self._cache_result(request.request_hash, result)
            generation_logger.log_generation_step("caching", request_id, "result cached for future use")

            logger.info(f"✅ Generation completed successfully in {processing_time:.2f}s")

            # Log pipeline completion
            generation_logger.log_pipeline_end("generation", request_id, processing_time, "completed",
                                            answer_length=len(answer), citations_used=len(citations))

            return result
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)

            processing_time = time.time() - start_time

            # Log pipeline failure
            generation_logger.log_pipeline_end("generation", request_id, processing_time, f"failed: {str(e)}")

            return GenerationResult(
                success=False,
                answer="",
                citations_used=[],
                model_used=self._get_model_name(request),
                processing_time=processing_time,
                session_id=request.session_id,
                user_id=request.user_id,
                error_message=error_msg
            )
    
    def _prepare_prompt(self, request: GenerationRequest) -> str:
        """Prepare the final prompt for LLM"""
        # The augmented_prompt from retrieval already contains system instructions,
        # query, context, and citations, so we can use it directly
        return request.augmented_prompt
    
    async def _call_llm(self, prompt: str, request: GenerationRequest) -> Optional[str]:
        """Call the LLM with the prepared prompt"""
        try:
            # Get generation parameters
            temperature = request.temperature or config.generation.temperature
            max_tokens = request.max_tokens or config.generation.max_output_tokens
            
            # Call LLM
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                expect_json=False  # We expect natural language response
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return None
    
    def _process_response(self, llm_response: str, request: GenerationRequest) -> tuple[str, List[Citation]]:
        """Process LLM response and extract citations"""
        # Clean and format the answer
        answer = self._clean_answer(llm_response)
        
        # Extract citations from the response
        citations = self._extract_citations(answer, request)
        
        return answer, citations
    
    def _clean_answer(self, response: str) -> str:
        """Clean and format the LLM response"""
        if not response:
            return ""
        
        # Remove any unwanted artifacts
        response = response.strip()
        
        # Ensure proper formatting
        response = re.sub(r'\n{3,}', '\n\n', response)  # Max 2 consecutive newlines
        
        return response
    
    def _extract_citations(self, answer: str, request: GenerationRequest) -> List[Citation]:
        """Extract citation information from the answer"""
        citations = []

        # Look for citation patterns in the answer
        # Support both old [CIT:chunk_id] and new (cit#N) formats

        # Find all (cit#N) patterns
        citation_pattern_new = r'\(cit#(\d+(?:\s*,\s*cit#\d+)*)\)'
        matches_new = re.findall(citation_pattern_new, answer)

        # Process new format citations
        for citation_match in matches_new:
            # Handle multiple citations like (cit#1, cit#2)
            citation_numbers = re.findall(r'cit#(\d+)', citation_match)

            for number in citation_numbers:
                citation_number = int(number)
                # We need to map citation number back to chunk_id
                # This is a limitation - we need to maintain a mapping
                chunk_id = f"citation_{citation_number}"  # Placeholder

                citation = Citation(
                    type=CitationType.CHUNK,
                    reference_id=chunk_id,
                    metadata={
                        'document_id': self._extract_document_id_from_chunk_id(chunk_id),
                        'citation_text': f"(cit#{citation_number})",
                        'citation_number': citation_number
                    }
                )
                citations.append(citation)

        # Also check for old [CIT:chunk_id] patterns for backward compatibility
        citation_pattern_old = r'\[CIT:([^\]]+)\]'
        matches_old = re.findall(citation_pattern_old, answer)

        for chunk_id in matches_old:
            citation = Citation(
                type=CitationType.CHUNK,
                reference_id=chunk_id,
                metadata={
                    'document_id': self._extract_document_id_from_chunk_id(chunk_id),
                    'citation_text': f"[{chunk_id}]",
                    'citation_format': 'old'
                }
            )
            citations.append(citation)

        return citations
    
    def _extract_document_id_from_chunk_id(self, chunk_id: str) -> str:
        """Extract document ID from chunk ID"""
        # Assuming chunk_id format is "document.pdf_chunk_0"
        if '_chunk_' in chunk_id:
            return chunk_id.split('_chunk_')[0]
        return chunk_id
    
    def _get_model_name(self, request: GenerationRequest) -> str:
        """Get the model name being used"""
        if request.model_name:
            return request.model_name
        
        # Default based on provider
        provider = request.model_provider or ModelProvider.GROQ
        if provider == ModelProvider.GROQ:
            return config.model.groq_model
        elif provider == ModelProvider.OPENAI:
            return config.model.openai_model
        elif provider == ModelProvider.ANTHROPIC:
            return config.model.anthropic_model
        
        return config.model.groq_model
    
    def _get_cached_result(self, request_hash: str) -> Optional[GenerationResult]:
        """Get cached generation result"""
        try:
            cache_key = f"generation_{request_hash}"
            cached_data = self.cache_manager.get(cache_key)
            
            if cached_data:
                # Convert cached citations back to Citation objects
                if 'citations_used' in cached_data and cached_data['citations_used']:
                    cached_data['citations_used'] = [
                        self._dict_to_citation(c) for c in cached_data['citations_used']
                    ]

                # Convert cached dict back to GenerationResult
                return GenerationResult(**cached_data)
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
        
        return None
    
    def _cache_result(self, request_hash: str, result: GenerationResult):
        """Cache generation result"""
        try:
            cache_key = f"generation_{request_hash}"
            # Cache as dict for easier serialization
            result_dict = {
                'success': result.success,
                'answer': result.answer,
                'citations_used': [self._citation_to_dict(c) for c in result.citations_used],
                'model_used': result.model_used,
                'processing_time': result.processing_time,
                'session_id': result.session_id,
                'user_id': result.user_id,
                'error_message': result.error_message
            }
            
            self.cache_manager.set(cache_key, result_dict, ttl_hours=1)
            
        except Exception as e:
            logger.warning(f"Caching failed: {str(e)}")

    def _citation_to_dict(self, citation: Citation) -> Dict[str, Any]:
        """Convert Citation object to dictionary for caching"""
        return {
            'type': citation.type.value,  # Convert enum to string
            'page_number': citation.page_number,
            'section': citation.section,
            'heading': citation.heading,
            'paragraph_number': citation.paragraph_number,
            'reference_id': citation.reference_id,
            'url': citation.url,
            'metadata': citation.metadata
        }

    def _dict_to_citation(self, citation_dict: Dict[str, Any]) -> Citation:
        """Convert dictionary back to Citation object"""
        return Citation(
            type=CitationType(citation_dict['type']),  # Convert string back to enum
            page_number=citation_dict.get('page_number'),
            section=citation_dict.get('section'),
            heading=citation_dict.get('heading'),
            paragraph_number=citation_dict.get('paragraph_number'),
            reference_id=citation_dict.get('reference_id'),
            url=citation_dict.get('url'),
            metadata=citation_dict.get('metadata', {})
        )

    def _deduplicate_answer(self, answer: str) -> str:
        """
        Remove duplicate sentences and phrases from generated answers

        Args:
            answer: The generated answer text

        Returns:
            Deduplicated answer
        """
        try:
            if not answer or len(answer.strip()) < 10:
                return answer

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

            # Remove duplicate sentences
            seen_sentences = set()
            deduplicated_sentences = []

            for sentence in sentences:
                sentence_clean = sentence.strip()
                if len(sentence_clean.split()) < 3:  # Skip very short sentences
                    deduplicated_sentences.append(sentence)
                    continue

                sentence_lower = sentence_clean.lower()

                # Check for exact duplicates
                if sentence_lower not in seen_sentences:
                    # Check for near-duplicates by looking for repeated key terms
                    is_duplicate = False
                    # Simple check: if sentence contains the same key medical terms as a previous sentence
                    key_terms = ['numbness', 'tingling', 'palpitations', 'sweating', 'trembling']
                    current_terms = [term for term in key_terms if term in sentence_lower]

                    for seen in seen_sentences:
                        seen_terms = [term for term in key_terms if term in seen]
                        # If sentences share the same key medical terms, likely duplicate
                        if current_terms and seen_terms and set(current_terms) == set(seen_terms):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        deduplicated_sentences.append(sentence_clean)
                        seen_sentences.add(sentence_lower)

            # Join sentences with proper punctuation
            result = '. '.join(s.rstrip('.') for s in deduplicated_sentences if s.strip())
            if result and not result.endswith('.'):
                result += '.'

            return result

        except Exception as e:
            logger.warning(f"Error deduplicating answer: {str(e)}")
            return answer

    def _add_markdown_formatting(self, answer: str) -> str:
        """
        Add comprehensive markdown formatting to improve answer readability

        Args:
            answer: The raw answer text

        Returns:
            Answer with comprehensive markdown formatting added
        """
        logger.info(f"Adding markdown formatting to answer (length: {len(answer)})")
        try:
            if not answer or len(answer.strip()) < 10:
                return answer

            formatted_answer = answer

            # Step 1: Clean up excessive spacing
            formatted_answer = self._clean_spacing(formatted_answer)

            # Step 2: Add structured formatting
            formatted_answer = self._add_structured_formatting(formatted_answer)

            # Step 3: Format section headers
            formatted_answer = self._format_section_headers(formatted_answer)

            # Step 4: Format lists and bullet points
            formatted_answer = self._format_lists_and_bullets(formatted_answer)

            # Step 5: Add emphasis formatting
            formatted_answer = self._add_emphasis_formatting(formatted_answer)

            # Step 6: Format code and technical terms
            formatted_answer = self._format_code_and_technical_terms(formatted_answer)

            # Step 7: Final cleanup
            formatted_answer = self._final_cleanup(formatted_answer)

            return formatted_answer

        except Exception as e:
            logger.warning(f"Error adding markdown formatting: {str(e)}")
            return answer

    def _clean_spacing(self, text: str) -> str:
        """Clean up excessive spacing and newlines for better readability"""
        if not text:
            return text

        # Replace 4+ consecutive newlines with double newlines
        text = re.sub(r'\n{4,}', '\n\n', text)

        # Remove excessive spaces (3+ consecutive spaces)
        text = re.sub(r' {3,}', '  ', text)

        # Clean up spacing around punctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+([.,;:])', r'\1', text)

        # Fix spacing after opening parentheses and before closing ones
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)

        # Ensure single space after colons and semicolons
        text = re.sub(r'[:;]\s*', r'\1 ', text)

        # Remove trailing spaces from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # Remove empty lines at the beginning and end
        text = text.strip()

        return text

    def _add_structured_formatting(self, text: str) -> str:
        """Add structured formatting for better readability"""
        lines = text.split('\n')
        formatted_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines
            if not line:
                formatted_lines.append('')
                continue

            # Format numbered steps or procedures
            if re.match(r'^\d+\.?\s+', line):
                formatted_lines.append(line)
                continue

            # Format key points or important information
            if line.startswith(('•', '-', '*')):
                formatted_lines.append(line)
                continue

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_section_headers(self, text: str) -> str:
        """Format section headers with proper markdown"""
        lines = text.split('\n')
        formatted_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Check if line looks like a section header
            if (re.match(r'^[A-Z][^.!?]*[:]?$', line) and
                len(line) < 60 and
                not line.startswith('(') and
                not line.startswith('**') and
                not line.startswith('#') and
                not re.match(r'^\d+\.', line)):

                # Convert to level 2 header if it's a main section
                if len(line) < 30 and not any(char.isdigit() for char in line):
                    formatted_lines.append(f'## {line.rstrip(":")}')
                else:
                    # Use bold for subsections
                    formatted_lines.append(f'**{line.rstrip(":")}**')

                # Add spacing after header
                if i < len(lines) - 1 and lines[i + 1].strip():
                    formatted_lines.append('')

            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_lists_and_bullets(self, text: str) -> str:
        """Format lists and bullet points properly"""
        lines = text.split('\n')
        formatted_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for bullet points
            if line.startswith(('•', '-', '*')):
                # Ensure proper markdown bullet format
                content = line.lstrip('•-*').strip()
                formatted_lines.append(f'- {content}')

                # Check for continuation lines
                j = i + 1
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(('•', '-', '*')):
                    if not re.match(r'^\d+\.', lines[j].strip()):
                        formatted_lines.append(f'  {lines[j].strip()}')
                        i = j
                    j += 1
            else:
                formatted_lines.append(line)

            i += 1

        return '\n'.join(formatted_lines)

    def _add_emphasis_formatting(self, text: str) -> str:
        """Add emphasis formatting for important terms"""
        # Define terms to emphasize based on context
        important_terms = [
            # Medical and psychological terms
            r'\b(depression|anxiety|disorder|symptoms)\b',
            r'\b(treatment|therapy|diagnosis)\b',
            r'\b(medication|therapy|counseling)\b',
            r'\b(severe|chronic|acute|persistent)\b',

            # Important concepts
            r'\b(important|critical|essential|key|main)\b',
            r'\b(recommended|advised|suggested)\b',
            r'\b(however|therefore|consequently)\b',
            r'\b(note|remember|please|important)\b',
        ]

        formatted_text = text

        for term in important_terms:
            # Add emphasis if not already emphasized and not in citation
            pattern = r'\b(' + term + r')\b'
            if not re.search(r'\*.*' + pattern + r'.*\*', formatted_text, re.IGNORECASE):
                formatted_text = re.sub(
                    pattern + r'(?!(?:\*\*|\(cit|#|\d))',
                    r'*\1*',
                    formatted_text,
                    flags=re.IGNORECASE
                )

        return formatted_text

    def _format_code_and_technical_terms(self, text: str) -> str:
        """Format code snippets and technical terms"""
        formatted_text = text

        # Format inline code for technical terms
        technical_terms = [
            r'\b(DSM-5|ICD-10|APA|WHO)\b',
            r'\b(MDD|GAD|PTSD|SAD)\b',
            r'\b(SSRIs|TCAs|MAOIs|SNRI)\b',
            r'\b(CBT|DBT|ACT|IPT)\b',
            r'\b(HTTP|HTTPS|API|URL|JSON|XML)\b',
            r'\b(SQL|NoSQL|REST|GraphQL)\b',
        ]

        for term in technical_terms:
            if not re.search(r'`.*' + term + r'.*`', formatted_text, re.IGNORECASE):
                formatted_text = re.sub(
                    term + r'(?!(?:`|\(cit))',
                    r'`\1`',
                    formatted_text,
                    flags=re.IGNORECASE
                )

        # Format code blocks for multi-line code
        formatted_text = self._format_code_blocks(formatted_text)

        # Format tables if present
        formatted_text = self._format_tables(formatted_text)

        return formatted_text

    def _format_code_blocks(self, text: str) -> str:
        """Format multi-line code blocks"""
        # Look for indented code blocks or code between triple backticks
        lines = text.split('\n')
        formatted_lines = []
        in_code_block = False
        code_block_lines = []

        for line in lines:
            # Check if line contains triple backticks
            if '```' in line:
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    code_block_lines = [line]
                else:
                    # End of code block
                    code_block_lines.append(line)
                    # Process the code block
                    formatted_lines.extend(self._process_code_block(code_block_lines))
                    in_code_block = False
                    code_block_lines = []
            elif in_code_block:
                code_block_lines.append(line)
            else:
                formatted_lines.append(line)

        # Handle unclosed code blocks
        if in_code_block and code_block_lines:
            formatted_lines.extend(self._process_code_block(code_block_lines))

        return '\n'.join(formatted_lines)

    def _process_code_block(self, code_lines: list[str]) -> list[str]:
        """Process and format a code block"""
        if not code_lines:
            return code_lines

        # Ensure proper code block formatting
        processed_lines = []

        # Add opening backticks if not present
        if not code_lines[0].strip().startswith('```'):
            processed_lines.append('```')
            processed_lines.extend(code_lines)
        else:
            processed_lines = code_lines

        # Add closing backticks if not present
        if processed_lines and not processed_lines[-1].strip().startswith('```'):
            processed_lines.append('```')

        return processed_lines

    def _format_tables(self, text: str) -> str:
        """Format tables in markdown format"""
        lines = text.split('\n')
        formatted_lines = []
        in_table = False
        table_lines = []

        for line in lines:
            # Check if line looks like a table row (contains | characters)
            if '|' in line and not line.strip().startswith('#'):
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table:
                    # Process the table
                    formatted_lines.extend(self._process_table(table_lines))
                    in_table = False
                    table_lines = []
                formatted_lines.append(line)

        # Handle unclosed tables
        if in_table and table_lines:
            formatted_lines.extend(self._process_table(table_lines))

        return '\n'.join(formatted_lines)

    def _process_table(self, table_lines: list[str]) -> list[str]:
        """Process and format table lines"""
        if len(table_lines) < 2:
            return table_lines

        processed_lines = []

        # Ensure all rows have consistent column separators
        for i, line in enumerate(table_lines):
            if '|' in line:
                # Ensure line starts and ends with |
                if not line.strip().startswith('|'):
                    line = '| ' + line.strip()
                if not line.strip().endswith('|'):
                    line = line.rstrip() + ' |'
                processed_lines.append(line)
            else:
                processed_lines.append(line)

        return processed_lines

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and formatting touches"""
        if not text:
            return text

        # Ensure proper spacing around headers
        text = re.sub(r'\n(#{1,6}\s*[^\n]+)\n', r'\n\n\1\n\n', text)

        # Ensure proper spacing around bold text
        text = re.sub(r'\n(\*\*[^*\n]+\*\*)\n', r'\n\n\1\n\n', text)

        # Ensure proper spacing around lists
        text = re.sub(r'\n(- [^\n]+)\n', r'\n\n\1\n', text)

        # Clean up any remaining excessive spacing
        text = re.sub(r'\n{4,}', '\n\n', text)

        # Ensure single blank line between paragraphs
        text = re.sub(r'\n{3}', '\n\n', text)

        # Remove trailing spaces and ensure consistent line endings
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # Remove empty lines at start and end
        text = text.strip()

        # Ensure the text ends with a single newline
        return text + '\n'

    async def shutdown(self):
        """Shutdown the pipeline"""
        logger.info("🛑 Shutting down generation pipeline...")
        # Add any cleanup logic here if needed


# Global pipeline instance
generation_pipeline = GenerationPipeline()
