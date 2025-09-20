"""
Shared Data Models for RAG Microservice
Combines models from ingestion, retrieval, and generation services
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import uuid


@dataclass
class DocumentMetadata:
    """Metadata associated with a document"""

    filename: str
    file_path: str
    file_size: int
    file_extension: str
    mime_type: str
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: Optional[int] = None
    language: Optional[str] = "en"
    source_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_hash(self) -> str:
        """Generate a hash based on file content and metadata"""
        content = f"{self.filename}{self.file_path}{self.file_size}{self.modification_date}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""

    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: Optional[DocumentMetadata] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def chunk_hash(self) -> str:
        """Generate a hash for the chunk content"""
        return hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with ranking info"""

    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    semantic_score: float
    keyword_score: float
    final_score: float
    citation: Any  # Will be defined later
    metadata: Dict[str, Any]
    ranking_position: int
    retrieval_method: str  # "semantic", "keyword", "hybrid"
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing"""
        # Ensure final_score is the maximum of semantic and keyword scores
        if self.final_score == 0.0:
            self.final_score = max(self.semantic_score, self.keyword_score)


@dataclass
class QueryResult:
    """Complete result from a query operation"""

    query: str
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    generated_answer: Optional[str] = None
    confidence_score: float = 0.0
    source_documents: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def has_answer(self) -> bool:
        """Check if a valid answer was generated"""
        return (
            self.generated_answer is not None
            and len(self.generated_answer.strip()) > 0
            and not self.generated_answer.lower().startswith("answer not found")
        )


@dataclass
class CacheEntry:
    """Cache entry for embeddings or responses"""

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    ttl_hours: int = 24

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        from datetime import timedelta
        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time


@dataclass
class IngestionStatus:
    """Status tracking for document ingestion"""

    document_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float = 0.0
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def is_completed(self) -> bool:
        return self.status == 'completed'

    @property
    def is_failed(self) -> bool:
        return self.status == 'failed'

    @property
    def processing_time(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class ChunkSummary:
    """Summary for a single document chunk"""

    chunk_summary_id: str
    document_id: str
    chunk_id: str
    summary_text: str
    created_at: datetime = field(default_factory=datetime.now)
    chunk_index: Optional[int] = None
    word_count: Optional[int] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        """Calculate word count after initialization"""
        if self.word_count is None:
            self.word_count = len(self.summary_text.split())


@dataclass
class DocumentSummary:
    """Final summary for an entire document"""

    summary_id: str
    document_id: str
    summary_text: str
    chunk_count: int
    created_at: datetime = field(default_factory=datetime.now)
    word_count: Optional[int] = None
    processing_time: Optional[float] = None
    summary_type: str = "comprehensive"  # "comprehensive", "executive", "brief"

    def __post_init__(self):
        """Calculate word count after initialization"""
        if self.word_count is None:
            self.word_count = len(self.summary_text.split())


@dataclass
class SummaryJob:
    """Background job for document summarization"""

    job_id: str
    document_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    total_chunks: int = 0
    processed_chunks: int = 0

    @property
    def progress(self) -> float:
        """Calculate progress percentage"""
        if self.total_chunks == 0:
            return 0.0
        return self.processed_chunks / self.total_chunks

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def processing_time(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# Additional models for the unified service

class QueryType(Enum):
    """Types of queries for retrieval"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FACTUAL = "factual"
    SUMMARIZATION = "summarization"


class GenerationType(Enum):
    """Types of generation requests"""
    ANSWER = "answer"
    SUMMARY = "summary"
    EXPLANATION = "explanation"


class ModelProvider(Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class CitationType(Enum):
    """Types of citations"""
    PAGE = "page"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    CHUNK = "chunk"
    HEADING = "heading"


@dataclass
class Citation:
    """Citation information for a retrieved chunk"""

    type: CitationType
    page_number: Optional[int] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    paragraph_number: Optional[int] = None
    reference_id: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuery:
    """Query for retrieval operations"""

    query: str
    query_type: QueryType = QueryType.HYBRID
    filters: Optional[Dict[str, Any]] = None
    document_ids: Optional[List[str]] = None
    top_k: int = 5
    similarity_threshold: float = 0.2
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing"""
        if self.filters is None:
            self.filters = {}

        # Generate query hash for caching
        self.query_hash = self._generate_query_hash()

    def _generate_query_hash(self) -> str:
        """Generate a hash for the query for caching purposes"""
        query_data = f"{self.query}_{self.query_type.value}_{self.top_k}_{self.similarity_threshold}"
        if self.filters:
            # Sort filters for consistent hashing
            sorted_filters = sorted(self.filters.items())
            query_data += f"_{sorted_filters}"
        if self.document_ids:
            # Sort document IDs for consistent hashing
            sorted_doc_ids = sorted(self.document_ids)
            query_data += f"_{sorted_doc_ids}"
        return hashlib.sha256(query_data.encode()).hexdigest()[:16]


@dataclass
class RetrievalResponse:
    """Complete retrieval response"""

    success: bool
    query: str
    query_type: QueryType
    results: List[RetrievalResult]
    total_results: int
    processing_time: float
    cache_hit: bool
    filters_applied: Dict[str, Any]
    query_hash: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None

    @property
    def has_results(self) -> bool:
        """Check if response has any results"""
        return len(self.results) > 0

    @property
    def top_result(self) -> Optional[RetrievalResult]:
        """Get the top-ranked result"""
        return self.results[0] if self.results else None


@dataclass
class GenerationRequest:
    """Request for text generation"""

    augmented_prompt: str
    generation_type: str = "answer"  # "answer", "summary", "explanation"
    model_provider: Optional[str] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing"""
        if self.conversation_history is None:
            self.conversation_history = []

        # Generate request hash for caching
        self.request_hash = self._generate_request_hash()
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    citation_style: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing"""
        if self.conversation_history is None:
            self.conversation_history = []

        # Generate request hash for caching
        self.request_hash = self._generate_request_hash()

    def _generate_request_hash(self) -> str:
        """Generate a hash for the request for caching purposes"""
        request_data = f"{self.augmented_prompt}_{self.generation_type}_{self.temperature or 0.1}_{self.max_tokens or 2000}"
        if self.conversation_history:
            # Include last few messages in hash
            history_str = "_".join([f"{m.get('role', '')}:{m.get('content', '')}" for m in self.conversation_history[-3:]])
            request_data += f"_{history_str}"
        return hashlib.sha256(request_data.encode()).hexdigest()[:16]


@dataclass
class GenerationResult:
    """Result from generation processing"""

    success: bool
    answer: str
    citations_used: List[Citation]
    model_used: str
    processing_time: float
    token_count: Optional[int] = None
    confidence_score: Optional[float] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None

    @property
    def has_answer(self) -> bool:
        """Check if a valid answer was generated"""
        return (
            self.success
            and self.answer is not None
            and len(self.answer.strip()) > 0
            and not self.answer.lower().startswith("error")
            and not self.answer.lower().startswith("sorry")
        )


@dataclass
class IngestionResult:
    """Result from document ingestion"""

    success: bool
    document_id: str
    filename: str
    chunks_created: int
    pages: int
    citation_mode: str
    processing_time: float
    message: str
    error_message: Optional[str] = None
    steps_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AugmentationConfigModel:
    """Per-request augmentation config overrides"""

    max_context_tokens: int = 1500
    min_score_threshold: float = 0.3
    top_k: int = 5
    dedup_similarity: float = 0.85
    citation_style: str = "page"


@dataclass
class AugmentedContextItem:
    """Normalized context item used for prompt assembly"""

    chunk_id: str
    document_id: str
    text: str
    citation: str
    score: float


@dataclass
class QueryExpansion:
    """Query expansion configuration and results"""
    
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    expansion_enabled: bool = True
    expansion_method: str = "semantic"  # "semantic", "synonym", "conceptual"
    intent: Optional[str] = None
    confidence: float = 0.0
    
    @property
    def expanded_terms(self) -> List[str]:
        """Get expanded terms for backward compatibility"""
        return self.expanded_queries[1:] if len(self.expanded_queries) > 1 else []
    
    @property
    def expanded_query(self) -> str:
        """Get the expanded query string"""
        if not self.expansion_enabled or not self.expanded_queries:
            return self.original_query
        
        # Combine original query with expanded queries
        return " ".join(self.expanded_queries)


@dataclass
class AugmentationResult:
    """Result of augmentation stage"""

    augmented_prompt: str
    citations: List[str]
    context_items: List[AugmentedContextItem]
    context_metadata: Dict[str, Any]
