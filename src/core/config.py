"""
Unified Configuration Management for RAG Microservice
Combines ingestion, retrieval, and generation configurations
"""

import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will rely on system env vars
    pass

# Setup logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for AI models and APIs"""

    # GROQ API Configuration
    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 4000
    groq_timeout: int = 30

    # Alternative LLM APIs (for future use)
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"

    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-haiku-20240307"
    
    # Embedding model configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384


@dataclass
class DatabaseConfig:
    """Configuration for databases and storage"""

    # ChromaDB Configuration
    chroma_db_path: str = "../data/chroma_db"
    chroma_collection_name: str = "rag_documents"
    chroma_allow_reset: bool = True
    chroma_anonymized_telemetry: bool = False

    # SQLite Configuration
    sqlite_db_path: str = "../data/sqlite/rag.db"
    sqlite_timeout: int = 30

    # Redis Configuration (optional)
    redis_url: Optional[str] = None
    redis_password: Optional[str] = None
    redis_db: int = 0


@dataclass
class IngestionConfig:
    """Configuration for document ingestion"""

    # Document Processing
    max_document_size_mb: int = 10
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf", ".html", ".txt", ".doc", ".docx", ".md", ".rtf"])

    # Chunking Configuration
    chunk_size: int = 300
    chunk_overlap: int = 40
    min_chunk_size: int = 100
    max_chunk_size: int = 500

    # Summarization Configuration
    max_document_summary_lines: int = 15
    min_chunk_summary_length: int = 50
    max_chunk_summary_length: int = 200
    batch_size: int = 5
    batch_summary_sentences: int = 6


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""

    # Search configuration
    enable_semantic_search: bool = True
    enable_keyword_search: bool = True
    enable_hybrid_search: bool = True
    enable_query_expansion: bool = True

    # Ranking configuration
    top_k_retrieval: int = 5
    top_k_max: int = 20
    similarity_threshold: float = 0.2
    hybrid_weight: float = 0.7

    # Advanced retrieval settings
    enable_reranking: bool = True
    rerank_top_k: int = 10

    # Freshness boosting settings
    freshness_boost: float = 0.0
    freshness_decay_days: int = 30

    # Section and content boosting
    section_boost: float = 0.0
    keyword_boost: float = 0.0
    author_boost: float = 0.0
    document_type_boost: float = 0.0

    # Additional boosting settings
    title_boost: float = 0.0
    content_boost: float = 0.0
    metadata_boost: float = 0.0

    # Performance settings
    max_query_length: int = 1000
    min_query_length: int = 3
    batch_size: int = 100


@dataclass
class GenerationConfig:
    """Configuration for text generation"""

    # Generation settings
    max_input_length: int = 16000
    max_output_tokens: int = 2000
    temperature: float = 0.1
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Citation settings
    citation_style: str = "page"
    inline_citation_format: str = "[CIT:{chunk_id}]"
    include_citation_list: bool = True

    # Performance settings
    max_concurrent_requests: int = 5


@dataclass
class CacheConfig:
    """Configuration for caching systems"""

    # Cache TTL Settings
    cache_ttl_hours: int = 24
    embedding_cache_ttl_hours: int = 168  # 7 days
    response_cache_ttl_hours: int = 24
    summary_cache_ttl_hours: int = 168    # 7 days

    # Cache Paths
    embedding_cache_path: str = "./data/cache/embeddings"
    response_cache_path: str = "./data/cache/responses"
    responses_cache_path: str = "./data/cache/responses"  # Alternative name for compatibility
    temp_cache_path: str = "./data/cache/temp"

    # Cache Settings
    max_cache_size_mb: int = 1000
    cache_cleanup_interval_hours: int = 24


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""

    log_level: str = "INFO"
    log_file: str = "./data/logs/rag.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_max_size_mb: int = 100
    log_backup_count: int = 5

    # Separate log files for different components
    api_log_file: str = "./data/logs/api.log"
    retrieval_log_file: str = "./data/logs/retrieval.log"
    generation_log_file: str = "./data/logs/generation.log"
    ingestion_log_file: str = "./data/logs/ingestion.log"
    error_log_file: str = "./data/logs/error.log"

    # Performance Monitoring
    enable_performance_logging: bool = True
    log_slow_operations_threshold_ms: int = 1000

    # Structured Logging
    enable_structured_logging: bool = True
    log_json_format: bool = False


@dataclass
class ServiceConfig:
    """Configuration for service settings"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    # Health Check
    health_check_interval: int = 30
    health_check_timeout: int = 10

    # Rate Limiting
    rate_limit_requests_per_minute: int = 120
    rate_limit_burst: int = 20

    # Security
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_header: Optional[str] = None


@dataclass
class Config:
    """Main configuration class combining all sub-configurations"""

    # Sub-configurations
    model: ModelConfig
    database: DatabaseConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    cache: CacheConfig
    logging: LoggingConfig
    service: ServiceConfig

    # Environment
    environment: str = "development"
    debug: bool = False

    def __post_init__(self):
        """Post-initialization processing for path resolution"""
        self._resolve_paths()

    def _resolve_paths(self):
        """Convert relative paths to absolute paths with proper directory creation"""
        config_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(config_file_dir)

        # Resolve database paths
        self.database.chroma_db_path = self._resolve_path(self.database.chroma_db_path, project_root)
        self.database.sqlite_db_path = self._resolve_path(self.database.sqlite_db_path, project_root)

        # Resolve cache paths - ensure they exist
        self.cache.embedding_cache_path = self._ensure_cache_path(self.cache.embedding_cache_path, project_root)
        self.cache.response_cache_path = self._ensure_cache_path(self.cache.response_cache_path, project_root)
        self.cache.temp_cache_path = self._ensure_cache_path(self.cache.temp_cache_path, project_root)

        # Resolve log path
        self.logging.log_file = self._resolve_path(self.logging.log_file, project_root)

    def _resolve_path(self, path: str, project_root: str) -> str:
        """Resolve a path to absolute if it's relative"""
        if os.path.isabs(path):
            return path
        return os.path.join(project_root, path.lstrip("./").lstrip("../"))

    def _ensure_cache_path(self, path: str, project_root: str) -> str:
        """Resolve cache path and ensure directory exists"""
        resolved_path = self._resolve_path(path, project_root)

        try:
            cache_path = Path(resolved_path)
            cache_path.mkdir(parents=True, exist_ok=True)

            # Verify the directory was created and is writable
            if not cache_path.exists():
                raise Exception(f"Failed to create cache directory: {cache_path}")
            if not cache_path.is_dir():
                raise Exception(f"Cache path exists but is not a directory: {cache_path}")

            # Test write permissions by creating a temporary test file
            test_file = cache_path / ".cache_test.tmp"
            test_file.write_text("test")
            test_file.unlink()

            logger.debug(f"Cache directory verified and writable: {cache_path}")
            return str(cache_path)

        except Exception as e:
            logger.error(f"Failed to ensure cache path {resolved_path}: {e}")
            # Fallback to system temp directory
            import tempfile
            fallback_path = Path(tempfile.gettempdir()) / "rag_cache" / Path(resolved_path).name
            fallback_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using fallback cache path: {fallback_path}")
            return str(fallback_path)


def load_config() -> Config:
    """Load comprehensive configuration from environment variables"""

    # Determine project root directory
    current_file = os.path.abspath(__file__)
    app_dir = os.path.dirname(current_file)  # src/core
    project_root = os.path.dirname(os.path.dirname(app_dir))  # BookMate directory

    # Model Configuration
    model_config = ModelConfig(
        groq_api_key=os.getenv("GROQ_API_KEY", "test_key_for_development"),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_temperature=float(os.getenv("GROQ_TEMPERATURE", "0.1")),
        groq_max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "4000")),
        groq_timeout=int(os.getenv("GROQ_TIMEOUT", "30")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    )

    # Database Configuration - Use absolute paths to avoid /app issues
    database_config = DatabaseConfig(
        chroma_db_path=os.getenv("CHROMA_DB_PATH", os.path.join(project_root, "data", "chroma_db")),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
        chroma_allow_reset=os.getenv("CHROMA_ALLOW_RESET", "true").lower() == "true",
        chroma_anonymized_telemetry=os.getenv("CHROMA_ANONYMIZED_TELEMETRY", "false").lower() == "true",
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", os.path.join(project_root, "data", "sqlite", "rag.db")),
        sqlite_timeout=int(os.getenv("SQLITE_TIMEOUT", "30")),
        redis_url=os.getenv("REDIS_URL"),
        redis_password=os.getenv("REDIS_PASSWORD"),
        redis_db=int(os.getenv("REDIS_DB", "0"))
    )

    # Ingestion Configuration
    allowed_extensions = [".pdf", ".html", ".txt", ".doc", ".docx", ".md", ".rtf"]
    extensions_str = os.getenv("ALLOWED_EXTENSIONS")
    if extensions_str:
        allowed_extensions = [ext.strip() for ext in extensions_str.split(",")]

    ingestion_config = IngestionConfig(
        max_document_size_mb=int(os.getenv("MAX_DOCUMENT_SIZE_MB", "10")),
        allowed_extensions=allowed_extensions,
        chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
        min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
        max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "500")),
        max_document_summary_lines=int(os.getenv("MAX_DOCUMENT_SUMMARY_LINES", "15")),
        min_chunk_summary_length=int(os.getenv("MIN_CHUNK_SUMMARY_LENGTH", "50")),
        max_chunk_summary_length=int(os.getenv("MAX_CHUNK_SUMMARY_LENGTH", "200")),
        batch_size=int(os.getenv("INGESTION_BATCH_SIZE", "5")),
        batch_summary_sentences=int(os.getenv("BATCH_SUMMARY_SENTENCES", "6"))
    )

    # Retrieval Configuration
    retrieval_config = RetrievalConfig(
        enable_semantic_search=os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() == "true",
        enable_keyword_search=os.getenv("ENABLE_KEYWORD_SEARCH", "true").lower() == "true",
        enable_hybrid_search=os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true",
        enable_query_expansion=os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true",
        top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "5")),
        top_k_max=int(os.getenv("TOP_K_MAX", "20")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.2")),
        hybrid_weight=float(os.getenv("HYBRID_WEIGHT", "0.7")),
        enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "10")),
        max_query_length=int(os.getenv("MAX_QUERY_LENGTH", "1000")),
        min_query_length=int(os.getenv("MIN_QUERY_LENGTH", "3")),
        batch_size=int(os.getenv("RETRIEVAL_BATCH_SIZE", "100"))
    )

    # Generation Configuration
    generation_config = GenerationConfig(
        max_input_length=int(os.getenv("MAX_INPUT_LENGTH", "16000")),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2000")),
        temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.1")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        presence_penalty=float(os.getenv("PRESENCE_PENALTY", "0.0")),
        frequency_penalty=float(os.getenv("FREQUENCY_PENALTY", "0.0")),
        citation_style=os.getenv("CITATION_STYLE", "page"),
        inline_citation_format=os.getenv("INLINE_CITATION_FORMAT", "[CIT:{chunk_id}]"),
        include_citation_list=os.getenv("INCLUDE_CITATION_LIST", "true").lower() == "true",
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    )

    # Cache Configuration
    cache_config = CacheConfig(
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        embedding_cache_ttl_hours=int(os.getenv("EMBEDDING_CACHE_TTL_HOURS", "168")),
        response_cache_ttl_hours=int(os.getenv("RESPONSE_CACHE_TTL_HOURS", "24")),
        summary_cache_ttl_hours=int(os.getenv("SUMMARY_CACHE_TTL_HOURS", "168")),
        embedding_cache_path=os.getenv("EMBEDDING_CACHE_PATH", os.path.join(project_root, "data", "cache", "embeddings")),
        response_cache_path=os.getenv("RESPONSE_CACHE_PATH", os.path.join(project_root, "data", "cache", "responses")),
        temp_cache_path=os.getenv("TEMP_CACHE_PATH", os.path.join(project_root, "data", "cache", "temp")),
        max_cache_size_mb=int(os.getenv("MAX_CACHE_SIZE_MB", "1000")),
        cache_cleanup_interval_hours=int(os.getenv("CACHE_CLEANUP_INTERVAL_HOURS", "24"))
    )

    # Logging Configuration
    logging_config = LoggingConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", os.path.join(project_root, "data", "logs", "rag.log")),
        log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        log_max_size_mb=int(os.getenv("LOG_MAX_SIZE_MB", "100")),
        log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        enable_performance_logging=os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true",
        log_slow_operations_threshold_ms=int(os.getenv("LOG_SLOW_OPERATIONS_THRESHOLD_MS", "1000"))
    )

    # Service Configuration
    cors_origins = ["*"]
    cors_origins_str = os.getenv("CORS_ORIGINS")
    if cors_origins_str:
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

    service_config = ServiceConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=int(os.getenv("WORKERS", "1")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
        health_check_timeout=int(os.getenv("HEALTH_CHECK_TIMEOUT", "10")),
        rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120")),
        rate_limit_burst=int(os.getenv("RATE_LIMIT_BURST", "20")),
        cors_origins=cors_origins,
        api_key_header=os.getenv("API_KEY_HEADER")
    )

    # Create main config
    config = Config(
        model=model_config,
        database=database_config,
        ingestion=ingestion_config,
        retrieval=retrieval_config,
        generation=generation_config,
        cache=cache_config,
        logging=logging_config,
        service=service_config,
        environment=os.getenv("ENVIRONMENT", "development"),
        debug=os.getenv("DEBUG", "false").lower() == "true"
    )

    # Validate critical configurations
    _validate_config(config)

    return config


def _validate_config(config: Config) -> None:
    """Validate critical configuration values"""
    if not config.model.groq_api_key or config.model.groq_api_key == "test_key_for_development":
        logging.warning("⚠️  WARNING: Using test API key. Set GROQ_API_KEY environment variable for production use.")

    if config.ingestion.chunk_size <= config.ingestion.chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    if config.ingestion.min_chunk_size >= config.ingestion.max_chunk_size:
        raise ValueError("min_chunk_size must be less than max_chunk_size")

    if config.retrieval.similarity_threshold < 0.0 or config.retrieval.similarity_threshold > 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    if config.generation.temperature < 0.0 or config.generation.temperature > 2.0:
        raise ValueError("generation temperature must be between 0.0 and 2.0")


# Global config instance
config = load_config()
