"""
Unified Service Manager for RAG Microservice
Handles initialization and lifecycle of all shared services
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time
import json

from src.core.config import config

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages initialization and lifecycle of all shared services"""

    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.initialized = False
        self.startup_time = None

    async def initialize_all_services(self) -> bool:
        """Initialize all shared services in the correct order"""
        try:
            logger.info("🚀 Starting unified RAG service initialization...")
            start_time = time.time()

            # Step 1: Setup directories
            await self._setup_directories()

            # Step 2: Initialize logging
            await self._setup_logging()

            # Step 3: Initialize databases
            await self._initialize_databases()

            # Step 4: Initialize cache manager
            await self._initialize_cache()

            # Step 5: Initialize LLM client
            await self._initialize_llm_client()

            # Step 6: Validate all services
            await self._validate_services()

            self.initialized = True
            self.startup_time = time.time() - start_time

            logger.info(f"✅ All shared services initialized successfully in {self.startup_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {str(e)}")
            await self._cleanup_services()
            return False

    async def _setup_directories(self):
        """Create necessary directories"""
        logger.info("📁 Setting up directories...")

        # Get the project root (where main.py is located)
        project_root = Path(__file__).parent.parent.parent

        directories = [
            project_root / config.database.chroma_db_path,
            project_root / os.path.dirname(config.database.sqlite_db_path),
            project_root / config.cache.embedding_cache_path,
            project_root / config.cache.response_cache_path,
            project_root / config.cache.temp_cache_path,
            project_root / "data" / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

        logger.info("✅ Directories setup completed")

    async def _setup_logging(self):
        """Setup enhanced logging configuration with multiple handlers"""
        try:
            logger.info("📝 Setting up enhanced logging...")

            # Get the project root (where main.py is located)
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "data" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Update config paths to be absolute for logging
            config.logging.log_file = str(log_dir / "rag.log")
            config.logging.error_log_file = str(log_dir / "error.log")

            # Clear any existing handlers to avoid duplicates
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            # Set log level
            log_level = getattr(logging, config.logging.log_level.upper(), logging.INFO)
            root_logger.setLevel(log_level)

            # Create formatters
            formatter = logging.Formatter(config.logging.log_format)
            if config.logging.log_json_format:
                formatter = logging.Formatter(
                    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
                )

            # Main log file handler
            file_handler = logging.FileHandler(config.logging.log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # Error log file handler (only errors and above)
            error_handler = logging.FileHandler(config.logging.error_log_file)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

            # Console handler for development
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

            # Set specific logger levels for different components
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("torch").setLevel(logging.WARNING)

            # Enable debug logging for our components if configured
            if config.logging.enable_performance_logging:
                logging.getLogger("src.processors.retrieval").setLevel(logging.DEBUG)
                logging.getLogger("src.processors.generation").setLevel(logging.DEBUG)
                logging.getLogger("src.api").setLevel(logging.DEBUG)

            logger.info("✅ Enhanced logging setup completed with multiple handlers")
            return True

        except Exception as e:
            print(f"❌ Failed to setup enhanced logging: {str(e)}")
            return False

    async def _initialize_databases(self):
        """Initialize database connections"""
        logger.info("🗄️ Initializing databases...")

        # Initialize ChromaDB
        try:
            from src.storage.vector_store import ChromaDatabase
            chroma_db = ChromaDatabase(persist_directory=config.database.chroma_db_path)
            self.services['chroma_db'] = chroma_db
            logger.info("✅ ChromaDB initialized")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {str(e)}")
            raise

        # Initialize SQLite
        try:
            from src.storage.metadata_store import SQLiteStorage
            sqlite_storage = SQLiteStorage(db_path=config.database.sqlite_db_path)
            self.services['sqlite_storage'] = sqlite_storage
            logger.info("✅ SQLite storage initialized")
        except Exception as e:
            logger.error(f"❌ SQLite initialization failed: {str(e)}")
            raise

    async def _initialize_cache(self):
        """Initialize cache manager"""
        logger.info("💾 Initializing cache manager...")

        try:
            from src.storage.cache_manager import CacheManager
            cache_manager = CacheManager()
            self.services['cache_manager'] = cache_manager
            logger.info("✅ Cache manager initialized")
        except Exception as e:
            logger.error(f"❌ Cache manager initialization failed: {str(e)}")
            raise

    async def _initialize_llm_client(self):
        """Initialize LLM client"""
        logger.info("🤖 Initializing LLM client...")

        try:
            from src.utils.llm_client import LLMClient, RateLimitConfig
            rate_config = RateLimitConfig(
                base_delay=1.5,
                max_delay=60.0,
                max_retries=3,
                max_concurrent_requests=config.generation.max_concurrent_requests
            )

            llm_client = LLMClient(config_rate=rate_config)
            self.services['llm_client'] = llm_client
            logger.info("✅ LLM client initialized")
        except Exception as e:
            logger.error(f"❌ LLM client initialization failed: {str(e)}")
            raise

    async def _validate_services(self):
        """Validate all services are working correctly with enhanced checks"""
        logger.info("🔍 Validating services...")

        validation_results = {
            'chroma_db': False,
            'sqlite_storage': False,
            'cache_manager': False,
            'llm_client': False,
            'semantic_search': False
        }

        # Test ChromaDB
        try:
            chroma_db = self.services['chroma_db']
            collection_info = chroma_db.collection.count()
            # Test collection access
            chroma_db.collection.get(limit=1)
            validation_results['chroma_db'] = True
            logger.debug(f"ChromaDB collection has {collection_info} documents - validation passed")
        except Exception as e:
            logger.warning(f"ChromaDB validation failed: {str(e)}")

        # Test SQLite
        try:
            sqlite_storage = self.services['sqlite_storage']
            stats = sqlite_storage.get_document_stats()
            # Test database access
            sqlite_storage.list_documents(limit=1)
            validation_results['sqlite_storage'] = True
            logger.debug(f"SQLite has {stats.get('total_documents', 0)} documents - validation passed")
        except Exception as e:
            logger.warning(f"SQLite validation failed: {str(e)}")

        # Test Cache Manager
        try:
            cache_manager = self.services['cache_manager']
            # Test cache operations
            test_key = f"validation_test_{int(time.time())}"
            cache_manager.set(test_key, "test_value", ttl_hours=0.001)  # Very short TTL
            cache_manager.get(test_key)
            validation_results['cache_manager'] = True
            logger.debug("Cache manager validation passed")
        except Exception as e:
            logger.warning(f"Cache manager validation failed: {str(e)}")

        # Test LLM client
        try:
            llm_client = self.services['llm_client']
            # Check if LLM client has required attributes and configuration
            required_attrs = ['model', 'generate_text', 'circuit_breaker', 'request_queue']
            has_required_attrs = all(hasattr(llm_client, attr) for attr in required_attrs)

            # Check if API key is configured (allow test key for development)
            api_key = getattr(config.model, 'groq_api_key', None)
            api_key_configured = bool(api_key and api_key != "test_key_for_development")
            test_key_used = api_key == "test_key_for_development"

            if has_required_attrs and (api_key_configured or test_key_used):
                validation_results['llm_client'] = True
                if test_key_used:
                    logger.info("LLM client validation passed (using test key for development)")
                else:
                    logger.debug("LLM client validation passed")
            else:
                missing_attrs = [attr for attr in required_attrs if not hasattr(llm_client, attr)]
                if missing_attrs:
                    logger.warning(f"LLM client missing required attributes: {missing_attrs}")
                if not api_key_configured and not test_key_used:
                    logger.warning("LLM client API key not configured")
                elif test_key_used:
                    # This shouldn't happen if has_required_attrs is True, but just in case
                    logger.info("LLM client using test key but validation logic issue")
        except Exception as e:
            logger.warning(f"LLM client validation failed: {str(e)}")

        # Test Semantic Search (optional)
        try:
            from src.processors.retrieval.semantic_retriever import SemanticRetriever
            retriever = SemanticRetriever()
            if retriever.embedding_model is not None:
                validation_results['semantic_search'] = True
                logger.debug("Semantic search validation passed")
            else:
                logger.warning("Semantic search model not loaded")
        except Exception as e:
            logger.warning(f"Semantic search validation failed: {str(e)}")

        # Log validation summary
        passed_count = sum(validation_results.values())
        total_count = len(validation_results)
        logger.info(f"✅ Service validation completed: {passed_count}/{total_count} services validated")

        if passed_count < total_count:
            failed_services = [service for service, passed in validation_results.items() if not passed]
            if failed_services == ['llm_client'] and getattr(config.model, 'groq_api_key', None) == "test_key_for_development":
                logger.info(f"ℹ️  Using test API key for development. Set GROQ_API_KEY for production.")
            else:
                logger.warning(f"⚠️  Some services failed validation. System may have reduced functionality.")
                for service in failed_services:
                    logger.warning(f"   - {service}: FAILED")
        else:
            logger.info("🎉 All services validated successfully")

    async def _cleanup_services(self):
        """Cleanup services on failure"""
        logger.info("🧹 Cleaning up services...")

        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'close'):
                    service.close()
                elif hasattr(service, 'cleanup'):
                    service.cleanup()
                logger.debug(f"Cleaned up service: {service_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup service {service_name}: {str(e)}")

        self.services.clear()
        logger.info("✅ Service cleanup completed")

    def get_service(self, service_name: str) -> Any:
        """Get a service by name"""
        if not self.initialized:
            raise RuntimeError("Services not initialized. Call initialize_all_services() first.")

        if service_name not in self.services:
            raise KeyError(f"Service '{service_name}' not found")

        return self.services[service_name]

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all services"""
        if not self.initialized:
            return {
                "status": "not_initialized",
                "services": {},
                "startup_time": None
            }

        service_status = {}
        for service_name, service in self.services.items():
            try:
                # Basic health check for each service
                if service_name == 'chroma_db':
                    service_status[service_name] = "healthy" if service.collection else "unhealthy"
                elif service_name == 'sqlite_storage':
                    service_status[service_name] = "healthy" if service.db_path else "unhealthy"
                elif service_name == 'llm_client':
                    service_status[service_name] = "healthy" if service else "unhealthy"
                elif service_name == 'cache_manager':
                    service_status[service_name] = "healthy" if service else "unhealthy"
                else:
                    service_status[service_name] = "healthy" if service else "unhealthy"
            except Exception:
                service_status[service_name] = "unhealthy"

        overall_status = "healthy" if all(status == "healthy" for status in service_status.values()) else "degraded"

        return {
            "status": overall_status,
            "services": service_status,
            "startup_time": self.startup_time,
            "initialized": self.initialized
        }

    async def shutdown(self):
        """Shutdown all services"""
        logger.info("🛑 Shutting down unified RAG service...")
        await self._cleanup_services()

    def log_structured(self, level: str, message: str, **kwargs):
        """Log structured data with additional context"""
        try:
            if config.logging.enable_structured_logging:
                structured_data = {
                    "message": message,
                    "timestamp": time.time(),
                    "level": level.upper(),
                    "logger": __name__,
                    **kwargs
                }

                if config.logging.log_json_format:
                    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(structured_data))
                else:
                    # Log in structured format but human readable
                    extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
                    logger.log(getattr(logging, level.upper(), logging.INFO), f"{message} | {extra_info}")
            else:
                logger.log(getattr(logging, level.upper(), logging.INFO), message)

        except Exception as e:
            logger.warning(f"Failed to log structured data: {str(e)}")
            logger.log(getattr(logging, level.upper(), logging.INFO), message)


# Global service manager instance
service_manager = ServiceManager()
