#!/usr/bin/env python3
"""
Unified RAG Microservice - Main Entry Point
Complete RAG system combining ingestion, retrieval, and generation
"""

import sys
import os
import signal
import asyncio
import logging
from pathlib import Path

from starlette.responses import JSONResponse
from contextlib import asynccontextmanager

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import services
from src.core.config import config
from src.core.service_manager import service_manager
from src.api.simplified_router import main_router

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager for modern FastAPI
@asynccontextmanager
async def lifespan(app: "FastAPI"):
    """Handle application startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Unified RAG Microservice...")

    try:
        # Initialize all services
        success = await service_manager.initialize_all_services()

        if success:
            logger.info("✅ All services initialized successfully")
        else:
            logger.error("❌ Service initialization failed")
            raise Exception("Service initialization failed")

    except Exception as e:
        logger.error(f"❌ Critical error during startup: {str(e)}")
        raise

    yield

    # Shutdown
    try:
        logger.info("🛑 Shutting down Unified RAG Microservice...")

        # Shutdown services
        await service_manager.shutdown()

        logger.info("✅ Shutdown complete")

    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}")


# Create FastAPI app with modern lifespan
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Unified RAG Microservice",
    description="Complete RAG system combining document ingestion, retrieval, and answer generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using wildcard origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

from fastapi import Request
import uuid

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add unique X-Request-ID to every request"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "BookMate RAG Agent - API",
        "version": "2.0.0",
        "description": " RAG system s",
        "endpoints": {
            "ingestion": "/ingestion - Upload and process documents",
            "query": "/query - Unified retrieval, augmentation, and generation",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        },
        "supported_formats": config.ingestion.allowed_extensions,
        "max_file_size_mb": config.ingestion.max_document_size_mb
    }

# Include main router
app.include_router(main_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for the unified RAG service"""
    try:
        health_status = service_manager.get_health_status()
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "timestamp": "unknown",
                "error": str(e)
            }
        )
 


def cleanup_on_exit():
    """Clean up resources on exit"""
    try:
        logger.info("🧹 Starting cleanup process...")

        # Clean up temporary files
        temp_dir = Path("/tmp")
        if temp_dir.exists():
            for temp_file in temp_dir.glob("*_rag_*.tmp"):
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")

        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
    cleanup_on_exit()
    sys.exit(0)


def main():
    """Main application entry point"""
    try:
        # Register cleanup functions
        import atexit
        atexit.register(cleanup_on_exit)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the FastAPI application (services will be initialized via startup event)
        import uvicorn
        logger.info(f"🌐 Starting web server on {config.service.host}:{config.service.port}")

        uvicorn.run(
            app,
            host=config.service.host,
            port=config.service.port,
            workers=config.service.workers,
            reload=config.service.reload,
            log_level=config.logging.log_level.lower(),
            access_log=True
        )

    except Exception as e:
        logger.error(f"❌ Critical error in main: {str(e)}")
        cleanup_on_exit()
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    main()
