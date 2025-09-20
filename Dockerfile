# Unified RAG Microservice - Production-Optimized Multi-stage Dockerfile
# Optimized for performance, security, and resource efficiency

# =============================================================================
# BUILD STAGE - Optimized for dependency compilation
# =============================================================================
FROM python:3.11-slim as builder

# Build arguments for optimization
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG PYTHON_VERSION=3.11

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=3 \
    PYTHONHASHSEED=random

# Install system dependencies for building with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y

# Create virtual environment with optimized settings
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies with optimization
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --compile \
    --global-option="-j$(nproc)" \
    -r requirements.txt

# =============================================================================
# RUNTIME STAGE - Minimal production image
# =============================================================================
FROM python:3.11-slim as production

# Build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Labels for metadata
LABEL maintainer="BookMate RAG Team" \
      version="${VERSION}" \
      description="Unified RAG Microservice for document processing and AI-powered search" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONHASHSEED=random \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Install minimal runtime dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set work directory
WORKDIR /app

# Create non-root user with specific UID/GID for security
RUN groupadd -r -g 1001 appuser && \
    useradd -r -g appuser -u 1001 -s /bin/bash appuser

# Create necessary directories with proper permissions and ownership
RUN mkdir -p /app/data/chroma_db \
    /app/data/sqlite \
    /app/data/cache/embeddings \
    /app/data/cache/responses \
    /app/data/cache/temp \
    /app/data/logs \
    /app/tmp \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app/data \
    && chmod -R 700 /app/tmp

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Set proper permissions for executable files
RUN chmod +x main.py && \
    chmod 644 requirements.txt && \
    chmod -R 755 /app/src

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with comprehensive monitoring
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create optimized startup script with error handling
RUN echo '#!/bin/bash\n\
set -euo pipefail\n\
\n\
# Function to log with timestamp\n\
log() {\n\
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*"\n\
}\n\
\n\
# Function to cleanup on exit\n\
cleanup() {\n\
    log "🛑 Service stopped, cleaning up..."\n\
    # Clean up temporary files\n\
    find /app/tmp -type f -name "*.tmp" -delete 2>/dev/null || true\n\
    log "✅ Cleanup completed"\n\
}\n\
\n\
# Set trap for cleanup\n\
trap cleanup EXIT INT TERM\n\
\n\
# Pre-flight checks\n\
log "🔍 Running pre-flight checks..."\n\
\n\
# Check if required directories exist and are writable\n\
for dir in /app/data/chroma_db /app/data/sqlite /app/data/cache/embeddings /app/data/cache/responses /app/data/logs; do\n\
    if [ ! -d "$dir" ]; then\n\
        log "❌ Directory $dir does not exist"\n\
        exit 1\n\
    fi\n\
    if [ ! -w "$dir" ]; then\n\
        log "❌ Directory $dir is not writable"\n\
        exit 1\n\
    fi\n\
done\n\
\n\
# Check Python environment\n\
if ! python -c "import sys; print(f\"Python {sys.version}\")" >/dev/null 2>&1; then\n\
    log "❌ Python environment check failed"\n\
    exit 1\n\
fi\n\
\n\
# Check critical dependencies\n\
python -c "import fastapi, uvicorn, chromadb" >/dev/null 2>&1 || {\n\
    log "❌ Critical dependencies missing"\n\
    exit 1\n\
}\n\
\n\
log "✅ Pre-flight checks passed"\n\
log "🚀 Starting Unified RAG Microservice..."\n\
log "📁 Auto-initializing databases and directories..."\n\
log "🧠 Loading AI models and services..."\n\
log "🌐 Server will be available at http://localhost:8000"\n\
log "📊 Health check available at http://localhost:8000/health"\n\
log "📚 API documentation at http://localhost:8000/docs"\n\
\n\
# Start the application with proper signal handling\n\
exec python main.py' > /app/start.sh && chmod +x /app/start.sh

# Use exec form for better signal handling and PID 1 behavior
CMD ["/app/start.sh"]
