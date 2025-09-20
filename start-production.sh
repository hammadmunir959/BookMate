#!/bin/bash

# Unified RAG Microservice - Production Startup Script
# Optimized for Docker and host system deployment

set -euo pipefail

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_PREFIX="[$(date +'%Y-%m-%d %H:%M:%S')]"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Function to log with timestamp and color
log() {
    local level="${1:-INFO}"
    local message="${2:-}"
    local color="${NC}"
    
    case "$level" in
        "ERROR") color="${RED}" ;;
        "WARN")  color="${YELLOW}" ;;
        "INFO")  color="${GREEN}" ;;
        "DEBUG") color="${BLUE}" ;;
        "SUCCESS") color="${GREEN}" ;;
        "HEADER") color="${PURPLE}" ;;
        "STEP") color="${CYAN}" ;;
    esac
    
    echo -e "${color}${LOG_PREFIX} [${level}] ${message}${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if we're in a container
is_container() {
    [ -f /.dockerenv ] || [ -n "${CONTAINER:-}" ]
}

# Function to check system requirements
check_system_requirements() {
    log "STEP" "Checking system requirements..."
    
    # Check Python installation
    if ! command_exists python3; then
        log "ERROR" "Python 3 is not installed"
        return 1
    fi
    
    # Check Python version
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "INFO" "Python version: $python_version"
    
    if [ "$(echo "$python_version < 3.8" | bc -l 2>/dev/null || echo "1")" -eq 1 ]; then
        log "ERROR" "Python 3.8 or higher is required"
        return 1
    fi
    
    # Check available memory
    if command_exists free; then
        local available_memory
        available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [ "$available_memory" -lt 1024 ]; then
            log "WARN" "Available memory is low: ${available_memory}MB (recommended: 1GB+)"
        else
            log "INFO" "Available memory: ${available_memory}MB"
        fi
    fi
    
    # Check disk space
    if command_exists df; then
        local available_space
        available_space=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
        if [ "$available_space" -lt 5 ]; then
            log "WARN" "Available disk space is low: ${available_space}GB (recommended: 5GB+)"
        else
            log "INFO" "Available disk space: ${available_space}GB"
        fi
    fi
    
    log "SUCCESS" "System requirements check passed"
    return 0
}

# Function to setup Python environment
setup_python_environment() {
    log "STEP" "Setting up Python environment..."
    
    if is_container; then
        log "INFO" "Running in container, using system Python"
        return 0
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        log "INFO" "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    log "INFO" "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    log "INFO" "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        log "INFO" "Installing dependencies..."
        pip install -r requirements.txt
    else
        log "ERROR" "requirements.txt not found"
        return 1
    fi
    
    log "SUCCESS" "Python environment setup completed"
    return 0
}

# Function to setup directories
setup_directories() {
    log "STEP" "Setting up directories..."
    
    local directories=(
        "data/chroma_db"
        "data/sqlite"
        "data/cache/embeddings"
        "data/cache/responses"
        "data/cache/temp"
        "data/logs"
        "tmp"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            log "INFO" "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Set proper permissions
    chmod -R 755 data/ 2>/dev/null || true
    chmod -R 700 tmp/ 2>/dev/null || true
    
    log "SUCCESS" "Directories setup completed"
    return 0
}

# Function to validate configuration
validate_configuration() {
    log "STEP" "Validating configuration..."
    
    # Check if main.py exists
    if [ ! -f "main.py" ]; then
        log "ERROR" "main.py not found"
        return 1
    fi
    
    # Check if src directory exists
    if [ ! -d "src" ]; then
        log "ERROR" "src directory not found"
        return 1
    fi
    
    # Check API key
    if [ -z "${GROQ_API_KEY:-}" ]; then
        log "WARN" "GROQ_API_KEY not set. Using test key for development."
        export GROQ_API_KEY="test_key_for_development"
    else
        log "INFO" "GROQ_API_KEY is configured"
    fi
    
    # Set environment variables
    export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/src"
    export PYTHONUNBUFFERED=1
    
    # Performance optimizations
    export MALLOC_ARENA_MAX=2
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    
    log "SUCCESS" "Configuration validation completed"
    return 0
}

# Function to run pre-flight checks
run_preflight_checks() {
    log "STEP" "Running pre-flight checks..."
    
    # Test Python imports
    if ! python3 -c "import fastapi, uvicorn, chromadb" >/dev/null 2>&1; then
        log "ERROR" "Critical dependencies missing"
        return 1
    fi
    
    # Test file permissions
    local test_file="tmp/.startup_test"
    if ! touch "$test_file" 2>/dev/null; then
        log "ERROR" "Cannot write to tmp directory"
        return 1
    fi
    rm -f "$test_file"
    
    # Test data directory permissions
    if ! touch "data/.startup_test" 2>/dev/null; then
        log "ERROR" "Cannot write to data directory"
        return 1
    fi
    rm -f "data/.startup_test"
    
    log "SUCCESS" "Pre-flight checks passed"
    return 0
}

# Function to cleanup on exit
cleanup() {
    log "INFO" "🛑 Service stopped, cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clean up temporary files
    find tmp/ -type f -name "*.tmp" -delete 2>/dev/null || true
    
    log "SUCCESS" "Cleanup completed"
}

# Function to display startup information
display_startup_info() {
    log "HEADER" "🚀 Starting Unified RAG Microservice"
    log "INFO" "Version: 1.0.0"
    log "INFO" "Environment: ${ENVIRONMENT:-production}"
    log "INFO" "Debug Mode: ${DEBUG:-false}"
    log "INFO" "Log Level: ${LOG_LEVEL:-INFO}"
    log "INFO" "Host: ${HOST:-0.0.0.0}"
    log "INFO" "Port: ${PORT:-8000}"
    log "INFO" "Workers: ${WORKERS:-1}"
    log "INFO" "Container: $(is_container && echo "Yes" || echo "No")"
    echo
    log "INFO" "🌐 Server will be available at http://localhost:${PORT:-8000}"
    log "INFO" "📊 Health check available at http://localhost:${PORT:-8000}/health"
    log "INFO" "📚 API documentation at http://localhost:${PORT:-8000}/docs"
    echo
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Set trap for cleanup
    trap cleanup EXIT INT TERM
    
    # Display header
    log "HEADER" "=========================================="
    log "HEADER" "  Unified RAG Microservice Startup"
    log "HEADER" "=========================================="
    echo
    
    # Run all setup steps
    check_system_requirements || exit 1
    setup_python_environment || exit 1
    setup_directories || exit 1
    validate_configuration || exit 1
    run_preflight_checks || exit 1
    
    # Display startup information
    display_startup_info
    
    # Start the application
    log "SUCCESS" "Starting RAG Microservice..."
    exec python3 main.py
}

# Run main function
main "$@"
