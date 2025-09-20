# 🤖 BookMMate

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)](https://www.docker.com/)

 BookMate, A production-ready microservice that combines document ingestion, intelligent retrieval, and answer generation into a complete RAG (Retrieval-Augmented Generation) system. Built with FastAPI, ChromaDB, and GROQ's API.

## 🌟 Key Features

- **🔄 Complete RAG Pipeline**: Single service handling ingestion → retrieval → generation
- **📚 Multi-Format Support**: PDF, DOCX, TXT, HTML, MD, RTF documents
- **🔍 Hybrid Search**: Semantic + keyword search with intelligent ranking
- **📝 Citation-Aware Responses**: Automatic source citation with clickable references
- **⚡ High Performance**: In-memory data flow with advanced caching
- **🐳 Docker Ready**: Production-ready containerization
- **🔧 Flexible Configuration**: Environment-based configuration for all components
- **📊 Comprehensive Monitoring**: Health checks, metrics, and structured logging
- **🎨 Modern UI**: React-based frontend with Material-UI components

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified RAG Service                      │
│                    (Single Container)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐ │
│  │   Ingestion │ │ Retrieval   │ │   Generation          │ │
│  │   Pipeline  │ │   Pipeline  │ │   Pipeline            │ │
│  └─────────────┘ └─────────────┘ └───────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐ │
│  │ ChromaDB    │ │ SQLite      │ │ LLM Client (GROQ)    │ │
│  │ (Vectors)   │ │ (Metadata)  │ │                       │ │
│  └─────────────┘ └─────────────┘ └───────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────────┐ │
│  │   Shared    │ │   Config    │ │   Progress Tracker    │ │
│  │   Cache     │ │   Manager   │ │                       │ │
│  └─────────────┘ └─────────────┘ └───────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Usage Examples](#-usage-examples)
- [Deployment](#-deployment)
- [Development](#-development)
- [Testing](#-testing)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 16+ (for UI development)
- GROQ API key (get one at [groq.com](https://groq.com))

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-microservice

# Copy environment configuration
cp env.example .env

# Edit .env with your GROQ API key
nano .env
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For UI development (optional)
cd ui && npm install
```

### 3. Run the Service

```bash
# Start the backend service
python main.py

# In another terminal, start the UI (optional)
cd ui && npm run dev
```

### 4. Test End-to-End

```bash
# Upload a document
curl -X POST "http://localhost:8000/ingestion" \
  -F "file=@sample.pdf"

# Ask a question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "document_ids": ["sample.pdf"]
  }'
```

## 📦 Installation

### Option 1: Docker (Recommended)

```bash
# Build and run with Docker
docker build -t rag-microservice .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_api_key \
  -v ./data:/app/data \
  rag-microservice
```

### Option 2: Docker Compose

```bash
# Using docker-compose
docker-compose up --build
```

### Option 3: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Service Configuration
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO

# Database Configuration
CHROMA_DB_PATH=./data/chroma_db
SQLITE_DB_PATH=./data/sqlite/rag.db

# Document Processing
MAX_DOCUMENT_SIZE_MB=10
CHUNK_SIZE=300
CHUNK_OVERLAP=40

# Search Configuration
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.2
ENABLE_HYBRID_SEARCH=true

# Generation Configuration
MAX_INPUT_LENGTH=16000
GENERATION_TEMPERATURE=0.1
MAX_CONCURRENT_REQUESTS=5
```

### Configuration Sections

- **🔧 Model**: LLM provider settings (GROQ/OpenAI/Anthropic)
- **💾 Database**: Vector and metadata storage configuration
- **📄 Ingestion**: Document processing and chunking parameters
- **🔍 Retrieval**: Search algorithms and ranking settings
- **🤖 Generation**: LLM generation parameters and citation settings
- **💰 Cache**: Response and embedding caching configuration
- **🌐 Service**: Server, security, and API settings

## 📚 API Documentation

### Core Endpoints

#### Document Ingestion
```http
POST /ingestion
Content-Type: multipart/form-data

Parameters:
- file: Document file (PDF, DOCX, TXT, etc.)
- custom_metadata: Optional JSON metadata
- uploader_id: Optional user identifier
```

#### Query Processing
```http
POST /query
Content-Type: application/json

{
  "query": "What are the main findings?",
  "document_ids": ["document1.pdf", "document2.pdf"],
  "max_tokens": 2000,
  "temperature": 0.1
}
```

#### Document Management
```http
GET    /documents/list
GET    /documents/details/{document_id}
DELETE /documents/delete?document_id={document_id}
POST   /reset_session
```

#### System Endpoints
```http
GET  /health          # Service health status
GET  /stats           # System statistics
GET  /                # API information
GET  /docs            # Interactive API documentation
```

### Response Format

#### Ingestion Response
```json
{
  "success": true,
  "document_id": "research_paper.pdf",
  "filename": "research_paper.pdf",
  "file_size": 2457600,
  "total_chunks": 45,
  "processing_stats": {
    "chunks_created": 45,
    "processing_time": 2.34
  },
  "steps_status": [...]
}
```

#### Query Response
```json
{
  "success": true,
  "query": "What are the main findings?",
  "answer": "The main findings indicate that...",
  "citations": [
    {
      "chunk_id": "chunk_123",
      "document_id": "research_paper.pdf",
      "content": "According to the study...",
      "relevance_score": 0.87,
      "citation_text": "(cit#1)"
    }
  ],
  "processing_time": 1.23,
  "model_used": "llama-3.1-8b-instant"
}
```

## 💡 Usage Examples

### Python Client

```python
import requests

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def upload_document(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/ingestion", files=files)
        return response.json()

    def ask_question(self, query, document_ids=None):
        payload = {"query": query}
        if document_ids:
            payload["document_ids"] = document_ids

        response = requests.post(f"{self.base_url}/query", json=payload)
        return response.json()

# Usage
client = RAGClient()
result = client.upload_document("research_paper.pdf")
answer = client.ask_question("What are the key contributions?")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class RAGClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.client = axios.create({ baseURL });
    }

    async uploadDocument(filePath) {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));

        const response = await this.client.post('/ingestion', formData, {
            headers: formData.getHeaders()
        });
        return response.data;
    }

    async askQuestion(query, documentIds = null) {
        const payload = { query };
        if (documentIds) payload.document_ids = documentIds;

        const response = await this.client.post('/query', payload);
        return response.data;
    }
}

// Usage
const client = new RAGClient();
const result = await client.uploadDocument('research_paper.pdf');
const answer = await client.askQuestion('What are the main findings?');
```

### cURL Examples

```bash
# Upload document
curl -X POST "http://localhost:8000/ingestion" \
  -F "file=@research_paper.pdf"

# Ask question about specific documents
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "document_ids": ["research_paper.pdf"],
    "max_tokens": 1000
  }'

# Get system health
curl http://localhost:8000/health

# List all documents
curl http://localhost:8000/documents/list
```

## 🐳 Deployment

### Production Docker Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  rag-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/data/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Environment-Specific Configuration

```bash
# Production environment
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export MAX_DOCUMENT_SIZE_MB=50
export MAX_CONCURRENT_REQUESTS=10

# Development environment
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export RELOAD=true
```

### Scaling Considerations

- **Single Instance**: Suitable for small to medium applications
- **Horizontal Scaling**: Use load balancer with multiple instances
- **Database Scaling**: ChromaDB supports distributed deployment
- **Cache Scaling**: Consider Redis for distributed caching

## 💻 Development

### Project Structure

```
BookMate/
├── src/
│   ├── core/              # Configuration and service management
│   ├── api/               # FastAPI routers and endpoints
│   ├── processors/        # Ingestion, retrieval, generation logic
│   ├── storage/           # Database and cache managers
│   └── utils/             # Utilities and LLM client
├── ui/                    # React frontend
├── data/                  # Persistent data storage
├── tests/                 # Test suitess
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
└── docker-compose.yml   # Multi-container setup
```

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd BookMate

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Start development server with auto-reload
export RELOAD=true
python main.py
```


### Health Checks

```bash
# Service health
curl http://localhost:8000/health

# System statistics
curl http://localhost:8000/stats

# Database connectivity test
curl http://localhost:8000/test-retrieval
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** - Modern, fast web framework for building APIs
- **ChromaDB** - AI-native vector database
- **Sentence Transformers** - State-of-the-art text embeddings
- **GROQ** - Fast LLM inference platform
- **Material-UI** - React component library

## 📞 Support

- **Documentation**: [API Docs](/docs)

---

            Built with ❤️
