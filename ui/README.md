# RAG UI - Simplified Microservice Frontend

This is the React-based frontend for the simplified RAG microservice system.

## Features

- **Document Upload**: Upload documents via the `/ingestion` endpoint
- **Query Processing**: Ask questions using the `/query` endpoint
- **Document Management**: List, delete, and view document details
- **Citation Display**: View citations from retrieved chunks
- **Session Management**: Reset session and clear all data

## API Endpoints Used

### Core Endpoints
- `POST /ingestion` - Upload and process documents
- `POST /query` - Unified retrieval, augmentation, and generation
- `GET /health` - Health check
- `GET /stats` - System statistics

### Document Management (Compatibility)
- `POST /documents/upload` - Upload single document
- `POST /documents/upload-batch` - Upload multiple documents
- `DELETE /documents/delete` - Delete document
- `GET /documents/list` - List all documents
- `GET /documents/details/{documentId}` - Get document details
- `POST /reset_session` - Reset session

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

1. Navigate to the UI directory:
```bash
cd ui/
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The UI will be available at `http://localhost:5173`

### Environment Configuration

Create a `.env` file in the `ui/` directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Key Components

- **App.tsx** - Main application component
- **ChatInterface.tsx** - Chat interface with citations
- **Sidebar.tsx** - Document sidebar with management
- **FileUpload.tsx** - File upload component

### API Integration

The UI uses simplified API calls that work with the new microservice endpoints:

- Uses `ingestDocument()` for file uploads (calls `/ingestion`)
- Uses `processQuery()` for queries (calls `/query`)
- Uses `listDocuments()` for sidebar display (calls `/documents/list`)
- Uses `deleteDocument()` for deletion (calls `/documents/delete`)

## Architecture

This UI is designed to work with the simplified RAG microservice that has only 2 core endpoints plus additional compatibility endpoints for smooth migration.

The interface provides:
- File upload with progress tracking
- Document listing and management
- Chat interface with citation display
- Session management capabilities

## Migration Notes

This UI has been adapted from the original complex API to work with the simplified microservice endpoints while maintaining backward compatibility through additional endpoint wrappers.