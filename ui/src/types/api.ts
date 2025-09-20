// Simplified API Types - matching our microservice endpoints

// Core endpoint types
export interface QueryRequest {
  query: string;
  document_ids?: string[];
  document_id?: string; // Legacy support
  system_prompt?: string;
  context?: string;
  conversation_history?: Array<{role: string, content: string}>;
  max_tokens?: number;
  temperature?: number;
  top_k?: number;
  similarity_threshold?: number;
}

export interface QueryResponse {
  success: boolean;
  query: string;
  answer: string;
  citations: CitationInfo[];
  total_chunks_retrieved: number;
  processing_stats: Record<string, unknown>;
  model_used: string;
  token_count?: number;
  confidence_score?: number;
  processing_time: number;
  error_message?: string;
}

export interface CitationInfo {
  chunk_id: string;
  document_id: string;
  content: string; // FULL chunk content (not truncated)
  relevance_score: number;
  citation_text: string; // Format: "(cit#1)", "(cit#2)", etc.
  // Enhanced metadata for better UX
  chunk_index: number;
  chunk_number: number; // Human-readable (1-based)
  display_name: string; // Clean filename without IDs/hashes
  token_count: number;
  start_position: number;
  end_position: number;
  page_number?: number;
  section?: string;
}

export interface ChunkInfo {
  chunk_id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  token_count: number;
  start_position: number;
  end_position: number;
  metadata: Record<string, unknown>;
  summary?: string;
}

export interface IngestionStepStatus {
  name: string;
  status: string;
  start_time?: number;
  end_time?: number;
  duration?: number;
  message?: string;
  error?: string;
}

export interface IngestionResponse {
  success: boolean;
  document_id: string;
  filename: string;
  file_size: number;
  mime_type: string;
  total_chunks: number;
  chunks: ChunkInfo[];
  summary: Record<string, unknown>;
  processing_stats: Record<string, unknown>;
  steps_status: IngestionStepStatus[];
  total_processing_time: number;
  citation_mode: string;
  error_message?: string;
}

// Legacy compatibility types (simplified)
export interface DocumentUploadResponse {
  success: boolean;
  document_id: string;
  message: string;
  chunks_created: number;
  processing_time: number;
  errors?: string[];
}

export interface DocumentDeleteResponse {
  success: boolean;
  document_id: string;
  chunks_deleted: number;
  message: string;
}

export interface BatchUploadResponse {
  total_documents: number;
  successful_uploads: number;
  failed_uploads: number;
  results: DocumentUploadResponse[];
  total_chunks_created: number;
  total_processing_time: number;
}

export interface DocumentListResponse {
  documents: Array<{
    id: string;
    filename: string;
    file_size: number;
    file_type: string;
    upload_date: string;
    status: string;
    chunks_count: number;
    notebook_id: string;
  }>;
  total_count: number;
}

export interface DocumentDetailsResponse {
  success: boolean;
  document_id: string;
  filename: string;
  file_size: number;
  mime_type: string;
  total_chunks: number;
  chunks: ChunkInfo[];
  summary: Record<string, unknown>;
  processing_stats: Record<string, unknown>;
  steps_status: IngestionStepStatus[];
  total_processing_time: number;
  citation_mode: string;
  error_message?: string;
}

export interface ResetSessionResponse {
  success: boolean;
  message: string;
  details: {
    chroma_cleared: boolean;
    sqlite_cleared: boolean;
    cache_cleared: boolean;
  };
}

// System types
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: number;
  services: Record<string, string>;
  stats: Record<string, unknown>;
}

export interface SystemStatsResponse {
  service: string;
  version: string;
  databases: {
    chromadb: Record<string, unknown>;
    sqlite: Record<string, unknown>;
  };
  cache: Record<string, unknown>;
  config: Record<string, unknown>;
}

export interface ErrorResponse {
  error: string;
  type: string;
  timestamp: number;
  details?: Record<string, unknown>;
}
