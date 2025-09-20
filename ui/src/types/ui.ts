// UI-specific types and interfaces

export interface Notebook {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  document_count: number;
  total_chunks: number;
}

export interface Document {
  id: string;
  filename: string;
  original_filename?: string;
  file_type: string;
  file_size: number;
  upload_date: string;
  notebook_id: string;
  metadata?: Record<string, unknown>;
  chunks_count: number;
  status: 'processing' | 'ready' | 'error' | 'uploading' | 'completed';
  mime_type?: string;
  chunks?: import('./api').ChunkInfo[];
  summary?: Record<string, unknown>;
  processing_stats?: Record<string, unknown>;
  steps_status?: import('./api').IngestionStepStatus[];
  total_processing_time?: number;
  citation_mode?: string;
  is_selected?: boolean; // For selected sources feature
  ingestion_progress?: {
    status: 'in_progress' | 'completed' | 'failed';
    progress: string; // "3/6"
    progress_percentage: number;
    current_step?: string;
    completed_steps: number;
    total_steps: number;
  };
}

export interface SelectedSourcesState {
  selectedDocuments: Document[];
  isEnabled: boolean;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    query?: string;
    response?: QueryResponse;
    sources?: unknown[];
    citations?: import('./api').CitationInfo[];
    reasoning_steps?: unknown[];
    reasoning?: string | null;
    suggested_questions?: string[] | null;
    confidence_score?: number;
    total_chunks_retrieved?: number;
    processing_stats?: Record<string, unknown>;
    model_used?: string;
    suggestions?: string[];
  };
}

export interface ChatSession {
  id: string;
  notebook_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message?: string;
}

export interface UIState {
  sidebarOpen: boolean;
  rightPanelOpen: boolean;
  currentNotebookId?: string;
  currentChatSessionId?: string;
  isLoading: boolean;
  error?: string;
  theme: 'light' | 'dark';
}

export interface DocumentStats {
  total_documents: number;
  total_chunks: number;
  supported_formats: string[];
  recent_uploads: Document[];
}

export interface SearchFilters {
  query: string;
  document_ids?: string[];
  date_from?: string;
  date_to?: string;
  file_types?: string[];
  min_score?: number;
}

export interface ProcessingProgress {
  task_id: string;
  type: 'upload' | 'query' | 'reindex';
  progress: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
  start_time: string;
  estimated_completion?: string;
}

// Component Props Types
export interface SidebarProps {
  notebooks: Notebook[];
  currentNotebookId?: string;
  onNotebookSelect: (notebookId: string) => void;
  onCreateNotebook: () => void;
  onSettingsClick: () => void;
  className?: string;
}

export interface MainWorkspaceProps {
  notebook?: Notebook;
  documents: Document[];
  chatSession?: ChatSession;
  messages: ChatMessage[];
  onDocumentUpload: (files: File[]) => void;
  onDocumentDelete: (documentId: string) => void;
  onSendMessage: (message: string) => void;
  onNotebookUpdate: (updates: Partial<Notebook>) => void;
  className?: string;
}

export interface RightPanelProps {
  notebook?: Notebook;
  insights: unknown[];
  summaries: unknown[];
  onGenerateSummary: (documentId: string) => void;
  onExtractHighlights: (documentId: string) => void;
  onCreateOutline: () => void;
  className?: string;
}

export interface ChatInterfaceProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSendMessage: (message: string) => void;
  className?: string;
  sourceCount?: number;
  sources?: Document[];
}

export interface DocumentCardProps {
  document: Document;
  onDelete: (documentId: string) => void;
  onPreview: (documentId: string) => void;
  className?: string;
}

export interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  onClose?: () => void;
  acceptedTypes?: string[];
  maxFiles?: number;
  maxSize?: number; // in MB
  className?: string;
}

// Import types from API
import { QueryResponse } from './api';
