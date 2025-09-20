import axios from 'axios';
import {
  QueryRequest,
  QueryResponse,
  DocumentUploadResponse,
  BatchUploadResponse,
  DocumentDeleteResponse,
  HealthCheckResponse,
  SystemStatsResponse,
  IngestionResponse
} from '../types';
import { API_CONFIG } from '../config/constants';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || API_CONFIG.BASE_URL;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.status, error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API Endpoints - Simplified for our microservice
export const apiEndpoints = {
  // Core endpoints
  ingestion: '/ingestion',
  query: '/query',

  // Additional endpoints for UI compatibility
  uploadDocument: '/documents/upload',
  uploadBatch: '/documents/upload-batch',
  deleteDocument: '/documents/delete',
  listDocuments: '/documents/list',
  getDocumentDetails: (documentId: string) => `/documents/details/${documentId}`,
  resetSession: '/reset_session',

  // System endpoints
  healthCheck: '/health',
  getSystemStats: '/stats',
  root: '/',
};

// API Functions - Simplified for our microservice

// Core endpoints
export const ingestDocument = (file: File, customMetadata?: Record<string, unknown>, uploaderId?: string) => {
  const formData = new FormData();
  formData.append('file', file);
  if (customMetadata) {
    formData.append('custom_metadata', JSON.stringify(customMetadata));
  }
  if (uploaderId) {
    formData.append('uploader_id', uploaderId);
  }

  return api.post<IngestionResponse>(apiEndpoints.ingestion, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const processQuery = (data: QueryRequest) =>
  api.post<QueryResponse>(apiEndpoints.query, data);

// Legacy endpoints for UI compatibility
export const uploadDocument = (file: File, customMetadata?: Record<string, unknown>) => {
  const formData = new FormData();
  formData.append('file', file);
  if (customMetadata) {
    formData.append('custom_metadata', JSON.stringify(customMetadata));
  }

  return api.post<DocumentUploadResponse>(apiEndpoints.uploadDocument, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const uploadBatchDocuments = (files: File[]) => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append(`files`, file);
  });

  return api.post<BatchUploadResponse>(apiEndpoints.uploadBatch, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const deleteDocument = (documentId: string) =>
  api.delete<DocumentDeleteResponse>(`${apiEndpoints.deleteDocument}?document_id=${documentId}`);

export const listDocuments = () => api.get(apiEndpoints.listDocuments);

export const getDocumentDetails = (documentId: string) =>
  api.get(`${apiEndpoints.getDocumentDetails(documentId)}`);

export const resetSession = () =>
  api.post(apiEndpoints.resetSession, {});

// System endpoints
export const getHealthCheck = () =>
  api.get<HealthCheckResponse>(apiEndpoints.healthCheck);

export const getSystemStats = () =>
  api.get<SystemStatsResponse>(apiEndpoints.getSystemStats);

export const getRoot = () => api.get(apiEndpoints.root);

// Error handling
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Helper function to handle API errors
export const handleApiError = (error: unknown): never => {
  const axiosError = error as any; // Type assertion for axios error

  if (axiosError.response) {
    const { status, data } = axiosError.response;
    throw new ApiError(
      data?.error || data?.message || `HTTP ${status}`,
      status,
      data
    );
  } else if (axiosError.request) {
    throw new ApiError('Network error - please check your connection', 0);
  } else {
    throw new ApiError((axiosError as Error).message || 'Unknown error occurred', 0);
  }
};

// Export the axios instance for advanced usage
export { api };
