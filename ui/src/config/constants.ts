/**
 * Application constants and configuration
 */

// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  WS_URL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
  TIMEOUT: 120000, // 2 minutes
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
} as const;

// UI Configuration
export const UI_CONFIG = {
  SIDEBAR_WIDTH: 320,
  MAX_MESSAGE_LENGTH: 10000,
  MAX_FILE_SIZE_MB: 50,
  MAX_FILES_COUNT: 10,
  LOADING_DEBOUNCE_MS: 300,
  SCROLL_DEBOUNCE_MS: 100,
} as const;

// Theme Configuration
export const THEME_CONFIG = {
  PRIMARY_COLOR: '#3A6FF7',
  SECONDARY_COLOR: '#6366f1',
  BACKGROUND_DEFAULT: '#0F0F0F',
  BACKGROUND_PAPER: '#1A1A1A',
  TEXT_PRIMARY: '#FFFFFF',
  TEXT_SECONDARY: '#B8BCC8',
  BORDER_RADIUS: 12,
  FONT_FAMILY: '"Inter", "SF Pro Display", "Roboto", "Helvetica", "Arial", sans-serif',
} as const;

// File Upload Configuration
export const FILE_CONFIG = {
  ALLOWED_TYPES: [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'text/plain',
    'text/markdown',
    'text/html',
  ] as const,
  ALLOWED_EXTENSIONS: ['.pdf', '.docx', '.doc', '.txt', '.md', '.html'] as const,
  MAX_SIZE_BYTES: 50 * 1024 * 1024, // 50MB
  CHUNK_SIZE: 1024 * 1024, // 1MB chunks for upload
} as const;

// Validation Configuration
export const VALIDATION_CONFIG = {
  MESSAGE_MIN_LENGTH: 1,
  MESSAGE_MAX_LENGTH: 10000,
  TITLE_MAX_LENGTH: 200,
  FILENAME_MAX_LENGTH: 255,
} as const;

// WebSocket Configuration
export const WS_CONFIG = {
  RECONNECT_ATTEMPTS: 5,
  RECONNECT_DELAY: 1000,
  HEARTBEAT_INTERVAL: 30000, // 30 seconds
  CONNECTION_TIMEOUT: 5000,
} as const;

// Feature Flags
export const FEATURES = {
  ENABLE_WEBSOCKET: false, // Disabled for simplified API
  ENABLE_VOICE_INPUT: false,
  ENABLE_FILE_PREVIEW: true,
  ENABLE_DARK_MODE: true,
  ENABLE_OFFLINE_MODE: false,
  ENABLE_ANALYTICS: false,
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Unable to connect to the server. Please check your connection.',
  UPLOAD_FAILED: 'File upload failed. Please try again.',
  MESSAGE_TOO_LONG: 'Message is too long. Maximum 10,000 characters allowed.',
  INVALID_FILE_TYPE: 'File type not supported.',
  FILE_TOO_LARGE: 'File is too large. Maximum 50MB allowed.',
  BACKEND_UNAVAILABLE: 'Backend server is currently unavailable.',
  GENERIC_ERROR: 'An unexpected error occurred. Please try again.',
} as const;

// Success Messages
export const SUCCESS_MESSAGES = {
  UPLOAD_SUCCESS: 'File uploaded successfully',
  MESSAGE_SENT: 'Message sent successfully',
  DOCUMENT_DELETED: 'Document deleted successfully',
  SESSION_RESET: 'Session reset successfully',
} as const;
