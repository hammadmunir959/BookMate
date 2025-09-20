/**
 * Security utilities for input sanitization and validation
 */

import { VALIDATION_CONFIG, FILE_CONFIG } from '../config/constants';

/**
 * Sanitizes user input to prevent XSS attacks
 * @param input - The input string to sanitize
 * @returns Sanitized string safe for rendering
 */
export const sanitizeInput = (input: string): string => {
  if (typeof input !== 'string') {
    return '';
  }

  return input
    // Remove script tags and their content
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    // Remove style tags and their content
    .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
    // Remove all HTML tags but preserve content
    .replace(/<[^>]*>/g, '')
    // Remove javascript: URLs
    .replace(/javascript:/gi, '')
    // Remove data: URLs that might contain scripts
    .replace(/data:[^;]*;base64,[a-zA-Z0-9+/]+/g, '[REMOVED]')
    // Remove event handlers
    .replace(/on\w+\s*=/gi, '')
    // Remove potentially dangerous attributes
    .replace(/\b(href|src|action|formaction)\s*=\s*["'][^"']*["']/gi, (match) => {
      // Only allow safe URLs
      const url = match.match(/["']([^"']*)["']/);
      if (url && (url[1].startsWith('http') || url[1].startsWith('https') || url[1].startsWith('/'))) {
        return match;
      }
      return '';
    })
    // Trim whitespace
    .trim();
};

/**
 * Validates message content for length and content
 * @param message - The message to validate
 * @returns Object with validation result and sanitized message
 */
export const validateMessage = (message: string): {
  isValid: boolean;
  sanitizedMessage: string;
  error?: string;
} => {
  if (typeof message !== 'string') {
    return {
      isValid: false,
      sanitizedMessage: '',
      error: 'Invalid input type'
    };
  }

  const trimmed = message.trim();

  if (!trimmed) {
    return {
      isValid: false,
      sanitizedMessage: '',
      error: 'Message cannot be empty'
    };
  }

  if (trimmed.length > VALIDATION_CONFIG.MESSAGE_MAX_LENGTH) {
    return {
      isValid: false,
      sanitizedMessage: '',
      error: `Message too long (maximum ${VALIDATION_CONFIG.MESSAGE_MAX_LENGTH} characters)`
    };
  }

  const sanitized = sanitizeInput(trimmed);

  if (!sanitized) {
    return {
      isValid: false,
      sanitizedMessage: '',
      error: 'Message contains invalid content'
    };
  }

  return {
    isValid: true,
    sanitizedMessage: sanitized
  };
};

/**
 * Checks if a file type is safe for upload
 * @param fileType - The MIME type of the file
 * @returns boolean indicating if the file type is allowed
 */
export const isSafeFileType = (fileType: string): boolean => {
  return FILE_CONFIG.ALLOWED_TYPES.includes(fileType as any);
};

/**
 * Generates a cryptographically secure random ID
 * @param length - Length of the ID
 * @returns Random string ID
 */
export const generateSecureId = (length: number = 16): string => {
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
};

/**
 * Cleans and formats filenames for better user experience
 * @param filename - Original filename
 * @returns Cleaned filename
 */
export const cleanFilename = (filename: string): string => {
  if (!filename) return filename;

  // Remove file extension for processing
  const lastDotIndex = filename.lastIndexOf('.');
  const nameWithoutExt = lastDotIndex !== -1 ? filename.substring(0, lastDotIndex) : filename;
  const extension = lastDotIndex !== -1 ? filename.substring(lastDotIndex) : '';

  // Clean up the name part
  let cleaned = nameWithoutExt
    // Remove timestamps and technical IDs (more aggressive)
    .replace(/\d{10,13}(\.\d+)?/g, '') // Unix timestamps with decimals
    .replace(/\d{4}-\d{2}-\d{2}[_\s]\d{2}-\d{2}-\d{2}/g, '') // Date-time patterns
    .replace(/_\d+(\.\d+)*$/g, '') // Trailing numbers with decimals
    .replace(/_[a-f0-9]{8,}$/gi, '') // Hex hashes
    .replace(/_\w{8,}$/g, '') // Long alphanumeric strings
    // Clean up separators and extra characters
    .replace(/[_\-\.]+/g, ' ') // Replace multiple separators with single space
    .replace(/\s+/g, ' ') // Replace multiple spaces with single space
    // Remove common unwanted patterns
    .replace(/copy\s*\d*/gi, '') // "copy", "copy 2", etc.
    .replace(/duplicate/gi, '')
    .replace(/temp/gi, '')
    .replace(/tmp/gi, '')
    .replace(/tempfile/gi, '')
    .replace(/download/gi, '')
    .replace(/document/gi, '') // Remove generic words
    .replace(/file/gi, '')
    // Clean up parentheses and brackets
    .replace(/\([^\)]*\)/g, '') // Remove content in parentheses
    .replace(/\[[^\]]*\]/g, '') // Remove content in brackets
    // Remove file extensions that might be in the name
    .replace(/\.(pdf|doc|docx|txt|md|html|rtf|odt)$/gi, '')
    // Trim and capitalize properly
    .trim()
    .replace(/\b\w/g, l => l.toUpperCase()); // Title case

  // If the cleaned name is too short or empty, use a generic name
  if (cleaned.length < 3) {
    cleaned = `Document${extension}`;
  }

  // Re-add extension
  return cleaned + extension;
};

/**
 * Extracts meaningful title from filename or generates one
 * @param filename - Original filename
 * @returns User-friendly title
 */
export const generateFriendlyTitle = (filename: string): string => {
  const cleaned = cleanFilename(filename);

  // Remove extension for title
  const lastDotIndex = cleaned.lastIndexOf('.');
  const title = lastDotIndex !== -1 ? cleaned.substring(0, lastDotIndex) : cleaned;

  // If title is too long, truncate it
  return title.length > 50 ? title.substring(0, 47) + '...' : title;
};
