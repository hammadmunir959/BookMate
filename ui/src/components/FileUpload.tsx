import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  LinearProgress,
  Chip,
  IconButton,
  Alert,
} from '@mui/material';
import {
  CloudUpload,
  Close,
  Description,
  PictureAsPdf,
  Article,
  InsertDriveFile,
} from '@mui/icons-material';
import { FileUploadProps } from '../types';

export const FileUpload: React.FC<FileUploadProps> = ({
  onFilesSelected,
  onClose,
  acceptedTypes = [],
  maxFiles = 5,
  maxSize = 10, // MB
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState<string>('');

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: { file: File; errors: { code: string; message: string }[] }[]) => {
      setError('');

      // Handle rejected files
      if (rejectedFiles.length > 0) {
        const errors = rejectedFiles.map(({ errors }) =>
          errors.map((error) => error.message).join(', ')
        );
        setError(`Some files were rejected: ${errors.join('; ')}`);
      }

      // Add accepted files
      setSelectedFiles(prev => {
        const combined = [...prev, ...acceptedFiles];
        return combined.slice(0, maxFiles); // Limit to max files
      });
    },
    [maxFiles]
  );

  const { getRootProps, getInputProps, isDragActive, open: openFileDialog } = useDropzone({
    onDrop,
    accept: acceptedTypes.reduce((acc, type) => {
      acc[type] = [];
      return acc;
    }, {} as Record<string, string[]>),
    maxSize: maxSize * 1024 * 1024, // Convert MB to bytes
    maxFiles,
    multiple: maxFiles > 1,
    noClick: false, // Allow clicking to open file dialog
    noKeyboard: false, // Allow keyboard navigation
  });

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = () => {
    if (selectedFiles.length > 0) {
      onFilesSelected(selectedFiles);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf':
        return <PictureAsPdf color="error" />;
      case 'doc':
      case 'docx':
        return <Description color="primary" />;
      case 'txt':
      case 'md':
        return <Article color="success" />;
      default:
        return <InsertDriveFile color="action" />;
    }
  };

  return (
    <Dialog
      open={true}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: '500px' },
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography component="span" sx={{ fontWeight: 600, fontSize: '1.25rem' }}>Upload Documents</Typography>
        <IconButton onClick={onClose} size="small">
          <Close />
        </IconButton>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ mb: 3 }}>
          {/* Drop Zone */}
          <Box
            {...getRootProps()}
            onClick={(e) => {
              // Prevent event bubbling and ensure click works
              e.stopPropagation();
              if (!isDragActive) {
                openFileDialog();
              }
            }}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: 'pointer',
              bgcolor: isDragActive ? 'primary.50' : 'grey.50',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'primary.50',
              },
            }}
          >
            <input {...getInputProps()} style={{ display: 'none' }} />
            <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
            </Typography>
            <Typography variant="body2" color="text.secondary" mb={2}>
              or click to browse files
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Supported formats: {acceptedTypes.join(', ')} (Max {maxSize}MB per file)
              {maxFiles > 1 && ` • Up to ${maxFiles} files`}
            </Typography>
            <Button
              variant="outlined"
              sx={{ mt: 2 }}
              onClick={(e) => {
                e.stopPropagation();
                openFileDialog();
              }}
            >
              Choose Files
            </Button>
          </Box>

          {/* Error Message */}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Box>

        {/* Selected Files List */}
        {selectedFiles.length > 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Selected Files ({selectedFiles.length}/{maxFiles})
            </Typography>

            <Box sx={{ maxHeight: '300px', overflow: 'auto' }}>
              {selectedFiles.map((file, index) => (
                <Box
                  key={index}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2,
                    p: 2,
                    mb: 1,
                    border: 1,
                    borderColor: 'divider',
                    borderRadius: 1,
                    bgcolor: 'background.paper',
                  }}
                >
                  {getFileIcon(file.name)}
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {file.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatFileSize(file.size)}
                    </Typography>
                  </Box>

                  {/* Upload Progress - Disabled in simplified version */}
                  {/* TODO: Re-enable when upload progress is implemented */}

                  <IconButton size="small" onClick={() => removeFile(index)}>
                    <Close fontSize="small" />
                  </IconButton>
                </Box>
              ))}
            </Box>
          </Box>
        )}

        {/* Accepted File Types Info */}
        <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
            Supported File Types:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {acceptedTypes.map((type) => (
              <Chip
                key={type}
                label={type.toUpperCase()}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.75rem' }}
              />
            ))}
          </Box>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3, pt: 0 }}>
        <Button onClick={onClose} color="inherit">
          Cancel
        </Button>
        <Button
          onClick={handleUpload}
          variant="contained"
          disabled={selectedFiles.length === 0}
          sx={{ minWidth: 120 }}
        >
          Upload {selectedFiles.length > 0 && `(${selectedFiles.length})`}
        </Button>
      </DialogActions>
    </Dialog>
  );
};
