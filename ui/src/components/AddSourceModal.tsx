import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  IconButton,
  LinearProgress,
  Alert,
  TextField,
  Fade,
  Slide,
  Chip,
  Card,
  CardContent,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Avatar,
  Badge,
  Tooltip,
  Paper,
  Stack,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  Close,
  CloudUpload,
  Description,
  PictureAsPdf,
  Article,
  ContentPaste,
  InsertDriveFile,
  CheckCircle,
  Error,
  Info,
  Add,
  Delete,
  FileUpload,
  TextSnippet,
  UploadFile,
  Warning,
  CheckCircleOutline,
} from '@mui/icons-material';
import { FILE_CONFIG, UI_CONFIG, THEME_CONFIG } from '../config/constants';

interface AddSourceModalProps {
  open: boolean;
  onClose: () => void;
  onFilesSelected: (files: File[]) => void;
  onTextPasted?: (text: string, title?: string) => void;
  maxFiles?: number;
  maxSize?: number; // in MB
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`source-tabpanel-${index}`}
      aria-labelledby={`source-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
};

export const AddSourceModal: React.FC<AddSourceModalProps> = ({
  open,
  onClose,
  onFilesSelected,
  onTextPasted,
  maxFiles = 10,
  maxSize = 50,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState<string>('');
  const [pastedText, setPastedText] = useState<string>('');
  const [textTitle, setTextTitle] = useState<string>('');
  const [activeTab, setActiveTab] = useState<number>(0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [dragActive, setDragActive] = useState<boolean>(false);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (!open) {
      setSelectedFiles([]);
      setError('');
      setPastedText('');
      setTextTitle('');
      setIsUploading(false);
      setUploadProgress(0);
      setDragActive(false);
      setActiveTab(0);
    }
  }, [open]);

  // Memoized file validation
  const validateFile = useCallback((file: File): { valid: boolean; error?: string } => {
    const maxSizeBytes = maxSize * 1024 * 1024;
    
    if (file.size > maxSizeBytes) {
      return { valid: false, error: `File too large (max ${maxSize}MB)` };
    }
    
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (!extension || !FILE_CONFIG.ALLOWED_EXTENSIONS.includes(`.${extension}` as any)) {
      return { valid: false, error: 'Invalid file type' };
    }
    
    return { valid: true };
  }, [maxSize]);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: { file: File; errors: { code: string; message: string }[] }[]) => {
      setError('');
      setDragActive(false);

      // Handle rejected files
      if (rejectedFiles.length > 0) {
        const errors = rejectedFiles.map(({ file, errors }) => {
          const errorMessages = errors.map(error => {
            switch (error.code) {
              case 'file-too-large':
                return `${file.name}: File too large (max ${maxSize}MB)`;
              case 'file-invalid-type':
                return `${file.name}: Invalid file type`;
              case 'too-many-files':
                return `Too many files (max ${maxFiles})`;
              default:
                return `${file.name}: ${error.message}`;
            }
          });
          return errorMessages.join(', ');
        });
        setError(`Some files were rejected: ${errors.join('; ')}`);
      }

      // Process accepted files
      if (acceptedFiles.length > 0) {
        const validFiles: File[] = [];
        const errors: string[] = [];

        acceptedFiles.forEach(file => {
          const validation = validateFile(file);
          if (validation.valid) {
            validFiles.push(file);
          } else {
            errors.push(`${file.name}: ${validation.error}`);
          }
        });

        if (errors.length > 0) {
          setError(errors.join('; '));
        }

        if (validFiles.length > 0) {
          setSelectedFiles(prev => {
            const combined = [...prev, ...validFiles];
            const limited = combined.slice(0, maxFiles);
            
            if (combined.length > maxFiles) {
              setError(`Only ${maxFiles} files allowed. Some files were not added.`);
            }
            
            return limited;
          });
        }
      }
    },
    [maxFiles, maxSize, validateFile]
  );

  const { getRootProps, getInputProps, isDragActive: dropzoneDragActive, open: openFileDialog } = useDropzone({
    onDrop,
    accept: FILE_CONFIG.ALLOWED_TYPES.reduce((acc, type) => {
      acc[type] = [...FILE_CONFIG.ALLOWED_EXTENSIONS];
      return acc;
    }, {} as Record<string, string[]>),
    maxSize: maxSize * 1024 * 1024,
    maxFiles: maxFiles,
    multiple: maxFiles > 1,
    noClick: false,
    noKeyboard: false,
    disabled: isUploading,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  // Update drag state based on dropzone
  useEffect(() => {
    setDragActive(dropzoneDragActive);
  }, [dropzoneDragActive]);

  const removeFile = useCallback((index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  const handleUpload = useCallback(async () => {
    if (selectedFiles.length > 0 && !isUploading) {
      setIsUploading(true);
      setUploadProgress(0);
      
      try {
        // Simulate upload progress
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + 10;
          });
        }, 100);

        // Call the actual upload handler
        await new Promise(resolve => setTimeout(resolve, 1000));
        onFilesSelected(selectedFiles);
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        setTimeout(() => {
          setSelectedFiles([]);
          setError('');
          setIsUploading(false);
          setUploadProgress(0);
          onClose();
        }, 500);
      } catch (error) {
        setError('Upload failed. Please try again.');
        setIsUploading(false);
        setUploadProgress(0);
      }
    }
  }, [selectedFiles, isUploading, onFilesSelected, onClose]);

  const handlePasteText = useCallback(() => {
    if (pastedText.trim() && onTextPasted && !isUploading) {
      onTextPasted(pastedText.trim(), textTitle.trim() || 'Pasted Text');
      setPastedText('');
      setTextTitle('');
      setError('');
      onClose();
    }
  }, [pastedText, textTitle, onTextPasted, isUploading, onClose]);

  const handleSubmit = useCallback(() => {
    if (activeTab === 0) {
      handleUpload();
    } else if (activeTab === 1) {
      handlePasteText();
    }
  }, [activeTab, handleUpload, handlePasteText]);

  const canSubmit = useCallback(() => {
    if (isUploading) return false;
    if (activeTab === 0) {
      return selectedFiles.length > 0;
    } else if (activeTab === 1) {
      return pastedText.trim().length > 0;
    }
    return false;
  }, [isUploading, activeTab, selectedFiles.length, pastedText]);

  const formatFileSize = useCallback((bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }, []);

  const getFileIcon = useCallback((filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    const iconProps = { sx: { fontSize: 20 } };
    
    switch (extension) {
      case 'pdf':
        return <PictureAsPdf {...iconProps} sx={{ ...iconProps.sx, color: '#e53e3e' }} />;
      case 'doc':
      case 'docx':
        return <Description {...iconProps} sx={{ ...iconProps.sx, color: '#2b6cb0' }} />;
      case 'txt':
        return <Article {...iconProps} sx={{ ...iconProps.sx, color: '#38a169' }} />;
      case 'md':
        return <Article {...iconProps} sx={{ ...iconProps.sx, color: '#f59e0b' }} />;
      case 'html':
        return <Article {...iconProps} sx={{ ...iconProps.sx, color: '#f97316' }} />;
      default:
        return <InsertDriveFile {...iconProps} sx={{ ...iconProps.sx, color: '#6b7280' }} />;
    }
  }, []);

  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
    setError('');
  }, []);

  const totalFileSize = useMemo(() => {
    return selectedFiles.reduce((total, file) => total + file.size, 0);
  }, [selectedFiles]);


  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      TransitionComponent={Slide}
      TransitionProps={{ direction: 'up' } as any}
      PaperProps={{
        sx: {
          backgroundColor: '#1a1a1a',
          color: '#ffffff',
          minHeight: '600px',
          maxHeight: '90vh',
          borderRadius: 3,
          overflow: 'hidden',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      }}
    >
      <DialogTitle 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          p: 4,
          pb: 2,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Avatar sx={{ 
            bgcolor: 'rgba(255, 255, 255, 0.2)',
            width: 48,
            height: 48,
          }}>
            <Add sx={{ fontSize: 24 }} />
          </Avatar>
          <Box>
            <Typography variant="h4" sx={{ 
              fontWeight: 700,
              fontSize: '1.75rem',
              mb: 0.5,
            }}>
              Add Knowledge Sources
            </Typography>
            <Typography variant="body2" sx={{ 
              color: 'rgba(255, 255, 255, 0.8)',
              fontSize: '0.95rem',
            }}>
              Upload documents or paste text to expand your knowledge base
            </Typography>
          </Box>
        </Box>
        <IconButton 
          onClick={onClose} 
          sx={{ 
            color: 'rgba(255, 255, 255, 0.8)',
            '&:hover': {
              color: 'white',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            },
          }}
        >
          <Close />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 0 }}>
        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 4, pt: 2 }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange}
            sx={{
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 600,
                fontSize: '1rem',
                minHeight: 48,
                color: 'text.secondary',
                '&.Mui-selected': {
                  color: '#667eea',
                },
              },
              '& .MuiTabs-indicator': {
                backgroundColor: '#667eea',
                height: 3,
                borderRadius: '3px 3px 0 0',
              },
            }}
          >
            <Tab 
              icon={<FileUpload />} 
              iconPosition="start"
              label="Upload Files" 
              disabled={isUploading}
            />
            <Tab 
              icon={<TextSnippet />} 
              iconPosition="start"
              label="Paste Text" 
              disabled={isUploading}
            />
          </Tabs>
        </Box>

        <Box sx={{ p: 4 }}>

          {/* Upload Tab Content */}
          <TabPanel value={activeTab} index={0}>
            <Stack spacing={3}>
              {/* Drop Zone */}
              <Paper
                {...getRootProps()}
                elevation={0}
                sx={{
                  border: '2px dashed',
                  borderColor: dragActive ? '#667eea' : 'rgba(255,255,255,0.2)',
                  borderRadius: 3,
                  p: 6,
                  textAlign: 'center',
                  cursor: isUploading ? 'not-allowed' : 'pointer',
                  bgcolor: dragActive ? 'rgba(102, 126, 234, 0.05)' : 'rgba(255,255,255,0.02)',
                  transition: 'all 0.3s ease',
                  position: 'relative',
                  overflow: 'hidden',
                  opacity: isUploading ? 0.6 : 1,
                  '&:hover': !isUploading ? {
                    borderColor: '#667eea',
                    bgcolor: 'rgba(102, 126, 234, 0.08)',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 8px 25px rgba(102, 126, 234, 0.15)',
                  } : {},
                }}
              >
                <input {...getInputProps()} />
                <Box sx={{ position: 'relative', zIndex: 1 }}>
                  <Avatar sx={{
                    width: 80,
                    height: 80,
                    bgcolor: dragActive ? 'rgba(102, 126, 234, 0.2)' : 'rgba(102, 126, 234, 0.1)',
                    mb: 3,
                    mx: 'auto',
                  }}>
                    <CloudUpload sx={{ fontSize: 40, color: dragActive ? '#667eea' : '#9ca3af' }} />
                  </Avatar>
                  
                  <Typography variant="h5" sx={{ 
                    fontWeight: 600, 
                    mb: 1,
                    color: 'text.primary',
                  }}>
                    {dragActive ? 'Drop files here' : 'Drag and drop files here'}
                  </Typography>
                  
                  <Typography variant="body1" sx={{ 
                    color: 'text.secondary', 
                    mb: 4,
                    fontSize: '1rem',
                  }}>
                    or click to browse files
                  </Typography>

                  <Stack direction="row" spacing={1} justifyContent="center" sx={{ mb: 4 }}>
                    {FILE_CONFIG.ALLOWED_EXTENSIONS.map((ext) => (
                      <Chip
                        key={ext}
                        label={ext.toUpperCase()}
                        size="small"
                        sx={{
                          bgcolor: 'rgba(102, 126, 234, 0.1)',
                          color: '#667eea',
                          fontWeight: 500,
                          fontSize: '0.75rem',
                        }}
                      />
                    ))}
                  </Stack>

                  <Typography variant="body2" sx={{ 
                    color: 'text.secondary',
                    mb: 3,
                  }}>
                    Maximum {maxSize}MB per file • Up to {maxFiles} files
                  </Typography>

                  <Button
                    variant="outlined"
                    disabled={isUploading}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (!isUploading) {
                        openFileDialog();
                      }
                    }}
                    startIcon={<UploadFile />}
                    sx={{
                      borderColor: 'rgba(255,255,255,0.3)',
                      color: 'text.primary',
                      px: 4,
                      py: 1.5,
                      '&:hover': {
                        borderColor: '#667eea',
                        bgcolor: 'rgba(102, 126, 234, 0.1)',
                      },
                    }}
                  >
                    Choose Files
                  </Button>
                </Box>
              </Paper>

              {/* Error Message */}
              {error && (
                <Alert 
                  severity="error" 
                  sx={{ 
                    bgcolor: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    color: 'text.primary',
                    '& .MuiAlert-icon': { color: '#ef4444' },
                  }}
                >
                  {error}
                </Alert>
              )}

              {/* Upload Progress */}
              {isUploading && (
                <Paper sx={{ p: 3, bgcolor: 'rgba(102, 126, 234, 0.05)' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <CircularProgress size={20} sx={{ mr: 2, color: '#667eea' }} />
                    <Typography variant="body2" sx={{ color: 'text.secondary', mr: 2 }}>
                      Uploading files...
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      {uploadProgress}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={uploadProgress}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      bgcolor: 'rgba(255,255,255,0.1)',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: '#667eea',
                      },
                    }}
                  />
                </Paper>
              )}

              {/* Selected Files List */}
              {selectedFiles.length > 0 && (
                <Paper sx={{ p: 3, bgcolor: 'rgba(255,255,255,0.02)' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
                      Selected Files
                    </Typography>
                    <Badge 
                      badgeContent={selectedFiles.length} 
                      color="primary"
                      sx={{
                        '& .MuiBadge-badge': {
                          bgcolor: selectedFiles.length >= maxFiles ? '#ef4444' : '#667eea',
                        },
                      }}
                    >
                      <Chip 
                        label={`${selectedFiles.length}/${maxFiles}`}
                        size="small"
                        sx={{
                          bgcolor: selectedFiles.length >= maxFiles ? 'rgba(239, 68, 68, 0.1)' : 'rgba(102, 126, 234, 0.1)',
                          color: selectedFiles.length >= maxFiles ? '#ef4444' : '#667eea',
                          fontWeight: 600,
                        }}
                      />
                    </Badge>
                  </Box>

                  <List sx={{ maxHeight: '300px', overflow: 'auto' }}>
                    {selectedFiles.map((file, index) => (
                      <ListItem
                        key={index}
                        sx={{
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: 2,
                          mb: 1,
                          bgcolor: 'rgba(255,255,255,0.02)',
                          transition: 'all 0.2s ease',
                          '&:hover': {
                            bgcolor: 'rgba(255,255,255,0.05)',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                          },
                        }}
                      >
                        <ListItemIcon>
                          {getFileIcon(file.name)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontWeight: 500,
                                color: 'text.primary',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {file.name}
                            </Typography>
                          }
                          secondary={
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {formatFileSize(file.size)}
                            </Typography>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Tooltip title="Remove file">
                            <IconButton 
                              size="small" 
                              onClick={() => removeFile(index)}
                              disabled={isUploading}
                              sx={{
                                color: 'text.secondary',
                                '&:hover': {
                                  color: '#ef4444',
                                  bgcolor: 'rgba(239, 68, 68, 0.1)',
                                },
                              }}
                            >
                              <Delete fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>

                  {/* File Limit Progress */}
                  <Box sx={{ mt: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                        File limit
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        {selectedFiles.length}/{maxFiles}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={(selectedFiles.length / maxFiles) * 100}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: 'rgba(255,255,255,0.1)',
                        '& .MuiLinearProgress-bar': {
                          bgcolor: selectedFiles.length >= maxFiles ? '#ef4444' : '#667eea',
                        },
                      }}
                    />
                  </Box>
                </Paper>
              )}
            </Stack>
          </TabPanel>

          {/* Paste Text Tab Content */}
          <TabPanel value={activeTab} index={1}>
            <Stack spacing={3}>
              {/* Text Title Input */}
              <TextField
                fullWidth
                label="Document Title (Optional)"
                value={textTitle}
                onChange={(e) => setTextTitle(e.target.value)}
                placeholder="Enter a title for your text content"
                variant="outlined"
                disabled={isUploading}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    bgcolor: 'rgba(255,255,255,0.05)',
                    color: 'text.primary',
                    borderRadius: 2,
                    '& fieldset': {
                      borderColor: 'rgba(255,255,255,0.2)',
                    },
                    '&:hover fieldset': {
                      borderColor: '#667eea',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#667eea',
                      borderWidth: 2,
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: 'text.secondary',
                    '&.Mui-focused': {
                      color: '#667eea',
                    },
                  },
                  '& .MuiOutlinedInput-input': {
                    '&::placeholder': {
                      color: 'text.secondary',
                      opacity: 0.7,
                    },
                  },
                }}
              />

              {/* Text Content Input */}
              <TextField
                fullWidth
                multiline
                rows={12}
                label="Paste your text content here"
                value={pastedText}
                onChange={(e) => setPastedText(e.target.value)}
                placeholder="Paste or type your text content here..."
                variant="outlined"
                disabled={isUploading}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    bgcolor: 'rgba(255,255,255,0.05)',
                    color: 'text.primary',
                    borderRadius: 2,
                    '& fieldset': {
                      borderColor: 'rgba(255,255,255,0.2)',
                    },
                    '&:hover fieldset': {
                      borderColor: '#667eea',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#667eea',
                      borderWidth: 2,
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: 'text.secondary',
                    '&.Mui-focused': {
                      color: '#667eea',
                    },
                  },
                  '& .MuiOutlinedInput-input': {
                    '&::placeholder': {
                      color: 'text.secondary',
                      opacity: 0.7,
                    },
                  },
                }}
              />

              {/* Character Count and Info */}
              <Paper sx={{ 
                p: 3,
                bgcolor: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Info sx={{ fontSize: 16, color: 'text.secondary' }} />
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      Text will be processed and added to your knowledge base
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ 
                    color: pastedText.length > 10000 ? '#ef4444' : 'text.secondary',
                    fontWeight: 500,
                  }}>
                    {pastedText.length.toLocaleString()} characters
                  </Typography>
                </Box>
              </Paper>
            </Stack>
          </TabPanel>
        </Box>
      </DialogContent>

      <DialogActions sx={{ 
        p: 4, 
        pt: 3, 
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        bgcolor: 'rgba(255, 255, 255, 0.02)',
        gap: 2,
      }}>
        <Button 
          onClick={onClose} 
          disabled={isUploading}
          variant="outlined"
          sx={{ 
            color: 'text.secondary',
            borderColor: 'rgba(255, 255, 255, 0.2)',
            px: 4,
            py: 1.5,
            borderRadius: 2,
            textTransform: 'none',
            fontWeight: 600,
            '&:hover': {
              color: 'text.primary',
              borderColor: 'rgba(255, 255, 255, 0.4)',
              bgcolor: 'rgba(255, 255, 255, 0.05)',
            },
          }}
        >
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={!canSubmit()}
          startIcon={isUploading ? <CircularProgress size={16} color="inherit" /> : activeTab === 0 ? <CloudUpload /> : <TextSnippet />}
          sx={{
            minWidth: 160,
            px: 4,
            py: 1.5,
            borderRadius: 2,
            background: canSubmit() 
              ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
              : 'rgba(255,255,255,0.1)',
            color: canSubmit() ? 'white' : 'text.secondary',
            fontWeight: 600,
            textTransform: 'none',
            fontSize: '1rem',
            boxShadow: canSubmit() 
              ? '0 4px 15px rgba(102, 126, 234, 0.3)'
              : 'none',
            '&:hover': canSubmit() ? {
              background: 'linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)',
              boxShadow: '0 8px 25px rgba(102, 126, 234, 0.4)',
              transform: 'translateY(-1px)',
            } : {},
            '&:disabled': {
              background: 'rgba(255,255,255,0.1)',
              color: 'text.secondary',
              transform: 'none',
              boxShadow: 'none',
            },
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        >
          {isUploading ? (
            'Processing...'
          ) : activeTab === 0 ? (
            `Add ${selectedFiles.length > 0 ? `(${selectedFiles.length})` : ''} Files`
          ) : (
            'Add Text'
          )}
        </Button>
      </DialogActions>
    </Dialog>
  );
};
