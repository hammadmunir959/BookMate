import { useState, useEffect } from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  Button,
  LinearProgress,
  CircularProgress,
} from '@mui/material';
import {
  Description,
  PictureAsPdf,
  Article,
  InsertDriveFile,
  Add,
  MoreVert,
  Delete,
  CheckCircle,
  Error,
  Info,
  Autorenew,
  Schedule,
  Refresh,
} from '@mui/icons-material';
import { Document } from '../types';
import { wsService } from '../services/websocket';
import { UI_CONFIG } from '../config/constants';
import { cleanFilename } from '../utils/security';

interface SourcesPanelProps {
  sources: Document[];
  onAddSource: () => void;
  onNewChat?: () => void;
  onSourceDelete: (sourceId: string) => void;
  onSourceReplace: (sourceId: string) => void;
  onSourceDetails?: (sourceId: string) => void;
  selectedSources?: Document[];
  onSourceSelect?: (source: Document) => void;
  onSourceDeselect?: (sourceId: string) => void;
  className?: string;
}

const DRAWER_WIDTH = UI_CONFIG.SIDEBAR_WIDTH;

export const SourcesPanel: React.FC<SourcesPanelProps> = ({
  sources,
  onAddSource,
  onNewChat,
  onSourceDelete,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onSourceReplace,
  onSourceDetails,
  selectedSources = [],
  onSourceSelect,
  onSourceDeselect,
  className,
}) => {
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [documentProgress, setDocumentProgress] = useState<Map<string, {
    status: string;
    progress: number;
    progress_percentage: number;
    current_step?: string;
  }>>(new Map());

  // Subscribe to WebSocket progress updates
  useEffect(() => {
    const handleProgressUpdate = (progress: {
      document_id: string;
      status: string;
      progress: number;
      progress_percentage: number;
      current_step?: string;
    }) => {
      setDocumentProgress(prev => {
        const newMap = new Map(prev);
        newMap.set(progress.document_id, progress);
        return newMap;
      });
    };

    // Subscribe to progress updates for all processing documents
    sources.forEach(source => {
      if (source.status === 'processing' || source.status === 'uploading') {
        wsService.subscribeToDocument(source.id, handleProgressUpdate);
      }
    });

    // Cleanup subscriptions on unmount or when sources change
    return () => {
      sources.forEach(source => {
        wsService.unsubscribeFromDocument(source.id);
      });
    };
  }, [sources]); // Added sources dependency for proper cleanup

  // Documents are now loaded by the parent App component
  // This prevents state synchronization issues

  // Function to get human-friendly filename with duplicate numbering
  const getDisplayFilename = (source: Document) => {
    const cleanedName = cleanFilename(source.filename);

    // Count occurrences of this cleaned name
    const sameNames = sources.filter(s => cleanFilename(s.filename) === cleanedName);

    if (sameNames.length === 1) {
      return cleanedName;
    }

    // Find the index of this source among duplicates
    const index = sameNames.findIndex(s => s.id === source.id);
    return `${cleanedName} (${index + 1})`;
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>, sourceId: string) => {
    setMenuAnchorEl(event.currentTarget);
    setSelectedSourceId(sourceId);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
    setSelectedSourceId(null);
  };

  const handleDeleteClick = () => {
    if (selectedSourceId) {
      onSourceDelete(selectedSourceId);
    }
    handleMenuClose();
  };


  const handleDetailsClick = () => {
    if (selectedSourceId && onSourceDetails) {
      onSourceDetails(selectedSourceId);
    }
    handleMenuClose();
  };

  const getFileIcon = (fileType: string) => {
    const iconSize = { fontSize: '1.25rem' }; // Smaller icons
    if (fileType.includes('pdf')) return <PictureAsPdf sx={{ color: '#e53e3e', ...iconSize }} />;
    if (fileType.includes('doc')) return <Description sx={{ color: '#2b6cb0', ...iconSize }} />;
    if (fileType.includes('txt') || fileType.includes('md')) return <Article sx={{ color: '#38a169', ...iconSize }} />;
    return <InsertDriveFile sx={{ color: '#718096', ...iconSize }} />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getStatusIcon = (source: Document) => {
    const progress = documentProgress.get(source.id);
    
    if (source.status === 'completed' || source.status === 'ready') {
      return <CheckCircle sx={{ color: '#10B981', fontSize: '0.9rem' }} />;
    }
    
    if (source.status === 'error') {
      return <Error sx={{ color: '#EF4444', fontSize: '0.9rem' }} />;
    }
    
    if (source.status === 'processing' || source.status === 'uploading') {
      if (progress?.status === 'in_progress') {
        return <CircularProgress size={14} sx={{ color: '#3A6FF7' }} />;
      }
      return <Autorenew sx={{ color: '#3A6FF7', fontSize: '0.9rem' }} />;
    }

    return <Schedule sx={{ color: '#6B7280', fontSize: '0.9rem' }} />;
  };

  const getStatusText = (source: Document) => {
    const progress = documentProgress.get(source.id);

    switch (source.status) {
      case 'uploading':
        return 'Uploading file...';
      case 'processing':
        if (progress?.current_step) {
          return `${progress.current_step}...`;
        }
        return 'Processing document...';
      case 'completed':
      case 'ready':
        return 'Ready to query';
      case 'error':
        return 'Processing failed';
      default:
        return 'Queued for processing';
    }
  };

  const getStatusColor = (source: Document) => {
    switch (source.status) {
      case 'uploading':
        return '#F59E0B'; // Amber
      case 'processing':
        return '#3B82F6'; // Blue
      case 'completed':
      case 'ready':
        return '#10B981'; // Green
      case 'error':
        return '#EF4444'; // Red
      default:
        return '#6B7280'; // Gray
    }
  };

  const renderProgressBar = (source: Document) => {
    const progress = documentProgress.get(source.id);
    
    if (source.status !== 'processing' && source.status !== 'uploading') {
      return null;
    }
    
    const progressValue = progress?.progress_percentage || 0;
    const currentStep = progress?.current_step || 'Processing...';
    const statusColor = getStatusColor(source);

      return (
      <Box sx={{ mt: 1, mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <LinearProgress 
            variant={progressValue > 0 ? "determinate" : "indeterminate"}
            value={progressValue}
            sx={{ 
              flex: 1,
              height: 6,
              borderRadius: 3,
              bgcolor: 'rgba(255,255,255,0.1)',
              '& .MuiLinearProgress-bar': {
                bgcolor: statusColor,
                borderRadius: 3,
              },
            }} 
          />
          {progressValue > 0 && (
            <Typography
              variant="caption"
              sx={{
                color: statusColor,
                fontSize: '0.7rem',
                fontWeight: 600,
                minWidth: '2.5rem',
                textAlign: 'right',
              }}
            >
              {Math.round(progressValue)}%
          </Typography>
          )}
        </Box>
        <Typography
          variant="caption"
          sx={{ 
            color: 'text.secondary',
            fontSize: '0.65rem',
            display: 'block',
            textAlign: 'center',
          }}
        >
          {currentStep.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </Typography>
      </Box>
    );
  };

  return (
    <Drawer
      variant="permanent"
      anchor="left"
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: DRAWER_WIDTH,
          boxSizing: 'border-box',
          border: 'none',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)',
          backgroundColor: 'rgba(26, 26, 26, 0.95)',
          backdropFilter: 'blur(20px)',
          boxShadow: '4px 0 20px rgba(0, 0, 0, 0.3)',
        },
      }}
      className={className}
    >
      {/* Panel Header */}
      <Box sx={{ p: 3, borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
        <Typography variant="h6" sx={{ 
          fontWeight: 700, 
          color: 'white',
          fontSize: '1.25rem',
          letterSpacing: '-0.01em',
        }}>
          Sources
        </Typography>
      </Box>

      {/* Action Buttons */}
      <Box sx={{ p: 2 }}>
        {/* Start New Chat Button */}
        {onNewChat && (
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <Button
              onClick={onNewChat}
              fullWidth
              variant="outlined"
              startIcon={<Refresh />}
              sx={{
                borderColor: 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                fontWeight: 600,
                borderRadius: 3,
                py: 1.5,
                textTransform: 'none',
                fontSize: '0.95rem',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  borderColor: 'rgba(255, 255, 255, 0.3)',
                  transform: 'translateY(-1px)',
                  boxShadow: '0 4px 15px rgba(255, 255, 255, 0.1)',
                },
              }}
            >
              Start new chat
            </Button>
          </Box>
        )}

        {/* Add Source Button */}
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Button
            onClick={onAddSource}
            fullWidth
            variant="contained"
            startIcon={<Add />}
            sx={{
              background: 'linear-gradient(135deg, #3A6FF7 0%, #6366f1 100%)',
              color: 'white',
              fontWeight: 700,
              borderRadius: 3,
              py: 1.5,
              textTransform: 'none',
              fontSize: '0.95rem',
              boxShadow: '0 4px 15px rgba(58, 111, 247, 0.3)',
              border: 'none',
              '&:hover': {
                background: 'linear-gradient(135deg, #2C5AA0 0%, #4F46E5 100%)',
                transform: 'translateY(-2px)',
                boxShadow: '0 8px 25px rgba(58, 111, 247, 0.4)',
              },
            }}
          >
            Add source
          </Button>
        </Box>
      </Box>

      <Divider sx={{ bgcolor: '#404040' }} />

      {/* Sources Panel */}
      <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Available Sources */}
        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {sources.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.6 }}>
                No sources yet
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6, fontSize: '0.8rem' }}>
                Click "Add source" above to get started
              </Typography>
            </Box>
          ) : (
            <>
            <Typography
              variant="subtitle2"
              sx={{
                  fontWeight: 600,
                  color: 'text.secondary',
                  fontSize: '0.8rem',
                  mb: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
                📚 Sources ({sources.length})
            </Typography>

              {sources.map((source) => {
                const isSelected = selectedSources.some(selected => selected.id === source.id);

                return (
                  <Box
                    key={source.id}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                      gap: 1.5,
                    p: 1.5,
                      mb: 1,
                      borderRadius: 2,
                      bgcolor: isSelected ? 'rgba(16, 185, 129, 0.08)' : 'transparent',
                      border: isSelected ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(255, 255, 255, 0.08)',
                    transition: 'all 0.2s ease',
                      cursor: 'pointer',
                      '&:hover': {
                        bgcolor: isSelected ? 'rgba(16, 185, 129, 0.12)' : 'rgba(255, 255, 255, 0.04)',
                        borderColor: isSelected ? 'rgba(16, 185, 129, 0.5)' : 'rgba(58, 111, 247, 0.3)',
                        transform: 'translateY(-1px)',
                    },
                  }}
                  onClick={() => {
                    if (isSelected) {
                      onSourceDeselect?.(source.id);
                    } else {
                      onSourceSelect?.(source);
                    }
                  }}
                >
                    {/* Checkbox */}
                    <Box
                      sx={{
                        width: 18,
                        height: 18,
                        borderRadius: 1,
                        border: `2px solid ${isSelected ? '#10B981' : 'rgba(255, 255, 255, 0.3)'}`,
                        bgcolor: isSelected ? '#10B981' : 'transparent',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      {isSelected && <Typography variant="caption" sx={{ color: 'white', fontSize: '0.7rem', fontWeight: 'bold' }}>✓</Typography>}
                    </Box>

                    {/* File Icon */}
                    <Box sx={{ color: isSelected ? '#10B981' : 'text.secondary' }}>
                    {getFileIcon(source.file_type)}
                    </Box>

                    {/* File Info */}
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography
                          variant="body2"
                          sx={{
                          fontWeight: isSelected ? 600 : 500,
                          color: 'text.primary',
                          fontSize: '0.8rem',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          lineHeight: 1.3,
                        }}
                      >
                        {getDisplayFilename(source)}
                        </Typography>
                      <Typography
                        variant="caption"
                        sx={{
                          color: getStatusColor(source),
                          fontSize: '0.7rem',
                          lineHeight: 1.2,
                          fontWeight: 500,
                        }}
                      >
                        {getStatusText(source)}
                      </Typography>
                    </Box>

                    {/* Menu Button */}
                  <IconButton
                    size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleMenuClick(e, source.id);
                      }}
                    sx={{
                      color: 'text.secondary',
                        p: 0.5,
                      '&:hover': {
                          color: 'text.primary',
                        bgcolor: 'rgba(255,255,255,0.1)',
                      },
                    }}
                  >
                      <MoreVert sx={{ fontSize: '0.9rem' }} />
                  </IconButton>
                </Box>
              );
            })}
            </>
        )}
        </Box>
      </Box>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={handleMenuClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        PaperProps={{
          sx: {
            backgroundColor: '#2A2A2A',
            border: '1px solid #404040',
          },
        }}
      >
        <MenuItem onClick={handleDetailsClick}>
          <Info sx={{ mr: 1, fontSize: 18 }} />
          Details
        </MenuItem>
        <MenuItem onClick={handleDeleteClick} sx={{ color: '#e53e3e' }}>
          <Delete sx={{ mr: 1, fontSize: 18 }} />
          Remove
        </MenuItem>
      </Menu>
    </Drawer>
  );
};
