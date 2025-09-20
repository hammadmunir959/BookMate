import React, { useState, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogActions,
  Button,
  Typography,
  Box,
  Chip,
  Tab,
  Tabs,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  ExpandMore,
  Description,
  Article,
  Analytics,
  Summarize,
  Code,
  Close,
} from '@mui/icons-material';
import { Document, ChunkInfo } from '../types';
import { getDocumentDetails } from '../services/api';

interface DocumentDetails {
  success: boolean;
  document_id: string;
  filename: string;
  file_size: number;
  mime_type: string;
  total_chunks: number;
  chunks: ChunkInfo[];
  summary: Record<string, unknown> | string;
  processing_stats: Record<string, unknown>;
  steps_status: Array<Record<string, unknown>>;
  total_processing_time: number;
  citation_mode: string;
  error_message?: string;
}

interface IngestionDetailsModalProps {
  open: boolean;
  onClose: () => void;
  document: Document | null;
}

export const IngestionDetailsModal: React.FC<IngestionDetailsModalProps> = ({
  open,
  onClose,
  document,
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [detailedData, setDetailedData] = useState<DocumentDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDocumentDetails = useCallback(async () => {
    if (!document) return;

    setLoading(true);
    setError(null);

    try {
      const response = await getDocumentDetails(document.id);
      if (response.data.success) {
        setDetailedData(response.data);
      } else {
        setError(response.data.error_message || 'Failed to load document details');
      }
    } catch (err) {
      setError('Failed to fetch document details');
      console.error('Error fetching document details:', err);
    } finally {
      setLoading(false);
    }
  }, [document]);

  useEffect(() => {
    if (open && document) {
      fetchDocumentDetails();
    }
  }, [open, document, fetchDocumentDetails]);


  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const renderOverviewTab = () => {
    if (!document || !detailedData) return null;

  return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Description />
          Document Overview
        </Typography>

        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 3, mt: 2 }}>
          <Card sx={{
            bgcolor: 'background.default',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:hover': { borderColor: 'primary.main' }
          }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                File Information
              </Typography>
              <Typography variant="h6" gutterBottom sx={{ color: 'text.primary' }}>
                {detailedData.filename}
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Size: {formatFileSize(detailedData.file_size)}
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Type: {detailedData.mime_type}
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Citation Mode: {detailedData.citation_mode}
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{
            bgcolor: 'background.default',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:hover': { borderColor: 'primary.main' }
          }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Processing Status
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Chip
                label={document.status}
                  color={document.status === 'completed' ? 'success' : document.status === 'processing' ? 'warning' : 'error'}
                size="small"
                  sx={{
                    color: 'white',
                    '& .MuiChip-label': { color: 'white' }
                  }}
              />
            </Box>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Chunks: {detailedData.total_chunks}
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Upload Date: {new Date(document.upload_date).toLocaleDateString()}
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{
            bgcolor: 'background.default',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:hover': { borderColor: 'primary.main' }
          }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                  Processing Statistics
                </Typography>
              {detailedData.processing_stats && (
                <Box>
                  {Object.entries(detailedData.processing_stats).map(([key, value]) => (
                    <Typography key={key} variant="body2" sx={{ mb: 0.5, color: 'text.secondary' }}>
                      <strong style={{ color: 'text.primary' }}>{key.replace(/_/g, ' ')}:</strong> {String(value)}
                    </Typography>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
                </Box>
              </Box>
    );
  };

  const renderChunksTab = () => {
    if (!detailedData?.chunks) return null;

    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Article />
          Document Chunks ({detailedData.total_chunks} total)
        </Typography>

        <Box sx={{ mt: 2 }}>
          {detailedData.chunks.map((chunk) => (
            <Accordion
              key={chunk.chunk_id}
              sx={{
                mb: 1,
                bgcolor: 'background.default',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                '&:before': { display: 'none' },
                '&.Mui-expanded': {
                  borderColor: 'primary.main',
                }
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMore sx={{ color: 'text.secondary' }} />}
                sx={{
                  '& .MuiAccordionSummary-content': {
                    alignItems: 'center'
                  }
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                  <Code sx={{ color: 'primary.main' }} />
                  <Typography variant="subtitle2" sx={{ color: 'text.primary' }}>
                    Chunk {chunk.chunk_index + 1}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {chunk.token_count} tokens • {chunk.content.length} chars
                </Typography>
                  <Chip
                    label={`Pos: ${chunk.start_position}-${chunk.end_position}`}
                    size="small"
                    variant="outlined"
                    sx={{
                      borderColor: 'rgba(255, 255, 255, 0.3)',
                      color: 'text.secondary',
                      '& .MuiChip-label': { color: 'text.secondary' }
                    }}
                  />
                </Box>
              </AccordionSummary>
              <AccordionDetails sx={{ bgcolor: 'background.paper' }}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" sx={{
                    fontFamily: 'monospace',
                    bgcolor: 'rgba(255, 255, 255, 0.05)',
                    color: 'text.primary',
                    p: 2,
                    borderRadius: 1,
                    whiteSpace: 'pre-wrap',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    {chunk.content}
                  </Typography>
                </Box>

                {chunk.summary && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      <strong style={{ color: 'text.primary' }}>Chunk Summary:</strong>
                    </Typography>
                    <Typography variant="body2" sx={{
                      bgcolor: 'rgba(58, 111, 247, 0.1)',
                      color: 'text.primary',
                      p: 2,
                      borderRadius: 1,
                      border: '1px solid rgba(58, 111, 247, 0.2)'
                    }}>
                      {chunk.summary}
                    </Typography>
              </Box>
            )}

                {chunk.metadata && (
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      <strong style={{ color: 'text.primary' }}>Metadata:</strong>
                </Typography>
                    <Box sx={{
                      bgcolor: 'rgba(255, 255, 255, 0.03)',
                      p: 2,
                      borderRadius: 1,
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}>
                      <pre style={{
                        fontSize: '0.8rem',
                        margin: 0,
                        color: 'text.primary',
                        fontFamily: 'monospace'
                      }}>
                        {JSON.stringify(chunk.metadata, null, 2)}
                      </pre>
                    </Box>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>
                  ))}
                </Box>
      </Box>
    );
  };

  const renderSummaryTab = () => {
    if (!detailedData?.summary) return null;

    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Summarize />
          Document Summary
        </Typography>

        <Box sx={{ mt: 2 }}>
          {typeof detailedData.summary === 'string' ? (
            <Typography variant="body1" sx={{
              lineHeight: 1.6,
              color: 'text.primary',
              bgcolor: 'rgba(255, 255, 255, 0.03)',
              p: 3,
              borderRadius: 1,
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}>
              {detailedData.summary}
            </Typography>
          ) : (
            <Box sx={{
              bgcolor: 'rgba(255, 255, 255, 0.03)',
              p: 3,
              borderRadius: 1,
              border: '1px solid rgba(255, 255, 255, 0.1)'
            }}>
              <pre style={{
                margin: 0,
                whiteSpace: 'pre-wrap',
                fontFamily: 'monospace',
                color: 'text.primary',
                fontSize: '0.9rem'
              }}>
                {JSON.stringify(detailedData.summary, null, 2)}
              </pre>
              </Box>
            )}
        </Box>
      </Box>
    );
  };

  const renderAnalyticsTab = () => {
    if (!detailedData) return null;

    const totalTokens = detailedData.chunks?.reduce((sum, chunk) => sum + chunk.token_count, 0) || 0;
    const avgChunkSize = detailedData.chunks?.length ? Math.round(totalTokens / detailedData.chunks.length) : 0;

    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Analytics />
          Analytics & Statistics
        </Typography>

        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 3, mt: 2 }}>
          <Card sx={{
            bgcolor: 'background.default',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:hover': { borderColor: 'primary.main' }
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                {detailedData.total_chunks}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Chunks
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{
            bgcolor: 'background.default',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:hover': { borderColor: 'primary.main' }
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                {totalTokens.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Tokens
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{
            bgcolor: 'background.default',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:hover': { borderColor: 'primary.main' }
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                {avgChunkSize}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Tokens/Chunk
              </Typography>
            </CardContent>
          </Card>

          <Card sx={{
            bgcolor: 'background.default',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:hover': { borderColor: 'primary.main' }
          }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                {formatFileSize(detailedData.file_size)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                File Size
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{ color: 'text.primary' }}>
            Chunk Distribution
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {detailedData.chunks?.map((chunk, _index) => (
              <Chip
                key={_index}
                label={`${chunk.token_count}t`}
                size="small"
                variant="outlined"
                sx={{
                  bgcolor: chunk.token_count > avgChunkSize ? 'rgba(34, 197, 94, 0.2)' : 'rgba(251, 191, 36, 0.2)',
                  borderColor: chunk.token_count > avgChunkSize ? 'rgba(34, 197, 94, 0.4)' : 'rgba(251, 191, 36, 0.4)',
                  color: chunk.token_count > avgChunkSize ? '#22c55e' : '#f59e0b',
                  '& .MuiChip-label': {
                    color: chunk.token_count > avgChunkSize ? '#22c55e' : '#f59e0b'
                  },
                  '&:hover': {
                    bgcolor: chunk.token_count > avgChunkSize ? 'rgba(34, 197, 94, 0.3)' : 'rgba(251, 191, 36, 0.3)',
                  }
                }}
              />
            ))}
          </Box>
        </Box>
      </Box>
    );
  };

  if (!document) return null;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          minHeight: '80vh',
          maxHeight: '90vh',
          bgcolor: 'background.paper',
          color: 'text.primary',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)',
        }
      }}
    >
      <DialogTitle sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        bgcolor: 'background.paper',
        color: 'text.primary',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <Typography variant="h6" sx={{ color: 'text.primary' }}>
          Document Details: {document.filename}
        </Typography>
        <IconButton onClick={onClose} sx={{ color: 'text.secondary' }}>
          <Close />
        </IconButton>
      </DialogTitle>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress sx={{ color: 'primary.main' }} />
          <Typography sx={{ ml: 2, color: 'text.primary' }}>Loading document details...</Typography>
        </Box>
      )}

      {error && (
        <Box sx={{ p: 3 }}>
          <Alert
            severity="error"
            sx={{
              bgcolor: 'rgba(239, 68, 68, 0.1)',
              color: 'text.primary',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              '& .MuiAlert-icon': { color: '#ef4444' }
            }}
          >
            {error}
          </Alert>
        </Box>
      )}

      {!loading && !error && detailedData && (
        <>
          <Box sx={{
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
            bgcolor: 'background.default'
          }}>
            <Tabs
              value={activeTab}
              onChange={(_, newValue) => setActiveTab(newValue)}
              sx={{
                '& .MuiTab-root': {
                  color: 'text.secondary',
                  '&.Mui-selected': {
                    color: 'primary.main',
                  },
                },
                '& .MuiTabs-indicator': {
                  bgcolor: 'primary.main',
                },
              }}
            >
              <Tab label="Overview" />
              <Tab label={`Chunks (${detailedData.total_chunks})`} />
              <Tab label="Summary" />
              <Tab label="Analytics" />
            </Tabs>
          </Box>

          <Box sx={{ flex: 1, overflow: 'auto' }}>
            {activeTab === 0 && renderOverviewTab()}
            {activeTab === 1 && renderChunksTab()}
            {activeTab === 2 && renderSummaryTab()}
            {activeTab === 3 && renderAnalyticsTab()}
          </Box>
        </>
      )}

      <DialogActions sx={{
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        bgcolor: 'background.paper',
        p: 2
      }}>
        <Button
          onClick={onClose}
          sx={{
            color: 'text.primary',
            borderColor: 'rgba(255, 255, 255, 0.3)',
            '&:hover': {
              bgcolor: 'rgba(255, 255, 255, 0.05)',
              borderColor: 'rgba(255, 255, 255, 0.5)'
            }
          }}
          variant="outlined"
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};
