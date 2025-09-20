import { useState, useEffect } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, Box } from '@mui/material';
import { GlobalHeader } from './components/GlobalHeader';
import { SourcesPanel } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';
import { AddSourceModal } from './components/AddSourceModal';
import { IngestionDetailsModal } from './components/IngestionDetailsModal';
import { getHealthCheck, ingestDocument, processQuery, deleteDocument, listDocuments, resetSession } from './services/api';
import { cleanFilename } from './utils/security';
import { UIState, Document, ChatMessage, CitationInfo } from './types';
import { UI_CONFIG } from './config/constants';

// Create theme - Modern dark mode with professional styling
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3A6FF7', // Modern blue
      light: '#5B8AFF',
      dark: '#2C5AA0',
    },
    secondary: {
      main: '#6366f1',
      light: '#8B5CF6',
      dark: '#4F46E5',
    },
    background: {
      default: '#0F0F0F', // Deeper dark background
      paper: '#1A1A1A',   // Slightly lighter panels
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#B8BCC8',
    },
    grey: {
      50: '#F8FAFC',
      100: '#F1F5F9',
      200: '#E2E8F0',
      300: '#CBD5E1',
      400: '#94A3B8',
      500: '#64748B',
      600: '#475569',
      700: '#334155',
      800: '#1E293B',
      900: '#0F172A',
    },
  },
  typography: {
    fontFamily: '"Inter", "SF Pro Display", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.3,
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.4,
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.4,
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.125rem',
      lineHeight: 1.4,
    },
    h6: {
      fontWeight: 600,
      fontSize: '1rem',
      lineHeight: 1.4,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          margin: 0,
          padding: 0,
          backgroundColor: '#0F0F0F',
          color: '#FFFFFF',
          backgroundImage: `
            radial-gradient(circle at 20% 80%, rgba(58, 111, 247, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(139, 92, 246, 0.05) 0%, transparent 50%)
          `,
          backgroundAttachment: 'fixed',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#1A1A1A',
          color: '#FFFFFF',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 12,
          fontWeight: 600,
          padding: '12px 24px',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 25px rgba(0, 0, 0, 0.3)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #3A6FF7 0%, #6366f1 100%)',
          boxShadow: '0 4px 15px rgba(58, 111, 247, 0.3)',
          '&:hover': {
            background: 'linear-gradient(135deg, #2C5AA0 0%, #4F46E5 100%)',
            boxShadow: '0 8px 25px rgba(58, 111, 247, 0.4)',
          },
        },
        outlined: {
          border: '2px solid rgba(255, 255, 255, 0.2)',
          '&:hover': {
            border: '2px solid rgba(255, 255, 255, 0.4)',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
        },
      },
    },
  },
});

function App() {
  const [uiState, setUIState] = useState<UIState>({
    sidebarOpen: true,
    rightPanelOpen: false, // No right panel in NotebookLM
    isLoading: true,
    theme: 'dark',
  });

  const [sources, setSources] = useState<Document[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAddSourceModalOpen, setIsAddSourceModalOpen] = useState(false);
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [isDetailsModalOpen, setIsDetailsModalOpen] = useState(false);
  const [selectedSources, setSelectedSources] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);

  // Check backend health and load documents on mount
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check backend health
        await getHealthCheck();

        // Load documents from backend
        const { data } = await listDocuments();
        console.log('Loaded documents from backend:', data);

        if (data.documents && data.documents.length > 0) {
          // Convert backend format to UI format
          const backendDocs = data.documents.map((doc: {
            id: string;
            filename: string;
            file_type: string;
            file_size: number;
            upload_date: string;
            notebook_id?: string;
            chunks_count?: number;
            status?: string;
            mime_type?: string;
          }) => ({
            id: doc.id,
            filename: doc.filename,
            original_filename: doc.filename,
            file_type: doc.file_type,
            file_size: doc.file_size,
            upload_date: doc.upload_date,
            notebook_id: doc.notebook_id || 'current',
            chunks_count: doc.chunks_count || 0,
            status: (doc.status === 'completed' ? 'ready' : doc.status || 'ready') as Document['status'],
            mime_type: doc.mime_type,
          }));

          setSources(backendDocs);
          console.log('Documents loaded and state updated');
        }

        setUIState(prev => ({ ...prev, isLoading: false }));
      } catch (error) {
        console.error('App initialization failed:', error);
        setUIState(prev => ({
          ...prev,
          isLoading: false,
          error: 'Unable to connect to backend server or load documents'
        }));
      }
    };

    initializeApp();
  }, []);



  const handleSettingsClick = () => {
    // TODO: Implement settings
    console.log('Open settings');
  };

  const handleThemeToggle = () => {
    // TODO: Implement theme toggle
    console.log('Toggle theme');
  };

  const handleAddSource = () => {
    setIsAddSourceModalOpen(true);
  };

  const handleSourceDelete = async (sourceId: string) => {
    try {
      // Delete from backend
      await deleteDocument(sourceId);
      console.log('Document deleted from backend:', sourceId);

      // Remove from local state
      setSources(prev => prev.filter(source => source.id !== sourceId));

      // Clear any chat messages that might reference this document
      setMessages(prev => prev.filter(message => {
        // Keep system messages and user messages
        if (message.type === 'system' || message.type === 'user') {
          return true;
        }
        
        // For assistant messages, check if they reference the deleted document
        if (message.metadata?.citations) {
          const hasReference = message.metadata.citations.some((citation: CitationInfo) =>
            citation.document_id === sourceId
          );
          return !hasReference;
        }
        
        return true;
      }));

      console.log('Document and related chat context cleared:', sourceId);

    } catch (error) {
      console.error('Failed to delete document:', error);
      // TODO: Show user-friendly error message
    }
  };

  const handleSourceSelect = (source: Document) => {
    setSelectedSources(prev => {
      const isAlreadySelected = prev.some(selected => selected.id === source.id);
      if (isAlreadySelected) {
        return prev.filter(selected => selected.id !== source.id); // Remove if already selected (deselect)
      }
      return [...prev, source]; // Add if not selected
    });
  };

  const handleSourceDeselect = (sourceId: string) => {
    setSelectedSources(prev => prev.filter(source => source.id !== sourceId));
  };

  const handleNewChat = async () => {
    try {
      // Clear all messages
      setMessages([]);

      // Deselect all sources (but keep them available)
      setSelectedSources([]);

      // Call backend to reset conversation history
      await resetSession();

      console.log('New chat started: messages cleared, sources deselected, session reset');
    } catch (error) {
      console.error('Failed to start new chat:', error);
      // TODO: Show user-friendly error message
    }
  };

  const handleSourceReplace = (sourceId: string) => {
    // TODO: Implement source replacement
    console.log('Replace source:', sourceId);
  };

  const handleSourceDetails = (sourceId: string) => {
    const document = sources.find(source => source.id === sourceId);
    if (document) {
      setSelectedDocument(document);
      setIsDetailsModalOpen(true);
    }
  };

  const handleFilesSelected = async (files: File[]) => {
    setIsAddSourceModalOpen(false); // Close modal immediately

    try {
      // Use a counter to prevent race conditions in temp ID generation
      let tempIdCounter = 0;
      const uploadPromises = files.map(async (file) => {
        // Create initial source entry with uploading status
        const tempId = `temp-${Date.now()}-${tempIdCounter++}`;
        const cleanedFilename = cleanFilename(file.name);
        const tempSource: Document = {
          id: tempId,
          filename: cleanedFilename,
          original_filename: file.name,
          file_type: file.type || 'unknown',
          file_size: file.size,
          upload_date: new Date().toISOString(),
          notebook_id: 'current',
          chunks_count: 0,
          status: 'uploading' as const,
        };

        // Add to sources list immediately
        setSources(prev => [...prev, tempSource]);

        try {
          // Upload to backend using new ingestion endpoint
          const { data } = await ingestDocument(file, {}, 'user123');
          console.log('Ingestion response:', data);

          // Update source with real data from backend
          // Use cleaned filename from backend, or clean it if not provided
          const finalFilename = data.filename ? cleanFilename(data.filename) : cleanedFilename;
          const updatedSource: Document = {
            id: data.document_id,
            filename: finalFilename,
            original_filename: file.name,
            file_type: data.mime_type || 'unknown',
            file_size: data.file_size,
            upload_date: new Date().toISOString(),
            notebook_id: 'current',
            chunks_count: data.total_chunks,
            status: 'completed' as const,
            mime_type: data.mime_type,
            chunks: data.chunks,
            summary: data.summary,
            processing_stats: data.processing_stats,
            steps_status: data.steps_status,
            total_processing_time: data.total_processing_time,
            citation_mode: data.citation_mode,
          };

          // Replace temp source with real source
          setSources(prev =>
            prev.map(source =>
              source.id === tempId ? updatedSource : source
            )
          );

          return updatedSource;
        } catch (error) {
          console.error('Upload failed:', error);

          // Update source with error status
          const errorSource: Document = {
            ...tempSource,
            status: 'error' as const,
          };

          setSources(prev =>
            prev.map(source =>
              source.id === tempId ? errorSource : source
            )
          );

          throw error;
        }
      });

      await Promise.all(uploadPromises);
      console.log('All files uploaded successfully');

    } catch (error) {
      console.error('File upload failed:', error);
      // Error is already handled above for individual files
    }
  };

  const handleTextPasted = async (text: string, title?: string) => {
    setIsAddSourceModalOpen(false); // Close modal immediately

    try {
      // Create a text file from the pasted content
      const textBlob = new Blob([text], { type: 'text/plain' });
      const rawFilename = `${title || 'Pasted Text'}.txt`;
      const cleanedFilename = cleanFilename(rawFilename);
      const textFile = new File([textBlob], cleanedFilename, { type: 'text/plain' });

      // Create initial source entry with uploading status (use counter for uniqueness)
      const tempId = `temp-${Date.now()}-text`;
      const tempSource: Document = {
        id: tempId,
        filename: cleanedFilename,
        original_filename: rawFilename,
        file_type: 'text/plain',
        file_size: textBlob.size,
        upload_date: new Date().toISOString(),
        notebook_id: 'current',
        chunks_count: 0,
        status: 'uploading' as const,
      };

      // Add to sources list immediately
      setSources(prev => [...prev, tempSource]);

      try {
        // Upload to backend using new ingestion endpoint
        const { data } = await ingestDocument(textFile, {}, 'user123');
        console.log('Text ingestion response:', data);

        // Update source with real data from backend
        const finalFilename = data.filename ? cleanFilename(data.filename) : cleanedFilename;
        const updatedSource: Document = {
          id: data.document_id,
          filename: finalFilename,
          original_filename: rawFilename,
          file_type: 'text/plain',
          file_size: data.file_size,
          upload_date: new Date().toISOString(),
          notebook_id: 'current',
          chunks_count: data.total_chunks,
          status: 'completed' as const,
          mime_type: data.mime_type,
          chunks: data.chunks,
          summary: data.summary,
          processing_stats: data.processing_stats,
          steps_status: data.steps_status,
          total_processing_time: data.total_processing_time,
          citation_mode: data.citation_mode,
        };

        // Replace temp source with real source
        setSources(prev =>
          prev.map(source =>
            source.id === tempId ? updatedSource : source
          )
        );

        console.log('Text uploaded successfully');

      } catch (error) {
        console.error('Text upload failed:', error);

        // Update source with error status
        const errorSource: Document = {
          ...tempSource,
          status: 'error' as const,
        };

        setSources(prev =>
          prev.map(source =>
            source.id === tempId ? errorSource : source
          )
        );

        throw error;
      }

    } catch (error) {
      console.error('Text processing failed:', error);
      // TODO: Show user-friendly error message
    }
  };

  const handleSendMessage = async (message: string) => {
    setIsSendingMessage(true);

    // Add user message immediately
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      type: 'user',
      content: message,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      // Send query to backend using new format
      const queryRequest = {
        query: message,
        document_ids: selectedSources.length > 0 ? selectedSources.map(source => source.id) : undefined,
        system_prompt: "You are a helpful AI assistant. Answer based on the provided document context and provide accurate citations.",
        context: "Please provide a comprehensive answer with relevant citations from the documents.",
        max_tokens: 2000,
        temperature: 0.1,
        top_k: 5,
        similarity_threshold: 0.2,
        conversation_history: messages.slice(-6).map(msg => ({ // Last 6 messages for context
          role: msg.type === 'user' ? 'user' : 'assistant',
          content: msg.content
        }))
      };

      const { data } = await processQuery(queryRequest);
      console.log('Query response:', data);

      // Add assistant message from backend
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        type: 'assistant',
        content: data.answer,
        timestamp: new Date().toISOString(),
        metadata: {
          citations: data.citations || [],
          total_chunks_retrieved: data.total_chunks_retrieved,
          processing_stats: data.processing_stats,
          model_used: data.model_used,
          confidence_score: data.confidence_score,
          // processing_time: data.processing_time, // Already included in processing_stats
        },
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Query failed:', error);

      // Add error message
      const errorMessage: ChatMessage = {
        id: `msg-${Date.now() + 2}`,
        type: 'assistant',
        content: `Sorry, I encountered an error while processing your query: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        timestamp: new Date().toISOString(),
        metadata: {
          sources: [],
          suggestions: [],
          confidence_score: 0,
        },
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsSendingMessage(false);
    }
  };


  if (uiState.isLoading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          display="flex"
          justifyContent="center"
          alignItems="center"
          minHeight="100vh"
          bgcolor="background.default"
        >
          Loading BookMate...
        </Box>
      </ThemeProvider>
    );
  }

  if (uiState.error) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          display="flex"
          flexDirection="column"
          justifyContent="center"
          alignItems="center"
          minHeight="100vh"
          bgcolor="background.default"
          p={3}
        >
          <Box mb={2}>⚠️ Connection Error</Box>
          <Box color="text.secondary" textAlign="center">
            {uiState.error}
            <br />
            Please make sure the backend server is running on http://localhost:8000
          </Box>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router
        future={{
          v7_startTransition: true,
          v7_relativeSplatPath: true,
        }}
      >
        <Box display="flex" flexDirection="column" minHeight="100vh" bgcolor="background.default" sx={{ pt: '64px' }}>
          {/* Global Header */}
          <GlobalHeader
            sourceCount={sources.length}
            selectedSourcesCount={selectedSources.length}
            onSettingsClick={handleSettingsClick}
            onThemeToggle={handleThemeToggle}
            isDarkTheme={true}
          />

          {/* Main Content Area */}
          <Box display="flex" flex={1} overflow="hidden">
            {/* Sources Panel */}
            <SourcesPanel
              sources={sources}
              onAddSource={handleAddSource}
              onNewChat={handleNewChat}
              onSourceDelete={handleSourceDelete}
              onSourceReplace={handleSourceReplace}
              onSourceDetails={handleSourceDetails}
              selectedSources={selectedSources}
              onSourceSelect={handleSourceSelect}
              onSourceDeselect={handleSourceDeselect}
            />

            {/* Chat Interface */}
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
              <ChatInterface
                messages={messages}
                isLoading={isSendingMessage}
                onSendMessage={handleSendMessage}
                sourceCount={sources.length}
                sources={sources}
              />
            </Box>
          </Box>

          {/* Add Source Modal */}
          <AddSourceModal
            open={isAddSourceModalOpen}
            onClose={() => setIsAddSourceModalOpen(false)}
            onFilesSelected={handleFilesSelected}
            onTextPasted={handleTextPasted}
          maxFiles={UI_CONFIG.MAX_FILES_COUNT}
          maxSize={UI_CONFIG.MAX_FILE_SIZE_MB}
          />

          {/* Ingestion Details Modal */}
          <IngestionDetailsModal
            open={isDetailsModalOpen}
            onClose={() => {
              setIsDetailsModalOpen(false);
              setSelectedDocument(null);
            }}
            document={selectedDocument}
          />
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;