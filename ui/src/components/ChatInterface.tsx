import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import {
  Box,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Chip,
  CircularProgress,
  Collapse,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
} from '@mui/material';
import {
  Send,
  SmartToy,
  Lightbulb,
  AccessTime,
  Close,
} from '@mui/icons-material';
import type { ChatInterfaceProps, ChatMessage, CitationInfo } from '../types';
import { validateMessage } from '../utils/security';
import { VALIDATION_CONFIG } from '../config/constants';

// Animated Scene Background Component
const AnimatedSceneBackground = () => {
  // Beautiful animated study atmosphere with CSS animations
  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1,
        overflow: 'hidden',
        pointerEvents: 'none',
        // Warm study room atmosphere with multiple layered gradients
        background: `
          radial-gradient(ellipse at 25% 30%, rgba(139, 92, 246, 0.08) 0%, transparent 60%),
          radial-gradient(ellipse at 75% 70%, rgba(59, 130, 246, 0.06) 0%, transparent 55%),
          radial-gradient(circle at 50% 50%, rgba(245, 158, 11, 0.04) 0%, transparent 70%),
          linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(59, 130, 246, 0.03) 40%, rgba(16, 185, 129, 0.04) 70%, rgba(245, 158, 11, 0.02) 100%)
        `,
        // Multiple floating elements simulating study scene
        '& .desk': {
          position: 'absolute',
          bottom: '15%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '200px',
          height: '20px',
          background: 'linear-gradient(90deg, rgba(101, 67, 33, 0.3) 0%, rgba(139, 69, 19, 0.4) 50%, rgba(101, 67, 33, 0.3) 100%)',
          borderRadius: '5px',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
        },
        '& .book': {
          position: 'absolute',
          bottom: '20%',
          left: '45%',
          width: '80px',
          height: '100px',
          background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.3) 100%)',
          border: '1px solid rgba(139, 92, 246, 0.4)',
          borderRadius: '2px',
          animation: 'bookFlip 8s infinite ease-in-out',
          '@keyframes bookFlip': {
            '0%, 100%': { transform: 'rotateY(0deg) scale(1)' },
            '25%': { transform: 'rotateY(-5deg) scale(1.02)' },
            '50%': { transform: 'rotateY(0deg) scale(1)' },
            '75%': { transform: 'rotateY(3deg) scale(0.98)' },
          },
          '&::before': {
            content: '""',
            position: 'absolute',
            top: '20%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '60%',
            height: '2px',
            background: 'rgba(59, 130, 246, 0.6)',
            borderRadius: '1px',
          },
          '&::after': {
            content: '""',
            position: 'absolute',
            top: '35%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '70%',
            height: '2px',
            background: 'rgba(59, 130, 246, 0.4)',
            borderRadius: '1px',
          },
        },
        '& .mug': {
          position: 'absolute',
          bottom: '22%',
          right: '35%',
          width: '30px',
          height: '35px',
          background: 'linear-gradient(145deg, rgba(245, 158, 11, 0.3) 0%, rgba(217, 119, 6, 0.4) 100%)',
          border: '1px solid rgba(245, 158, 11, 0.5)',
          borderRadius: '2px 2px 8px 8px',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: '-5px',
            left: '20%',
            width: '60%',
            height: '8px',
            background: 'rgba(245, 158, 11, 0.4)',
            borderRadius: '50%',
            animation: 'steam 4s infinite ease-in-out',
            '@keyframes steam': {
              '0%, 100%': { transform: 'translateY(0px) scaleY(1)', opacity: 0.6 },
              '50%': { transform: 'translateY(-8px) scaleY(0.8)', opacity: 0.3 },
            },
          },
        },
        '& .curtain': {
          position: 'absolute',
          top: '10%',
          right: '10%',
          width: '80px',
          height: '120px',
          background: 'linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.3) 100%)',
          borderRadius: '2px',
          animation: 'curtainWave 6s infinite ease-in-out 1s',
          '@keyframes curtainWave': {
            '0%, 100%': { transform: 'skewX(0deg)' },
            '25%': { transform: 'skewX(1deg)' },
            '50%': { transform: 'skewX(0deg)' },
            '75%': { transform: 'skewX(-0.5deg)' },
          },
        },
        // Multiple floating particles (dust in light beams)
        '& .particle-1': {
          position: 'absolute',
          width: '3px',
          height: '3px',
          background: 'rgba(255, 255, 255, 0.4)',
          borderRadius: '50%',
          top: '25%',
          left: '30%',
          animation: 'float1 10s infinite ease-in-out',
        },
        '& .particle-2': {
          position: 'absolute',
          width: '2px',
          height: '2px',
          background: 'rgba(255, 255, 255, 0.5)',
          borderRadius: '50%',
          top: '40%',
          right: '25%',
          animation: 'float2 8s infinite ease-in-out 2s',
        },
        '& .particle-3': {
          position: 'absolute',
          width: '4px',
          height: '4px',
          background: 'rgba(255, 255, 255, 0.3)',
          borderRadius: '50%',
          top: '35%',
          left: '60%',
          animation: 'float3 12s infinite ease-in-out 4s',
        },
        '@keyframes float1': {
          '0%, 100%': { transform: 'translateY(0px) translateX(0px)', opacity: 0.4 },
          '25%': { transform: 'translateY(-15px) translateX(8px)', opacity: 0.7 },
          '50%': { transform: 'translateY(-8px) translateX(-5px)', opacity: 0.5 },
          '75%': { transform: 'translateY(-20px) translateX(3px)', opacity: 0.3 },
        },
        '@keyframes float2': {
          '0%, 100%': { transform: 'translateY(0px) translateX(0px)', opacity: 0.5 },
          '30%': { transform: 'translateY(-12px) translateX(-6px)', opacity: 0.8 },
          '60%': { transform: 'translateY(-18px) translateX(4px)', opacity: 0.4 },
          '90%': { transform: 'translateY(-10px) translateX(-2px)', opacity: 0.6 },
        },
        '@keyframes float3': {
          '0%, 100%': { transform: 'translateY(0px) translateX(0px)', opacity: 0.3 },
          '20%': { transform: 'translateY(-10px) translateX(7px)', opacity: 0.6 },
          '40%': { transform: 'translateY(-16px) translateX(-4px)', opacity: 0.4 },
          '70%': { transform: 'translateY(-12px) translateX(6px)', opacity: 0.5 },
          '90%': { transform: 'translateY(-8px) translateX(-3px)', opacity: 0.7 },
        },
      }}
    >
      {/* Scene Elements */}
      <Box className="desk" />
      <Box className="book" />
      <Box className="mug" />
      <Box className="curtain" />
      <Box className="particle-1" />
      <Box className="particle-2" />
      <Box className="particle-3" />
    </Box>
  );
};

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  isLoading,
  onSendMessage,
  sourceCount = 0,
  sources = [], // Add sources prop for better contextual placeholders
}) => {
  // Function to get citations for a specific message
  const getMessageCitations = useCallback((message: ChatMessage) => {
    return message.metadata?.citations || [];
  }, []);
  const [inputValue, setInputValue] = useState('');
  const [isInputExpanded, setIsInputExpanded] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState<{[key: string]: boolean}>({});
  const [expandedSources, setExpandedSources] = useState<{[key: string]: boolean}>({});
  const [citationModal, setCitationModal] = useState<{
    open: boolean;
    citation: CitationInfo | null;
    citationIndex: number;
  }>({ open: false, citation: null, citationIndex: -1 });
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Generate contextual placeholder based on available sources
  const getContextualPlaceholder = useCallback(() => {
    return "Ask about your selected sources...";
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const toggleReasoning = useCallback((messageId: string) => {
    setExpandedReasoning(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  }, []);

  const toggleSources = useCallback((messageId: string) => {
    setExpandedSources(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  }, []);

  const formatRelativeTime = useCallback((timestamp: string) => {
    const now = new Date();
    const messageTime = new Date(timestamp);
    const diffInSeconds = Math.floor((now.getTime() - messageTime.getTime()) / 1000);

    if (diffInSeconds < 60) return 'now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return messageTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const validateAndSendMessage = useCallback(() => {
    const validation = validateMessage(inputValue);

    if (!validation.isValid) {
      console.warn('Message validation failed:', validation.error);
      // TODO: Show user-friendly error message in UI
      return;
    }

    if (!isLoading) {
      onSendMessage(validation.sanitizedMessage);
      setInputValue('');
    }
  }, [inputValue, isLoading, onSendMessage]);

  const handleSend = useCallback(() => {
    validateAndSendMessage();
  }, [validateAndSendMessage]);

  const handleKeyPress = useCallback((event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  }, []);

  const handleCitationClick = useCallback((citation: CitationInfo, index: number) => {
    setCitationModal({
      open: true,
      citation,
      citationIndex: index
    });
  }, []);

  const handleCloseCitationModal = useCallback(() => {
    setCitationModal({ open: false, citation: null, citationIndex: -1 });
  }, []);

  // Function to parse and render text with clickable citations
  const renderTextWithCitations = useCallback((text: string, citations: CitationInfo[]) => {
    // Regular expression to match citation patterns like (cit#1), (cit#2), etc.
    const citationRegex = /\(cit#(\d+)\)/g;

    // Split text by citations and create parts
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = citationRegex.exec(text)) !== null) {
      // Add text before citation
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.slice(lastIndex, match.index)
        });
      }

      // Add clickable citation
      const citationIndex = parseInt(match[1]) - 1; // Convert to 0-based index
      const citation = citations[citationIndex];

      parts.push({
        type: 'citation',
        content: match[0],
        citationIndex: citationIndex,
        citation: citation
      });

      lastIndex = citationRegex.lastIndex;
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.slice(lastIndex)
      });
    }

    return parts;
  }, []);

  // Component to render text with clickable citations
  const CitationTextRenderer = useCallback(({ text, citations }: { text: string; citations: CitationInfo[] }) => {
    // Create a mapping of citation markers to React components
    const citationComponents: Record<string, React.ReactElement> = {};

    citations.forEach((citation, index) => {
      const marker = `(cit#${index + 1})`;
      citationComponents[marker] = (
        <Tooltip
          key={`citation-${index}`}
          title={
            citation ? (
              <Box sx={{ p: 1, maxWidth: 300 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  {citation.display_name} • Chunk {citation.chunk_number}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Relevance: {(citation.relevance_score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  {citation.content.substring(0, 120)}...
                </Typography>
              </Box>
            ) : (
              <Typography variant="body2">Citation data not available</Typography>
            )
          }
          placement="top"
          arrow
        >
          <Chip
            label={marker}
            size="small"
            onClick={() => handleCitationClick(citation, index)}
            sx={{
              bgcolor: 'rgba(34, 197, 94, 0.15)',
              color: '#22C55E',
              border: '1px solid rgba(34, 197, 94, 0.4)',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s',
              mx: 0.25,
              my: 0.25,
              '&:hover': {
                bgcolor: 'rgba(34, 197, 94, 0.25)',
                transform: 'translateY(-1px)',
                boxShadow: '0 2px 4px rgba(34, 197, 94, 0.2)',
              },
            }}
          />
        </Tooltip>
      );
    });

    // Create a custom text processing function that handles citations within text
    const processTextWithCitations = (text: string): (string | React.ReactElement)[] => {
      if (typeof text !== 'string') return [text];

      const parts: (string | React.ReactElement)[] = [];
      const regex = /(\(cit#\d+\))/g;
      let lastIndex = 0;
      let match;

      while ((match = regex.exec(text)) !== null) {
        // Add text before citation
        if (match.index > lastIndex) {
          parts.push(text.slice(lastIndex, match.index));
        }

        // Add citation component
        const citationKey = match[0];
        if (citationComponents[citationKey]) {
          parts.push(React.cloneElement(citationComponents[citationKey], {
            key: `citation-${match.index}`
          }));
        } else {
          parts.push(match[0]); // Fallback to text if component not found
        }

        lastIndex = regex.lastIndex;
      }

      // Add remaining text
      if (lastIndex < text.length) {
        parts.push(text.slice(lastIndex));
      }

      return parts.length > 0 ? parts : [text];
    };

    // Create custom markdown components that can handle citations
    const markdownComponents: Components = {
      // Override the text processing for all elements
      p: ({ children, ...props }) => (
        <p {...props}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </p>
      ),
      li: ({ children, ...props }) => (
        <li {...props}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </li>
      ),
      strong: ({ children, ...props }) => (
        <strong {...props}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </strong>
      ),
      em: ({ children, ...props }) => (
        <em {...props}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </em>
      ),
      code: ({ children, ...props }) => (
        <code {...props}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </code>
      ),
      // Add header components
      h1: ({ children, ...props }) => (
        <h1 {...props} style={{
          fontSize: '1.5rem',
          fontWeight: 700,
          margin: '1rem 0 0.5rem 0',
          color: 'rgba(255,255,255,0.95)',
          borderBottom: '2px solid rgba(58, 111, 247, 0.3)',
          paddingBottom: '0.25rem'
        }}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </h1>
      ),
      h2: ({ children, ...props }) => (
        <h2 {...props} style={{
          fontSize: '1.25rem',
          fontWeight: 600,
          margin: '1rem 0 0.5rem 0',
          color: 'rgba(255,255,255,0.9)',
          borderBottom: '1px solid rgba(58, 111, 247, 0.2)',
          paddingBottom: '0.25rem'
        }}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </h2>
      ),
      h3: ({ children, ...props }) => (
        <h3 {...props} style={{
          fontSize: '1.1rem',
          fontWeight: 600,
          margin: '0.75rem 0 0.25rem 0',
          color: 'rgba(255,255,255,0.85)',
        }}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </h3>
      ),
      // Add blockquote component
      blockquote: ({ children, ...props }) => (
        <blockquote {...props} style={{
          borderLeft: '4px solid rgba(58, 111, 247, 0.4)',
          paddingLeft: '1rem',
          margin: '1rem 0',
          backgroundColor: 'rgba(58, 111, 247, 0.05)',
          padding: '0.5rem 1rem',
          borderRadius: '0 4px 4px 0',
          fontStyle: 'italic',
          color: 'rgba(255,255,255,0.8)',
          wordBreak: 'break-word',
        }}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </blockquote>
      ),
      // Enhanced list components
      ul: ({ children, ...props }) => (
        <ul {...props} style={{
          margin: '0.75rem 0',
          paddingLeft: '1.5rem',
          wordBreak: 'break-word',
        }}>
          {children}
        </ul>
      ),
      ol: ({ children, ...props }) => (
        <ol {...props} style={{
          margin: '0.75rem 0',
          paddingLeft: '1.5rem',
          wordBreak: 'break-word',
        }}>
          {children}
        </ol>
      ),
      // Table components
      table: ({ children, ...props }) => (
        <table {...props} style={{
          borderCollapse: 'collapse',
          margin: '1rem 0',
          width: '100%',
          fontSize: '0.9em',
          wordBreak: 'break-word',
        }}>
          {children}
        </table>
      ),
      thead: ({ children, ...props }) => (
        <thead {...props} style={{
          backgroundColor: 'rgba(58, 111, 247, 0.1)',
        }}>
          {children}
        </thead>
      ),
      tbody: ({ children, ...props }) => (
        <tbody {...props}>
          {children}
        </tbody>
      ),
      tr: ({ children, ...props }) => (
        <tr {...props} style={{
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}>
          {children}
        </tr>
      ),
      th: ({ children, ...props }) => (
        <th {...props} style={{
          padding: '0.5rem',
          textAlign: 'left',
          fontWeight: 600,
          color: 'rgba(255,255,255,0.9)',
          borderRight: '1px solid rgba(255,255,255,0.1)',
          wordBreak: 'break-word',
        }}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </th>
      ),
      td: ({ children, ...props }) => (
        <td {...props} style={{
          padding: '0.5rem',
          borderRight: '1px solid rgba(255,255,255,0.1)',
          wordBreak: 'break-word',
        }}>
          {React.Children.map(children, (child) => {
            if (typeof child === 'string') {
              return processTextWithCitations(child);
            }
            return child;
          })}
        </td>
      ),
    };

    return (
      <Box
        sx={{
          '& p': {
            margin: '0.5rem 0',
            wordBreak: 'break-word',
            whiteSpace: 'pre-wrap',
          },
          '& p:first-child': { marginTop: 0 },
          '& p:last-child': { marginBottom: 0 },
          '& strong, & b': {
            fontWeight: 600,
            color: 'rgba(255,255,255,0.95)',
          },
          '& em, & i': {
            fontStyle: 'italic',
            opacity: 0.9,
          },
          '& code': {
            bgcolor: 'rgba(0,0,0,0.3)',
            px: 0.5,
            py: 0.25,
            borderRadius: 0.5,
            fontSize: '0.85em',
            fontFamily: 'Monaco, Consolas, monospace',
            wordBreak: 'break-all',
            whiteSpace: 'pre-wrap',
          },
          '& pre': {
            bgcolor: 'rgba(0,0,0,0.3)',
            p: 1,
            borderRadius: 1,
            overflow: 'auto',
            fontSize: '0.85em',
            maxWidth: '100%',
            '& code': {
              bgcolor: 'transparent',
              px: 0,
              py: 0,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
            }
          },
          '& ul, & ol': {
            margin: '0.5rem 0',
            paddingLeft: '1.5rem',
            wordBreak: 'break-word',
          },
          '& li': {
            margin: '0.25rem 0',
            wordBreak: 'break-word',
          },
          '& blockquote': {
            borderLeft: '3px solid #3A6FF7',
            pl: 1,
            ml: 0,
            fontStyle: 'italic',
            opacity: 0.9,
            wordBreak: 'break-word',
          },
          '& h1, & h2, & h3, & h4, & h5, & h6': {
            fontWeight: 600,
            margin: '1rem 0 0.5rem 0',
            wordBreak: 'break-word',
            '&:first-child': { marginTop: 0 }
          },
          '& *': {
            maxWidth: '100%',
            overflow: 'hidden',
          }
        }}
      >
        <ReactMarkdown components={markdownComponents}>
          {text}
        </ReactMarkdown>
      </Box>
    );
  }, [handleCitationClick]);

  // Function to extract reasoning from content if it wasn't separated by backend
  const extractReasoningFromContent = (content: string) => {
    // Look for common reasoning patterns
    const thinkBlockMatch = content.match(/<think>(.*?)<\/think>/is);
    if (thinkBlockMatch) {
      const reasoning = thinkBlockMatch[1].trim();
      const cleanContent = content.replace(/<think>.*?<\/think>/is, '').trim();
      return { cleanContent, reasoning };
    }

    // Look for "Reasoning:" or similar prefixes
    const reasoningMatch = content.match(/^(Reasoning:|Analysis:|Chain of thought:)(.*?)(?:\n\n|Final Answer:|Answer:|Conclusion:)/ism);
    if (reasoningMatch) {
      const reasoning = reasoningMatch[2].trim();
      const cleanContent = content.replace(reasoningMatch[0], '').trim();
      return { cleanContent, reasoning };
    }

    return { cleanContent: content, reasoning: null };
  };

  const renderMessage = useCallback((message: ChatMessage) => {
    const isAssistant = message.type === 'assistant';
    const isSystem = message.type === 'system';
    
    // Handle system messages differently
    if (isSystem) {
      return (
        <Box key={message.id} sx={{ mb: 2, px: 2, display: 'flex', justifyContent: 'center' }}>
          <Box
            sx={{
              px: 2,
              py: 1,
              borderRadius: 2,
              bgcolor: 'rgba(58, 111, 247, 0.1)',
              border: '1px solid rgba(58, 111, 247, 0.3)',
              color: 'rgba(255,255,255,0.8)',
              fontSize: '0.85rem',
              fontStyle: 'italic',
            }}
          >
            {message.content}
          </Box>
        </Box>
      );
    }
    
    // Extract reasoning from content if not already separated
    let messageContent = message.content;
    let reasoning = message.metadata?.reasoning;
    
    if (isAssistant && !reasoning) {
      const extracted = extractReasoningFromContent(message.content);
      messageContent = extracted.cleanContent;
      reasoning = extracted.reasoning;
    }
    
    const hasReasoning = Boolean(reasoning);
    const isReasoningExpanded = expandedReasoning[message.id] || false;

    return (
      <Box key={message.id} sx={{ mb: 3, px: 2, width: '100%', maxWidth: '100%' }}>
        {/* Main Message Bubble */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: isAssistant ? 'flex-start' : 'flex-end',
            width: '100%',
            maxWidth: '100%',
          }}
        >
          <Box
            sx={{
              maxWidth: {
                xs: isAssistant ? '90%' : '85%', // Mobile
                sm: isAssistant ? '85%' : '80%', // Tablet
                md: isAssistant ? '85%' : '80%', // Desktop
              },
              minWidth: {
                xs: '150px',
                sm: '200px',
              },
              width: 'fit-content',
              p: {
                xs: 2,
                sm: 2.5,
              },
              borderRadius: isAssistant ? '0 16px 16px 16px' : '16px 0 16px 16px',
              bgcolor: isAssistant ? '#2A2A2A' : '#3A6FF7',
              color: 'white',
              border: isAssistant ? '1px solid #404040' : 'none',
              boxShadow: isAssistant ? '0 1px 3px rgba(0,0,0,0.3)' : '0 1px 3px rgba(58, 111, 247, 0.3)',
              position: 'relative',
              wordWrap: 'break-word',
              overflowWrap: 'break-word',
              hyphens: 'auto',
              overflow: 'visible', // Allow content to be fully visible
            }}
          >
            {/* Action Buttons - Top Right */}
            {isAssistant && (hasReasoning || message.metadata?.citations !== undefined) && (
              <Box sx={{ position: 'absolute', top: 8, right: 8, display: 'flex', gap: 1 }}>
                {/* Citations Button */}
                {message.metadata?.citations !== undefined && (
                  <Button
                    size="small"
                    onClick={() => toggleSources(message.id)}
                    sx={{
                      minWidth: 'auto',
                      px: 1.5,
                      py: 0.5,
                      fontSize: '0.7rem',
                      bgcolor: expandedSources[message.id] ? 'rgba(34, 197, 94, 0.3)' : 'rgba(34, 197, 94, 0.15)',
                      color: expandedSources[message.id] ? 'white' : 'rgba(255,255,255,0.9)',
                      border: '1px solid rgba(34, 197, 94, 0.4)',
                      borderRadius: 1.5,
                      fontWeight: 500,
                      transition: 'all 0.2s',
                      '&:hover': {
                        bgcolor: 'rgba(34, 197, 94, 0.4)',
                        transform: 'translateY(-1px)',
                        boxShadow: '0 2px 4px rgba(34, 197, 94, 0.2)',
                      },
                    }}
                    startIcon={<AccessTime sx={{ fontSize: '0.8rem' }} />}
                  >
                    Citations ({message.metadata.citations?.length || 0})
                  </Button>
                )}
                
                {/* Reasoning Button */}
                {hasReasoning && (
                  <Button
                    size="small"
                    onClick={() => toggleReasoning(message.id)}
                    sx={{
                      minWidth: 'auto',
                      px: 1.5,
                      py: 0.5,
                      fontSize: '0.7rem',
                      bgcolor: isReasoningExpanded ? 'rgba(58, 111, 247, 0.3)' : 'rgba(58, 111, 247, 0.15)',
                      color: isReasoningExpanded ? 'white' : 'rgba(255,255,255,0.9)',
                      border: '1px solid rgba(58, 111, 247, 0.4)',
                      borderRadius: 1.5,
                      fontWeight: 500,
                      transition: 'all 0.2s',
                      '&:hover': {
                        bgcolor: 'rgba(58, 111, 247, 0.4)',
                        transform: 'translateY(-1px)',
                        boxShadow: '0 2px 4px rgba(58, 111, 247, 0.2)',
                      },
                    }}
                    startIcon={<Lightbulb sx={{ fontSize: '0.8rem' }} />}
                  >
                    {isReasoningExpanded ? 'Hide' : 'Show'} Reasoning
                  </Button>
                )}
              </Box>
            )}

            {/* Message Content with Markdown Support */}
            <Box sx={{ 
              pr: (hasReasoning || message.metadata?.citations !== undefined) ? 7 : 0,
              pt: (hasReasoning || message.metadata?.citations !== undefined) ? 2 : 0, // Add top padding if buttons are present
              width: '100%',
              overflow: 'visible',
            }}>
              <Box sx={{
                width: '100%',
                wordBreak: 'break-word',
                whiteSpace: 'pre-wrap',
                '& p': {
                  margin: '0.5rem 0',
                  lineHeight: 1.6,
                  wordBreak: 'break-word',
                  whiteSpace: 'pre-wrap',
                },
                '& p:last-child': { marginBottom: 0 },
                '& p:first-child': { marginTop: 0 },
                '& strong, & b': {
                  fontWeight: 600,
                  color: 'rgba(255,255,255,0.95)'
                },
                '& em, & i': {
                  fontStyle: 'italic',
                  opacity: 0.9
                },
                '& code': {
                  bgcolor: 'rgba(0,0,0,0.4)',
                  px: 0.75,
                  py: 0.25,
                  borderRadius: 0.75,
                  fontSize: '0.85em',
                  fontFamily: 'Monaco, Consolas, monospace',
                  wordBreak: 'break-all',
                  whiteSpace: 'pre-wrap',
                  border: '1px solid rgba(255,255,255,0.1)',
                },
                '& pre': {
                  bgcolor: 'rgba(0,0,0,0.5)',
                  p: 1.5,
                  borderRadius: 1,
                  overflow: 'auto',
                  fontSize: '0.85em',
                  maxWidth: '100%',
                  border: '1px solid rgba(255,255,255,0.1)',
                  margin: '0.75rem 0',
                  '& code': {
                    bgcolor: 'transparent',
                    px: 0,
                    py: 0,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all',
                  }
                },
                '& ul, & ol': {
                  margin: '0.75rem 0',
                  paddingLeft: '1.5rem',
                  wordBreak: 'break-word',
                },
                '& li': {
                  margin: '0.25rem 0',
                  wordBreak: 'break-word',
                  lineHeight: 1.5,
                },
                '& blockquote': {
                  borderLeft: '4px solid rgba(58, 111, 247, 0.4)',
                  paddingLeft: '1rem',
                  margin: '1rem 0',
                  backgroundColor: 'rgba(58, 111, 247, 0.05)',
                  padding: '0.5rem 1rem',
                  borderRadius: '0 4px 4px 0',
                  fontStyle: 'italic',
                  color: 'rgba(255,255,255,0.8)',
                  wordBreak: 'break-word',
                },
                '& h1, & h2, & h3, & h4, & h5, & h6': {
                  fontWeight: 600,
                  margin: '1rem 0 0.5rem 0',
                  wordBreak: 'break-word',
                  color: 'rgba(255,255,255,0.9)',
                  '&:first-child': { marginTop: 0 }
                },
                // Improved spacing for better readability
                '& h1 + p, & h2 + p, & h3 + p': { marginTop: '0.25rem' },
                '& p + h1, & p + h2, & p + h3': { marginTop: '1rem' },
                '& ul + p, & ol + p': { marginTop: '0.75rem' },
                '& p + ul, & p + ol': { marginTop: '0.5rem' },
                '& li + li': { marginTop: '0.25rem' },
                '& blockquote + p': { marginTop: '0.75rem' },
                '& p + blockquote': { marginTop: '0.5rem' },
                // Table spacing
                '& table + p': { marginTop: '0.75rem' },
                '& p + table': { marginTop: '0.5rem' },
                '& table + table': { marginTop: '1rem' },
                // Code block spacing
                '& pre + p': { marginTop: '0.75rem' },
                '& p + pre': { marginTop: '0.5rem' },
                // Ensure consistent spacing between sections
                '& > *:not(:last-child)': { marginBottom: '0.5rem' },
                '& *': {
                  maxWidth: '100%',
                  overflow: 'hidden',
                }
              }}>
                <CitationTextRenderer text={messageContent} citations={getMessageCitations(message)} />
              </Box>
            </Box>


            {/* Collapsible Reasoning Panel */}
            {isAssistant && hasReasoning && (
              <Collapse in={isReasoningExpanded} timeout="auto" unmountOnExit>
                <Box sx={{
                  mt: 2,
                  p: 2,
                  bgcolor: 'rgba(0,0,0,0.2)',
                  borderRadius: 1,
                  border: '1px solid rgba(255,255,255,0.1)'
                }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'rgba(255,255,255,0.7)',
                      fontSize: '0.75rem',
                      fontStyle: 'italic',
                      mb: 1,
                      display: 'block'
                    }}
                  >
                    Model reasoning (hidden by default)
                  </Typography>
                  <Box sx={{
                    fontSize: '0.85rem',
                    lineHeight: 1.5,
                    opacity: 0.9,
                    width: '100%',
                    wordBreak: 'break-word',
                    whiteSpace: 'pre-wrap',
                    maxWidth: '100%',
                    overflow: 'hidden',
                    '& p': { 
                      margin: '0.5rem 0',
                      wordBreak: 'break-word',
                      whiteSpace: 'pre-wrap',
                    },
                    '& p:first-child': { marginTop: 0 },
                    '& p:last-child': { marginBottom: 0 },
                    '& *': {
                      maxWidth: '100%',
                      overflow: 'hidden',
                    }
                  }}>
                    <CitationTextRenderer text={reasoning} citations={getMessageCitations(message)} />
                  </Box>
                </Box>
              </Collapse>
            )}

            {/* Collapsible Citations Panel */}
            {isAssistant && message.metadata?.citations !== undefined && (
              <Collapse in={expandedSources[message.id]} timeout="auto" unmountOnExit>
                <Box sx={{
                  mt: 2,
                  p: 2,
                  bgcolor: 'rgba(0,0,0,0.2)',
                  borderRadius: 1,
                  border: '1px solid rgba(255,255,255,0.1)'
                }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'rgba(255,255,255,0.7)',
                      fontSize: '0.75rem',
                      fontStyle: 'italic',
                      mb: 1,
                      display: 'block'
                    }}
                  >
                    Citations used for this response
                  </Typography>
                  {message.metadata.citations && message.metadata.citations.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                      {message.metadata.citations.map((citation: CitationInfo, index: number) => (
                        <Tooltip
                          key={index}
                          title={
                            <Box sx={{ p: 1, maxWidth: 300 }}>
                              <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                                {citation.display_name} • Chunk {citation.chunk_number}
                              </Typography>
                              <Typography variant="body2" sx={{ mb: 1 }}>
                                Relevance: {(citation.relevance_score * 100).toFixed(1)}%
                              </Typography>
                              <Typography variant="body2">
                                {citation.content.substring(0, 120)}...
                              </Typography>
                            </Box>
                          }
                          placement="top"
                          arrow
                        >
                          <Chip
                            label={`(cit#${index + 1})`}
                          size="small"
                            onClick={() => handleCitationClick(citation, index)}
                          sx={{
                              bgcolor: 'rgba(34, 197, 94, 0.15)',
                            color: '#22C55E',
                            border: '1px solid rgba(34, 197, 94, 0.4)',
                            fontSize: '0.75rem',
                              fontWeight: 600,
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            '&:hover': {
                                bgcolor: 'rgba(34, 197, 94, 0.25)',
                              transform: 'translateY(-1px)',
                                boxShadow: '0 2px 4px rgba(34, 197, 94, 0.2)',
                            },
                          }}
                        />
                        </Tooltip>
                      ))}
                    </Box>
                  ) : (
                    <Box sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                      p: 1.5,
                      bgcolor: 'rgba(255, 193, 7, 0.1)',
                      border: '1px solid rgba(255, 193, 7, 0.3)',
                      borderRadius: 1,
                    }}>
                      <Typography sx={{ fontSize: '1rem', color: '#FFC107' }}>⚠️</Typography>
                      <Typography variant="body2" sx={{
                        color: 'rgba(255, 193, 7, 0.9)',
                        fontSize: '0.8rem',
                        fontStyle: 'italic',
                      }}>
                        No related sources found. Answer is based on model's knowledge.
                      </Typography>
                    </Box>
                  )}
                </Box>
              </Collapse>
            )}
          </Box>
        </Box>

        {/* Metadata Display */}
        <Box sx={{
          display: 'flex',
          justifyContent: isAssistant ? 'flex-start' : 'flex-end',
          mt: 0.5,
          px: 2
        }}>
          <Box sx={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 1.5,
            p: 0.75,
            bgcolor: 'rgba(255,255,255,0.05)',
            borderRadius: 1,
            border: '1px solid rgba(255,255,255,0.1)',
          }}>
            {isAssistant && message.metadata?.confidence_score && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography
                  variant="caption"
                  sx={{
                    color: 'rgba(255,255,255,0.7)',
                    fontSize: '0.7rem',
                    fontWeight: 500,
                  }}
                >
                  Confidence: {(message.metadata.confidence_score * 100).toFixed(0)}%
                </Typography>
                <Box sx={{ 
                  width: '2px', 
                  height: '12px', 
                  bgcolor: 'rgba(255,255,255,0.2)',
                  borderRadius: 1,
                }} />
              </Box>
            )}
            <Typography
              variant="caption"
              sx={{
                color: 'rgba(255,255,255,0.7)',
                fontSize: '0.7rem',
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                fontWeight: 500,
              }}
            >
              <AccessTime sx={{ fontSize: '0.8rem' }} />
              {formatRelativeTime(message.timestamp)}
            </Typography>
          </Box>
        </Box>

        {/* Suggested Questions as Chips */}
        {isAssistant && message.metadata?.suggested_questions && message.metadata.suggested_questions.length > 0 && (
          <Box sx={{ mt: 2, px: 2 }}>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {message.metadata.suggested_questions.slice(0, 3).map((suggestion: string, index: number) => (
                <Chip
                  key={index}
                  label={suggestion}
                  onClick={() => setInputValue(suggestion)}
                  sx={{
                    bgcolor: 'rgba(58, 111, 247, 0.1)',
                    color: '#3A6FF7',
                    border: '1px solid rgba(58, 111, 247, 0.3)',
                    fontSize: '0.8rem',
                    fontWeight: 400,
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    '&:hover': {
                      bgcolor: 'rgba(58, 111, 247, 0.15)',
                      transform: 'translateY(-1px)',
                      boxShadow: '0 2px 4px rgba(58, 111, 247, 0.2)',
                    },
                  }}
                />
              ))}
            </Box>
          </Box>
        )}
      </Box>
    );
  }, [toggleReasoning, toggleSources, formatRelativeTime, expandedReasoning, expandedSources]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Messages Area - Scrollable */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          overflowX: 'hidden', // Prevent horizontal scrolling
          p: 3,
          bgcolor: '#1E1E1E',
          pb: '140px', // Increased space for fixed input
          minHeight: 0, // Allow flex child to shrink
        }}
      >
        {messages.length === 0 && sourceCount === 0 ? (
          // No sources: Show "Add source" empty state
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              textAlign: 'center',
              py: 8,
            }}
          >
            <Typography variant="h4" color="text.secondary" sx={{ mb: 2 }}>
              Add a source to get started
            </Typography>
            <Typography variant="body1" color="text.secondary" mb={4}>
              Upload PDFs, documents, or other files to start asking questions about your content.
            </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                No sources added yet. Click "Add source" in the left panel to begin.
              </Typography>
          </Box>
        ) : messages.length === 0 && sourceCount > 0 ? (
          // Sources exist but no messages: Show animated book with inviting message
          <Box
            sx={{
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              textAlign: 'center',
              py: 6,
              overflow: 'hidden',
            }}
          >
            {/* Animated Scene Background */}
            <AnimatedSceneBackground />

            {/* Text Overlay */}
            <Box sx={{ position: 'relative', zIndex: 10 }}>
              <Typography
                variant="h4"
                sx={{
                  mb: 3,
                  color: 'text.primary',
                  fontWeight: 600,
                  fontSize: '2rem',
                  textAlign: 'center',
                }}
              >
                ✨ Ready when you are!
              </Typography>
              <Typography
                variant="body1"
                sx={{
                  mb: 4,
                  color: 'text.secondary',
                  maxWidth: 400,
                  lineHeight: 1.6,
                  fontSize: '1.1rem',
                  textAlign: 'center',
                }}
              >
                Pick a document from the sidebar and ask me anything about it.
              </Typography>
            </Box>
          </Box>
        ) : (
          messages.map(renderMessage)
        )}

        {isLoading && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3, px: 2 }}>
            <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
              <SmartToy fontSize="small" />
            </Avatar>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                BookMate is thinking
              </Typography>
              <CircularProgress size={16} />
            </Box>
          </Box>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Minimal Query Input - Bottom Aligned */}
      <Box
        sx={{
          position: 'fixed',
          bottom: 0,
          left: {
            xs: 0,
            md: 320,
          },
          right: 0,
          p: 3,
          zIndex: 1200,
        }}
      >
        <Box sx={{ maxWidth: '1200px', mx: 'auto' }}>
          {/* Slim Input Row */}
          <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 2 }}>
              <TextField
                fullWidth
                multiline
              maxRows={6}
                value={inputValue}
              onChange={handleInputChange}
                onKeyPress={handleKeyPress}
              onFocus={() => setIsInputExpanded(true)}
              onBlur={() => setIsInputExpanded(false)}
              placeholder="Ask about your selected sources..."
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                  bgcolor: 'rgba(255, 255, 255, 0.05)',
                  color: 'text.primary',
                  borderRadius: 2,
                  minHeight: isInputExpanded ? '56px' : '48px',
                  height: inputValue ? 'auto' : (isInputExpanded ? '56px' : '48px'),
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  transition: 'all 0.2s ease',
                    '& fieldset': {
                      border: 'none',
                    },
                    '&:hover': {
                    border: '1px solid rgba(58, 111, 247, 0.4)',
                    bgcolor: 'rgba(255, 255, 255, 0.08)',
                    },
                    '&.Mui-focused': {
                    border: '1px solid #3A6FF7',
                    bgcolor: 'rgba(255, 255, 255, 0.08)',
                    minHeight: '56px',
                    boxShadow: '0 0 0 3px rgba(58, 111, 247, 0.1)',
                    },
                  },
                  '& .MuiOutlinedInput-input': {
                  padding: isInputExpanded ? '16px' : '14px 16px',
                  fontSize: '0.95rem',
                  lineHeight: 1.4,
                  fontWeight: 400,
                  transition: 'all 0.2s ease',
                    '&::placeholder': {
                    color: 'text.secondary',
                      opacity: 1,
                    fontSize: '0.95rem',
                    fontWeight: 400,
                    },
                  },
                }}
                disabled={isLoading}
              />

            {/* Minimal Send Button */}
              <IconButton
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                sx={{
                width: 40,
                height: 40,
                bgcolor: inputValue.trim() ? 'primary.main' : 'rgba(255,255,255,0.1)',
                  color: inputValue.trim() ? 'white' : 'rgba(255,255,255,0.4)',
                borderRadius: 2,
                  '&:hover': {
                  bgcolor: inputValue.trim() ? 'primary.dark' : 'rgba(255,255,255,0.15)',
                  },
                  '&:disabled': {
                    bgcolor: 'rgba(255,255,255,0.08)',
                    color: 'rgba(255,255,255,0.3)',
                  },
                }}
              >
              <Send sx={{ fontSize: '1.1rem' }} />
              </IconButton>
            </Box>

            {/* AI Disclaimer */}
            <Box sx={{
              display: 'flex',
              justifyContent: 'center',
              mt: 1,
              px: 2,
            }}>
              <Typography
                variant="caption"
                sx={{
                  color: 'text.secondary',
                  fontSize: '0.75rem',
                  textAlign: 'center',
                  fontStyle: 'italic',
                  opacity: 0.7,
                  maxWidth: '400px',
                }}
              >
                BookMate can make mistakes. Check important info.
              </Typography>
          </Box>
        </Box>
      </Box>

      {/* Citation Modal */}
      <Dialog
        open={citationModal.open}
        onClose={handleCloseCitationModal}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: 'background.paper',
            color: 'text.primary',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 2,
          }
        }}
      >
        <DialogTitle sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          bgcolor: 'background.paper',
          color: 'text.primary',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          pb: 1
        }}>
          <Typography variant="h6" sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            color: 'text.primary'
          }}>
            📄 Citation #{citationModal.citationIndex + 1}
          </Typography>
          <IconButton
            onClick={handleCloseCitationModal}
            sx={{ color: 'text.secondary' }}
          >
            <Close />
          </IconButton>
        </DialogTitle>

        <DialogContent sx={{ p: 3 }}>
          {citationModal.citation && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* Enhanced Citation Header */}
              <Box sx={{
                p: 2,
                bgcolor: 'rgba(34, 197, 94, 0.1)',
                border: '1px solid rgba(34, 197, 94, 0.3)',
                borderRadius: 2,
              }}>
                <Typography variant="body2" sx={{
                  color: 'text.secondary',
                  mb: 1,
                  fontSize: '0.9rem'
                }}>
                  📄 {citationModal.citation.display_name || 'Unknown Document'}
                </Typography>
                <Typography variant="body2" sx={{
                  color: 'text.secondary',
                  mb: 1,
                  fontSize: '0.9rem'
                }}>
                  Chunk {citationModal.citation.chunk_number} of document
                </Typography>
                <Typography variant="body2" sx={{
                  color: 'text.secondary',
                  fontSize: '0.9rem'
                }}>
                  Relevance: {(citationModal.citation.relevance_score * 100).toFixed(1)}%
                </Typography>
      </Box>

              {/* FULL CHUNK CONTENT - Not just snippet */}
              <Box sx={{
                p: 3,
                bgcolor: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: 2,
                maxHeight: '500px',
                overflow: 'auto'
              }}>
                <Typography variant="body1" sx={{
                  color: 'text.primary',
                  lineHeight: 1.7,
                  fontSize: '1rem',
                  whiteSpace: 'pre-wrap',  // Preserve formatting
                  wordBreak: 'break-word'
                }}>
                  {citationModal.citation.content}
                </Typography>
              </Box>

              {/* Enhanced Metadata */}
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                <Typography variant="caption" sx={{
                  color: 'text.secondary',
                  bgcolor: 'rgba(255, 255, 255, 0.05)',
                  px: 1.5,
                  py: 0.5,
                  borderRadius: 1,
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  fontSize: '0.8rem'
                }}>
                  📊 {citationModal.citation.token_count} tokens
                </Typography>

                <Typography variant="caption" sx={{
                  color: 'text.secondary',
                  bgcolor: 'rgba(255, 255, 255, 0.05)',
                  px: 1.5,
                  py: 0.5,
                  borderRadius: 1,
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  fontSize: '0.8rem'
                }}>
                  📍 Position {citationModal.citation.start_position}-{citationModal.citation.end_position}
                </Typography>

                {citationModal.citation.page_number && (
                  <Typography variant="caption" sx={{
                    color: 'text.secondary',
                    bgcolor: 'rgba(255, 255, 255, 0.05)',
                    px: 1.5,
                    py: 0.5,
                    borderRadius: 1,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    fontSize: '0.8rem'
                  }}>
                    📄 Page {citationModal.citation.page_number}
                  </Typography>
                )}

                {citationModal.citation.section && (
                  <Typography variant="caption" sx={{
                    color: 'text.secondary',
                    bgcolor: 'rgba(255, 255, 255, 0.05)',
                    px: 1.5,
                    py: 0.5,
                    borderRadius: 1,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    fontSize: '0.8rem'
                  }}>
                    📑 {citationModal.citation.section}
                  </Typography>
                )}
              </Box>
            </Box>
          )}
        </DialogContent>

        <DialogActions sx={{
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          bgcolor: 'background.paper',
          p: 2
        }}>
          <Button
            onClick={handleCloseCitationModal}
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
    </Box>
  );
};
