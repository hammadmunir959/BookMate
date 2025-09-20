/**
 * WebSocket service for real-time ingestion progress updates
 */

// Removed import for IngestionProgressResponse as it's not used in simplified API

export interface WebSocketMessage {
  type: string;
  data?: unknown;
  document_id?: string;
  error?: string;
}

export type ProgressCallback = (progress: unknown) => void;
export type EventCallback = (event: WebSocketMessage) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private connectionId: string = '';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private progressCallbacks: Map<string, ProgressCallback> = new Map();
  private eventCallbacks: Set<EventCallback> = new Set();

  constructor() {
    this.connectionId = this.generateConnectionId();
  }

  private generateConnectionId(): string {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  async connect(): Promise<void> {
    // For simplified API, WebSocket is disabled
    console.log('WebSocket connection disabled in simplified API');
    return Promise.resolve();
  }

  async connect_original(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      try {
        const wsUrl = `${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}/ingestion/ws/${this.connectionId}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.ws = null;

          // Attempt to reconnect if not a normal closure
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnect failed:', error);
      });
    }, delay);
  }

  private handleMessage(message: WebSocketMessage): void {
    console.log('WebSocket message received:', message);

    // Handle different message types
    switch (message.type) {
      case 'progress_update':
        if (message.data && message.document_id) {
          const callback = this.progressCallbacks.get(message.document_id);
          if (callback) {
            callback(message.data);
          }
        }
        break;

      case 'ingestion_event':
        if ((message as any).data && (message as any).document_id) {
          const callback = this.progressCallbacks.get((message as any).document_id);
          if (callback && (message as any).data.type === 'progress') {
            callback((message as any).data.data);
          }
        }
        break;

      case 'pong':
        // Handle ping/pong for connection health
        break;

      case 'error':
        console.error('WebSocket error message:', message.error);
        break;

      default:
        console.log('Unknown message type:', message.type);
    }

    // Notify all event callbacks
    this.eventCallbacks.forEach(callback => {
      try {
        callback(message);
      } catch (error) {
        console.error('Error in event callback:', error);
      }
    });
  }

  subscribeToDocument(documentId: string, callback: ProgressCallback): void {
    this.progressCallbacks.set(documentId, callback);
    
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.send({
        type: 'subscribe',
        document_id: documentId
      });
    }
  }

  unsubscribeFromDocument(documentId: string): void {
    this.progressCallbacks.delete(documentId);
    
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.send({
        type: 'unsubscribe',
        document_id: documentId
      });
    }
  }

  addEventListener(callback: EventCallback): void {
    this.eventCallbacks.add(callback);
  }

  removeEventListener(callback: EventCallback): void {
    this.eventCallbacks.delete(callback);
  }

  private send(message: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message:', message);
    }
  }

  ping(): void {
    this.send({ type: 'ping' });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.progressCallbacks.clear();
    this.eventCallbacks.clear();
  }

  getConnectionState(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'closed';
      default:
        return 'unknown';
    }
  }

  isConnected(): boolean {
    // For simplified API, WebSocket is disabled
    return false;
  }
}

// Export singleton instance
export const wsService = new WebSocketService();

// Auto-connect disabled for simplified API
// WebSocket functionality is not implemented in the simplified router
// Uncomment the following lines if you add WebSocket endpoints to the backend:
/*
wsService.connect().catch(error => {
  console.error('Failed to connect WebSocket on startup:', error);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  wsService.disconnect();
});
*/

// Log that WebSocket is disabled
console.log('WebSocket service disabled - not implemented in simplified API');
