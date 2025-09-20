class BookMateApp {
  constructor() {
    this.initElements();
    this.bindEvents();
    this.currentDocument = null;
    this.loadDocuments();
  }

  initElements() {
    this.newChatBtn = document.querySelector('.new-chat-btn');
    this.documentUpload = document.getElementById('document-upload');
    this.documentUpload.accept = ".pdf,.doc,.docx,.txt"; 
    this.uploadProgress = document.querySelector('.upload-progress');
    this.uploadProgressBar = this.uploadProgress.querySelector('.progress-bar');
    this.uploadProgressText = this.uploadProgress.querySelector('.progress-text');
    this.documentList = document.querySelector('.document-list');
    this.chatMessages = document.querySelector('.chat-messages');
    this.inputArea = document.querySelector('.input-area input');
    this.sendBtn = document.querySelector('.send-btn');
    this.currentDocumentSpan = document.querySelector('.current-document');
  }

  bindEvents() {
    this.documentUpload.addEventListener('change', (e) => this.handleDocumentUpload(e));
    this.newChatBtn.addEventListener('click', () => this.startNewSession());
    this.sendBtn.addEventListener('click', () => this.sendMessage());
    this.inputArea.addEventListener('keypress', (e) => e.key === 'Enter' && this.sendMessage());
  }

  async loadDocuments() {
    try {
      const response = await fetch('/documents');
      const { documents } = await response.json();
      documents.forEach((doc) => this.addDocumentToList(doc));
    } catch (error) {
      console.error('Error loading documents:', error);
      this.addSystemMessage('Error loading documents. Please refresh the page.');
    }
  }

  setUploadState(isLoading) {
    this.uploadProgress.style.display = isLoading ? 'block' : 'none';
    this.sendBtn.disabled = isLoading;
  }

  async handleDocumentUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
  
    this.setUploadState(true);
  
    // Initialize progressSteps array
    const progressSteps = [
      { threshold: 20, message: "Uploading file..." },
      { threshold: 40, message: "Extracting text..." },
      { threshold: 60, message: "Chunking content..." },
      { threshold: 80, message: "Generating embeddings..." },
      { threshold: 100, message: "Ready for queries!" },
    ];
  
    // Pass progressSteps array instead of a string
    this.updateProgress(0, progressSteps); // Fix: Pass the array here
  
    const progressInterval = setInterval(() => this.updateProgress(5, progressSteps), 200);
  
    try {
      const formData = new FormData();
      formData.append('file', file);
  
      const response = await fetch('/upload', { method: 'POST', body: formData });
      const result = await response.json();
  
      clearInterval(progressInterval);
      this.updateProgress(100, progressSteps);
      await new Promise((resolve) => setTimeout(resolve, 500));
  
      this.setUploadState(false);
  
      if (result.status === 'success') {
        this.addDocumentToList(result.file_path);
        this.setCurrentDocument(result.file_path);
        this.addSystemMessage(`Document "${file.name}" uploaded successfully!`);
      } else {
        throw new Error(result.message || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      clearInterval(progressInterval);
      this.setUploadState(false);
      this.addSystemMessage(`Error: ${error.message}`);
    }
  }

  updateProgress(progress, steps) {
    progress = Math.min(progress, 100);
    this.uploadProgressBar.style.width = `${progress}%`;
    this.uploadProgressText.textContent = `${progress}%`;
  
    // Check if steps is an array
    if (!Array.isArray(steps)) {
      console.error("Steps is not an array:", steps);
      return;
    }
  
    // Find the current step message
    const stepMsg = steps.find((step) => progress >= step.threshold)?.message || '';
  
    // Update the step message if the element exists
    const messageElement = this.uploadProgress.querySelector('.upload-step-message');
    if (messageElement) {
      messageElement.textContent = stepMsg;
    } else {
      console.warn("Warning: .upload-step-message element not found");
    }
  }



  addDocumentToList(documentPath) {
    const placeholder = this.documentList.querySelector('.no-documents-placeholder');
    if (placeholder) placeholder.remove();

    const documentItem = document.createElement('div');
    documentItem.classList.add('document-item');
    documentItem.innerHTML = `
      <span class="document-icon">📄</span>
      <span class="document-name">${documentPath.split('/').pop()}</span>
      <button class="remove-document">✖</button>
    `;

    documentItem.addEventListener('click', (e) => {
      if (!e.target.classList.contains('remove-document')) this.setCurrentDocument(documentPath);
    });

    documentItem.querySelector('.remove-document').addEventListener('click', (e) => {
      e.stopPropagation();
      this.deleteDocument(documentPath).then((success) => {
        if (success) {
          documentItem.remove();
          this.addSystemMessage(`Document "${documentPath.split('/').pop()}" removed.`);
          if (this.currentDocument === documentPath) {
            this.currentDocument = null;
            this.currentDocumentSpan.textContent = 'Please select a document';
          }
        }
      });
    });

    this.documentList.appendChild(documentItem);
  }

  async deleteDocument(documentPath) {
    try {
      const response = await fetch('/delete-document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_path: documentPath }),
      });
      const result = await response.json();
      return result.status === 'success';
    } catch (error) {
      console.error('Delete error:', error);
      return false;
    }
  }

  async setCurrentDocument(documentPath) {
    if (!documentPath) {
        console.error("No document path provided");
        this.addSystemMessage("Error: Document path is missing.");
        return;
    }

    try {
        // Ensure documentPath starts with 'uploads/'
        if (!documentPath.startsWith("uploads")) {
            documentPath = "uploads/" + documentPath;
        }

        // Send data as JSON
        const response = await fetch('/set-document', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },  // Set the content type to JSON
            body: JSON.stringify({ document_path: documentPath })  // Send JSON body
        });
        
        const result = await response.json();

        if (result.status === 'success') {
            this.currentDocument = documentPath;
            this.currentDocumentSpan.textContent = documentPath.split('/').pop();
            this.highlightDocument(documentPath);
        } else {
            throw new Error(result.message || 'Failed to set document');
        }
    } catch (error) {
        console.error('Set document error:', error);
        this.addSystemMessage(`Error: ${error.message}`);
    }
}


  highlightDocument(documentPath) {
    const documentItems = this.documentList.querySelectorAll('.document-item');
    documentItems.forEach((item) => {
      item.classList.toggle(
        'selected',
        item.querySelector('.document-name').textContent === documentPath.split('/').pop()
      );
    });
  }
// /////
  async sendMessage() {
    const message = this.inputArea.value.trim();
    if (!message) return;

    this.addUserMessage(message);
    this.inputArea.value = '';

    const loadingMessage = this.addLoadingMessage();

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message }),
      });
      const result = await response.json();
      this.replaceLoadingMessage(loadingMessage, result.answer || 'No answer found.');
    } catch (error) {
      console.error('Query error:', error);
      this.replaceLoadingMessage(loadingMessage, 'Error processing your query.');
    }
  }

  addLoadingMessage() {
    const loadingMessage = document.createElement('div');
    loadingMessage.classList.add('chat-message', 'assistant-message', 'loading-message');
    loadingMessage.innerHTML = `
      <div class="message-content">Typing...</div>
      <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
    `;
    this.chatMessages.appendChild(loadingMessage);
    this.scrollToBottom();
    return loadingMessage;
  }

  replaceLoadingMessage(loadingMessage, reply) {
    const replyMessage = document.createElement('div');
    replyMessage.classList.add('chat-message', 'assistant-message');
    replyMessage.innerHTML = `
      <div class="message-content">${reply}</div>
      <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
    `;
    loadingMessage.replaceWith(replyMessage);
    this.scrollToBottom();
  }

  addUserMessage(message) {
    this.addMessage(message, 'user-message');
  }

  addSystemMessage(message) {
    this.addMessage(message, 'system-message');
  }

  addMessage(message, className) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', className);
    messageElement.innerHTML = `
      <div class="message-content">${message}</div>
      <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
    `;
    this.chatMessages.appendChild(messageElement);
    this.scrollToBottom();
  }

  startNewSession() {
    this.chatMessages.innerHTML = `
      <div class="chat-message system-message">
        Welcome to BookMate. Please add and select a document to start your session.
      </div>
    `;
    this.currentDocument = null;
    this.currentDocumentSpan.textContent = 'Please select a document';
  }

  scrollToBottom() {
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.bookMateApp = new BookMateApp();
  console.log('BookMate App Initialized');
});