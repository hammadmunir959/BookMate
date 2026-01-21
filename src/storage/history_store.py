"""
History Store for RAG Agent
Handles persistence of chat sessions and messages using SQLite
"""

import sqlite3
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
from pathlib import Path

from src.core.config import config

logger = logging.getLogger(__name__)


class SQLiteHistoryStore:
    """SQLite implementation of history storage"""

    def __init__(self, db_path: str = None):
        """Initialize history store"""
        self.db_path = db_path or config.database.sqlite_db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        title TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                
                # Messages table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                ''')
                
                conn.commit()
                # logger.info("History database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing history database: {str(e)}")

    def create_session(self, user_id: str = "default_user", title: str = None, metadata: Dict = None) -> str:
        """Create a new chat session"""
        try:
            session_id = str(uuid.uuid4())
            ts = datetime.utcnow().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, user_id, title, ts, ts, json.dumps(metadata or {}))
                )
                conn.commit()
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return str(uuid.uuid4()) # Fallback

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> str:
        """Add a message to a session"""
        try:
            message_id = str(uuid.uuid4())
            ts = datetime.utcnow().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Ensure session exists (auto-create if not found?)
                # Preferably expected to exist, but let's be safe or just fail if strict.
                # Here we assume session exists.
                
                cursor.execute(
                    "INSERT INTO messages (message_id, session_id, role, content, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (message_id, session_id, role, content, ts, json.dumps(metadata or {}))
                )
                
                # Update session timestamp
                cursor.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (ts, session_id)
                )
                
                conn.commit()
            
            return message_id
            
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {str(e)}")
            return ""

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get message history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit)
                )
                
                rows = cursor.fetchall()
                
                # Convert to list and reverse to chronological order (oldest first)
                messages = []
                for row in rows:
                    msg = dict(row)
                    if msg.get('metadata'):
                        try:
                            msg['metadata'] = json.loads(msg['metadata'])
                        except:
                            msg['metadata'] = {}
                    messages.append(msg)
                
                return list(reversed(messages))
                
        except Exception as e:
            logger.error(f"Error getting history for session {session_id}: {str(e)}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            return False

# Global instance
history_store = SQLiteHistoryStore()
