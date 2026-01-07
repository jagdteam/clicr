#!/usr/bin/env python3
"""
History Management for CodeReader
Manages chat sessions, conversation history, and query logs.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


# Configuration
HISTORY_DIR = "./chat_history"
SESSIONS_FILE = "sessions.json"
QUERIES_FILE = "queries.json"


class HistoryManager:
    """Manages chat sessions and query history."""

    def __init__(self, history_dir: str = HISTORY_DIR):
        """
        Initialize the history manager.

        Args:
            history_dir: Directory to store history files
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        
        self.sessions_file = self.history_dir / SESSIONS_FILE
        self.queries_file = self.history_dir / QUERIES_FILE
        
        # Initialize files if they don't exist
        if not self.sessions_file.exists():
            self._save_json(self.sessions_file, {"sessions": []})
        if not self.queries_file.exists():
            self._save_json(self.queries_file, {"queries": []})

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """
        Load JSON data from file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary containing JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load {file_path}: {e}")
            return {}

    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Save JSON data to file.

        Args:
            file_path: Path to JSON file
            data: Dictionary to save
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Warning: Could not save {file_path}: {e}")

    def create_session(self, session_name: Optional[str] = None) -> str:
        """
        Create a new chat session.

        Args:
            session_name: Optional name for the session

        Returns:
            Session ID
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not session_name:
            session_name = f"Session {session_id}"
        
        session = {
            "id": session_id,
            "name": session_name,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
        
        # Save session metadata
        data = self._load_json(self.sessions_file)
        data["sessions"].append({
            "id": session_id,
            "name": session_name,
            "created_at": session["created_at"]
        })
        self._save_json(self.sessions_file, data)
        
        # Create session file
        session_file = self.history_dir / f"{session_id}.json"
        self._save_json(session_file, session)
        
        return session_id

    def add_message(self, session_id: str, role: str, content: str, 
                   sources: Optional[List[str]] = None) -> None:
        """
        Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role (user/assistant)
            content: Message content
            sources: Optional list of source files cited
        """
        session_file = self.history_dir / f"{session_id}.json"
        
        if not session_file.exists():
            print(f"⚠️  Warning: Session {session_id} not found")
            return
        
        session = self._load_json(session_file)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if sources:
            message["sources"] = sources
        
        session["messages"].append(message)
        self._save_json(session_file, session)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session data or None if not found
        """
        session_file = self.history_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        return self._load_json(session_file)

    def list_sessions(self) -> List[Dict[str, str]]:
        """
        List all sessions.

        Returns:
            List of session metadata dictionaries
        """
        data = self._load_json(self.sessions_file)
        return data.get("sessions", [])

    def export_session_markdown(self, session_id: str, output_file: Optional[str] = None) -> str:
        """
        Export a session to markdown format.

        Args:
            session_id: Session ID
            output_file: Optional output file path

        Returns:
            Path to exported file
        """
        session = self.get_session(session_id)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Generate markdown content
        lines = [
            f"# {session['name']}",
            f"\n**Created:** {session['created_at']}",
            f"\n**Session ID:** {session['id']}",
            "\n---\n"
        ]
        
        for msg in session["messages"]:
            timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            
            if msg["role"] == "user":
                lines.append(f"\n## 🙋 User ({timestamp})\n")
                lines.append(f"{msg['content']}\n")
            else:
                lines.append(f"\n## 🤖 Assistant ({timestamp})\n")
                lines.append(f"{msg['content']}\n")
                
                if "sources" in msg and msg["sources"]:
                    lines.append("\n**Sources:**\n")
                    for source in msg["sources"]:
                        lines.append(f"- `{source}`\n")
        
        markdown_content = "\n".join(lines)
        
        # Save to file
        if not output_file:
            output_file = self.history_dir / f"{session_id}_export.md"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(output_file)

    def log_query(self, query: str, response_preview: str, sources: List[str]) -> None:
        """
        Log a query to the query history.

        Args:
            query: User query
            response_preview: Preview of the response (first 200 chars)
            sources: List of source files used
        """
        data = self._load_json(self.queries_file)
        
        query_entry = {
            "query": query,
            "response_preview": response_preview[:200],
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        data["queries"].append(query_entry)
        
        # Keep only last 100 queries
        if len(data["queries"]) > 100:
            data["queries"] = data["queries"][-100:]
        
        self._save_json(self.queries_file, data)

    def get_query_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get query history.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of query entries
        """
        data = self._load_json(self.queries_file)
        queries = data.get("queries", [])
        return queries[-limit:][::-1]  # Return most recent first

    def search_queries(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search query history by keyword.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching query entries
        """
        data = self._load_json(self.queries_file)
        queries = data.get("queries", [])
        
        keyword_lower = keyword.lower()
        matching = [
            q for q in queries
            if keyword_lower in q["query"].lower() or 
               keyword_lower in q["response_preview"].lower()
        ]
        
        return matching[::-1]  # Return most recent first

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Remove session file
            session_file = self.history_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Remove from sessions list
            data = self._load_json(self.sessions_file)
            data["sessions"] = [s for s in data["sessions"] if s["id"] != session_id]
            self._save_json(self.sessions_file, data)
            
            return True
        except Exception as e:
            print(f"⚠️  Error deleting session: {e}")
            return False

    def get_conversation_history(self, session_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Get conversation history for context in chat.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        session = self.get_session(session_id)
        
        if not session or not session.get("messages"):
            return []
        
        messages = session["messages"][-limit * 2:]  # Get last N pairs of user/assistant messages
        
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

