"""
Thread-safe session management for multimodal depression screening
"""
import uuid
import threading
from datetime import datetime
from typing import Dict, Optional

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._lock = threading.RLock()
    
    def create_session(self) -> str:
        """Create new session and return session_id"""
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "consent_given": False,
                "questionnaire": [],
                "video_results": [],
                "text_analysis": [],
                "final_result": None,
                "crisis_flag": False,
                "phq_total_score": 0,
                "video_files": []
            }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        with self._lock:
            return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: dict) -> bool:
        """Update session data"""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id].update(updates)
            return True
    
    def add_questionnaire_response(self, session_id: str, response: dict) -> bool:
        """Add single questionnaire response"""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id]["questionnaire"].append(response)
            return True
    
    def add_video_result(self, session_id: str, result: dict) -> bool:
        """Add video analysis result"""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id]["video_results"].append(result)
            return True
    
    def add_text_analysis(self, session_id: str, analysis: dict) -> bool:
        """Add text semantic analysis"""
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id]["text_analysis"].append(analysis)
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session after report generation"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def get_all_sessions_count(self) -> int:
        """Get total active sessions (for monitoring)"""
        with self._lock:
            return len(self._sessions)

# Global session manager instance
session_manager = SessionManager()
