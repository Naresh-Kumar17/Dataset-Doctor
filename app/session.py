"""
In-memory session manager for Dataset Doctor.
Stores active DataFrames, profiles, target settings, and user decisions by session UUID.
Enforces automatic 1-hour expiration.
"""
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd


@dataclass
class SessionData:
    session_id: str
    df: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    profile: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # "classification" | "regression"
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        self.last_accessed = time.time()


# In-memory session store: session_id -> SessionData
SESSIONS: Dict[str, SessionData] = {}
SESSION_EXPIRY_SECONDS = 3600  # 1 hour


def create_session() -> SessionData:
    """Creates a new session with a random UUID."""
    cleanup_expired_sessions()
    sess_id = str(uuid.uuid4())
    session = SessionData(session_id=sess_id)
    SESSIONS[sess_id] = session
    return session


def get_session(session_id: Optional[str]) -> Optional[SessionData]:
    """Retrieves an existing session if valid and not expired."""
    if not session_id or session_id not in SESSIONS:
        return None
    session = SESSIONS[session_id]
    if time.time() - session.last_accessed > SESSION_EXPIRY_SECONDS:
        del SESSIONS[session_id]
        return None
    session.touch()
    return session


def cleanup_expired_sessions():
    """Removes sessions inactive for more than 1 hour."""
    now = time.time()
    expired = [
        sid for sid, sess in SESSIONS.items()
        if now - sess.last_accessed > SESSION_EXPIRY_SECONDS
    ]
    for sid in expired:
        del SESSIONS[sid]
