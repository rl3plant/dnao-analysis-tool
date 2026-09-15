import os
import json
import logging
from typing import List, Optional, Dict
from datetime import datetime
from .data_models import Session, Annotation

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages session storage, retrieval, and persistence"""
    
    def __init__(self, sessions_dir=None):
        """Initialize the session manager
        
        Args:
            sessions_dir: Directory to store session files (defaults to user directory)
        """
        # Set default storage directory if none provided
        if sessions_dir is None:
            self.sessions_dir = os.path.join(os.path.expanduser("~"), ".dnao_sessions")
        else:
            self.sessions_dir = sessions_dir
            
        # Ensure storage directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # In-memory cache of loaded sessions
        self._sessions_cache: Dict[str, Session] = {}
        
        logger.info(f"Session manager initialized with storage at: {self.sessions_dir}")
    
    def get_session_file_path(self, session_name: str) -> str:
        """Get the file path for a session
        
        Args:
            session_name: Name of the session
            
        Returns:
            Path to the session file
        """
        # Replace invalid filename characters
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in session_name)
        return os.path.join(self.sessions_dir, f"{safe_name}.json")
    
    def list_sessions(self) -> List[str]:
        """List all available sessions
        
        Returns:
            List of session names
        """
        if not os.path.exists(self.sessions_dir):
            return []
            
        sessions = []
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith('.json'):
                # Remove .json extension to get session name
                session_name = os.path.splitext(filename)[0]
                sessions.append(session_name)
                
        return sessions
    
    def create_session(self, name: str, dnao_types: List[str] = None) -> Session:
        """Create a new session
        
        Args:
            name: Name of the session
            dnao_types: Types of DNAO expected in this session
            
        Returns:
            The newly created session
        """
        if dnao_types is None:
            dnao_types = ["intact", "damaged"]
            
        session = Session(
            name=name,
            dnao_types=dnao_types
        )
        
        # Save to cache
        self._sessions_cache[name] = session
        
        # Save to file
        self.save_session(session)
        
        return session
    
    def load_session(self, name: str) -> Optional[Session]:
        """Load a session
        
        Args:
            name: Name of the session
            
        Returns:
            The session if found, otherwise None
        """
        # Return from cache if available
        if name in self._sessions_cache:
            logger.debug(f"Returning session {name} from cache")
            return self._sessions_cache[name]
        
        # Load from file
        file_path = self.get_session_file_path(name)
        if not os.path.exists(file_path):
            logger.warning(f"No session file found at {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Convert JSON to Session object
            session = Session.from_dict(data)
            
            # Cache the results
            self._sessions_cache[name] = session
            logger.info(f"Loaded session: {name}")
            return session
            
        except Exception as e:
            logger.error(f"Error loading session from {file_path}: {e}")
            return None
    
    def save_session(self, session: Session) -> bool:
        """Save a session
        
        Args:
            session: The session to save
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self.get_session_file_path(session.name)
        
        try:
            # Update modified timestamp
            session.modified = datetime.now().isoformat()
            
            # Convert session to serializable dict
            session_data = session.to_dict()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Update cache
            self._sessions_cache[session.name] = session
            
            logger.info(f"Saved session to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session to {file_path}: {e}")
            return False
    
    def delete_session(self, name: str) -> bool:
        """Delete a session
        
        Args:
            name: Name of the session
            
        Returns:
            True if successful, False otherwise
        """
        file_path = self.get_session_file_path(name)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                
            # Remove from cache
            self.clear_cache_for_session(name)
                
            logger.info(f"Deleted session: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {name}: {e}")
            return False
    
    def clear_cache(self):
        """Clear the session cache"""
        self._sessions_cache.clear()
        logger.debug("Cleared session cache")
        
    def clear_cache_for_session(self, name: str):
        """Clear a specific session from cache
        
        Args:
            name: Name of the session to clear
        """
        if name in self._sessions_cache:
            del self._sessions_cache[name]
            logger.debug(f"Cleared session {name} from cache")
