import logging
import os
from typing import Optional, List, Dict
from PySide6.QtWidgets import QMessageBox, QInputDialog, QLineEdit
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor
from PySide6.QtCore import Qt, Signal, QObject, Slot

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_handling.session_manager import SessionManager
from data_handling.data_models import Session, TaskType
import json
import copy

logger = logging.getLogger(__name__)

class SessionController(QObject):
    """Handles session management operations"""
    
    # Define signals as class attributes
    session_changed = Signal()
    session_saved = Signal()
    session_loaded = Signal()
    image_preprocessed = Signal(str)  # Signal when an image is preprocessed
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SessionController"""
        return cls._instance
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent = parent_window
        self.session_manager = SessionManager()
        self.current_session: Optional[Session] = None
        self.current_folder: Optional[str] = None  # Keep track of most recently added folder
        self.has_unsaved_changes: bool = False
        
        # Store the instance
        SessionController._instance = self
    
    @Slot(str)
    def image_preprocessed_safe(self, image_name: str):
        """Thread-safe slot to emit the image_preprocessed signal
        
        This method will be called using QMetaObject.invokeMethod from
        a worker thread, and will safely emit the signal from the main thread.
        
        Args:
            image_name: Name of the preprocessed image
        """
        logger.info(f"Safely emitting image_preprocessed signal for {image_name} from main thread")
        self.image_preprocessed.emit(image_name)
    
    def update_session_list(self):
        """Update the session list display"""
        sessions = self.session_manager.list_sessions()
        
        # Create list model
        model = QStandardItemModel()
        for session_name in sessions:
            item = QStandardItem(session_name)
            model.appendRow(item)
            
        self.parent.lst_sessions.setModel(model)
        
        # Select the first session if one exists
        if sessions:
            first_index = model.index(0, 0)
            self.parent.lst_sessions.setCurrentIndex(first_index)
            # Preview the first session
            self.parent.preview_session(first_index)
    
    @property
    def has_active_session(self) -> bool:
        return self.current_session is not None
    
    def create_session(self, name: str, task_type: TaskType = TaskType.YIELD) -> bool:
        """Create a new session
        
        Args:
            name: Name of the session
            task_type: Type of analysis task (default: YIELD)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create new session
            self.current_session = Session(
                name=name,
                task_type=task_type
            )
            # Classifier lives next to the session's own data, not wherever the
            # app happens to be run from.
            self.current_session.set_default_classifier_path(self.session_manager.sessions_dir)

            # Save the session
            self.save_current_session()
            
            # Update the session list
            self.update_session_list()
            
            # Emit signal
            self.session_changed.emit()
            
            return True
            
        except Exception as e:
            logger.exception(f"Error creating session: {str(e)}")
            return False
    
    def save_session(self, session_file: str) -> bool:
        """Save session to file
        
        Args:
            session_file: Path to save the session file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session:
            logger.warning("No session to save")
            return False
            
        try:
            # Ensure the sessions directory exists
            os.makedirs(os.path.dirname(session_file), exist_ok=True)
            
            # Make a deep copy of the session data to avoid race conditions
            session_dict = self.current_session.to_dict()
            
            # Save to file
            with open(session_file, 'w') as f:
                json.dump(session_dict, f, indent=2)
                
            # Update the session list
            self.update_session_list()
            
            # Reset unsaved changes flag
            self.has_unsaved_changes = False
            
            # Emit session saved signal
            self.session_saved.emit()
            
            logger.info(f"Saved session to {session_file}")
            return True
            
        except Exception as e:
            logger.exception(f"Error saving session to {session_file}: {e}")
            return False
    
    def load_session(self, file_path: str) -> None:
        """Load a session from a file"""
        try:
            # Load from file
            with open(file_path, 'r') as f:
                session_data = json.load(f)
                
            # Debug logging
            logger.debug(f"Loading session data: {session_data.get('name', 'Unknown')}")
            logger.debug(f"Task type from file: {session_data.get('task_type', 'Not found')}")
            
            # Create session from data
            self.current_session = Session.from_dict(session_data)
            # Always (re)point at sessions_dir: normalizes any session saved
            # before this was fixed (bare filename, relative to whatever the
            # CWD was at the time) onto the correct location too.
            self.current_session.set_default_classifier_path(self.session_manager.sessions_dir)

            # Debug logging after loading
            logger.debug(f"Loaded session task_type: {self.current_session.task_type}, type: {type(self.current_session.task_type)}")
            
            # Update current folder to the most recently added folder if any images exist
            if self.current_session.image_files:
                last_image = self.current_session.image_files[-1]
                if last_image in self.current_session.folder_paths:
                    self.current_folder = self.current_session.folder_paths[last_image]
            
            self.session_loaded.emit()
            self.session_changed.emit()
            logger.info(f"Session loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            raise
    
    def close_session(self) -> None:
        """Close the current session"""
        self.current_session = None
        self.current_folder = None
        self.session_changed.emit()
        
    def get_image_path(self, image_name: str) -> Optional[str]:
        """Get the full path for an image in the current session"""
        if not self.current_session:
            return None
        return self.current_session.get_image_path(image_name)
    
    def get_unprocessed_images(self) -> List[Dict]:
        """Get a list of images that need preprocessing
        
        Returns:
            List of dictionaries with image name and path
        """
        if not self.current_session:
            return []
            
        queue = []
        for image_name in self.current_session.image_files:
            # Skip images that have already been preprocessed
            if image_name in self.current_session.preprocessed_images:
                continue
                
            # Construct the full path to the image
            image_path = self.current_session.get_image_path(image_name)
            if image_path:
                queue.append({
                    'name': image_name,
                    'path': image_path
                })
                
        return queue
    
    def save_current_session(self) -> bool:
        """Save the current session to its file"""
        if not self.current_session:
            logger.warning("No session to save")
            return False
            
        try:
            # Get the file path for the session
            session_file = os.path.join(self.session_manager.sessions_dir, f"{self.current_session.name}.json")
            success = self.save_session(session_file)
            
            if success:
                logger.info(f"Successfully saved session {self.current_session.name}")
                return True
            else:
                logger.error(f"Failed to save session {self.current_session.name}")
                return False
                
        except Exception as e:
            logger.exception(f"Error saving current session: {e}")
            return False
    
    def delete_session(self, session_name):
        """Delete a session"""
        # Confirm deletion
        reply = QMessageBox.question(
            self.parent,
            "Confirm Deletion",
            f"Are you sure you want to delete session '{session_name}'?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            logger.info(f"Attempting to delete session: {session_name}")
            
            # Before deleting, force clear cache for this session
            self.session_manager.clear_cache_for_session(session_name)
            logger.debug(f"Cleared cache for session: {session_name}")
            
            # Delete the session file
            success = self.session_manager.delete_session(session_name)
            
            if success:
                logger.info(f"Successfully deleted session: {session_name}")
                
                # Check if the current session was deleted
                if self.current_session and self.current_session.name == session_name:
                    logger.info("Deleted session was current session, resetting state")
                    # Reset session state
                    self.current_session = None
                    self.current_folder = None
                
                # Force clear the entire cache to ensure no stale data
                self.session_manager.clear_cache()
                logger.debug("Cleared entire session cache")
                
                # Update the session list display
                self.update_session_list()
                logger.debug("Updated session list display")
                
                # QMessageBox.information(self.parent, "Success", f"Session '{session_name}' has been deleted.")
                return True
            else:
                logger.error(f"Failed to delete session: {session_name}")
                QMessageBox.critical(self.parent, "Error", f"Failed to delete session '{session_name}'.")
        
        return False
    
    def display_session_stats(self, session: Session, list_widget):
        """Display session statistics in a list widget"""
        # Create a model for the stats list
        model = QStandardItemModel()
        
        # Add session info
        model.appendRow(QStandardItem(f"Session: {session.name}"))
        model.appendRow(QStandardItem(f"Created: {session.created[:10]}"))
        model.appendRow(QStandardItem(f"Images: {len(session.image_files)}"))
        model.appendRow(QStandardItem(f"Total DNAO: {session.get_annotation_count()}"))
        
        # Add task type
        if hasattr(session, 'task_type'):
            if session.task_type == TaskType.CHIRALITY:
                task_str = "Chirality (Z-shape/S-shape)"
                model.appendRow(QStandardItem(f"Task: {task_str}"))
                model.appendRow(QStandardItem(f"Z-shape DNAO: {session.get_z_shape_count()}"))
                model.appendRow(QStandardItem(f"S-shape DNAO: {session.get_s_shape_count()}"))
            elif session.task_type == TaskType.YIELD:
                task_str = "Yield/Quality (Intact/Defect)"
                model.appendRow(QStandardItem(f"Task: {task_str}"))
                model.appendRow(QStandardItem(f"Intact DNAO: {session.get_intact_count()}"))
                model.appendRow(QStandardItem(f"Damaged DNAO: {session.get_damaged_count()}"))
            else:
                task_str = str(session.task_type)
                model.appendRow(QStandardItem(f"Task: {task_str}"))
                model.appendRow(QStandardItem(f"Intact DNAO: {session.get_intact_count()}"))
                model.appendRow(QStandardItem(f"Damaged DNAO: {session.get_damaged_count()}"))
        else:
            # Fallback for sessions without task_type
            model.appendRow(QStandardItem(f"Intact DNAO: {session.get_intact_count()}"))
            model.appendRow(QStandardItem(f"Damaged DNAO: {session.get_damaged_count()}"))
        
        # Set the model
        list_widget.setModel(model)
