import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from PySide6.QtWidgets import (
    QStackedWidget, QFileDialog, QMessageBox, QMenu, QListWidgetItem, QInputDialog, QLineEdit,
    QLabel, QScrollArea, QSizePolicy
)
from PySide6.QtGui import (
    QPixmap, QStandardItemModel, QStandardItem, QKeySequence, QShortcut, QPainter, QBrush, QColor, QImage, QPen, QPolygon
)
from PySide6.QtCore import (
    Qt, QPoint, QTimer, Signal, QObject, QThread, QMetaObject, Q_ARG, Slot, QPointF, QRect, QSize
)
import json
import time
import cv2
from datetime import datetime
import numpy as np
import threading

# Use relative imports in a way that works when run as a script
from .DNAO_tool_ui import Ui_StackedWidget
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our new controller classes
from .session_controller import SessionController
from .annotation_controller import AnnotationController
from .image_utils import (
    scale_pixmap_to_label,
    convert_to_image_coordinates,
    is_point_in_any_annotation
)
from data_handling.data_models import Session, TaskType  # Add the missing Session import

# Import SVMClassifier, extract_mask_features, numpy, and os
from src.client.pipeline.instance_segmentation.segmentation_classification.svm_classifier import SVMClassifier
from src.client.pipeline.features_shared import extract_mask_features
from src.client.pipeline.instance_segmentation.pipeline import run_pipeline
from src.client.pipeline.downstream_tasks.chirality_classifier import fit_kmeans_chirality, assign_chirality_labels_kmeans

logger = logging.getLogger(__name__)

class PreprocessingThread(QThread):
    """Thread for preprocessing images in background"""
    progress_updated = Signal(int, int)  # current, total
    preprocessing_complete = Signal()
    preprocessing_failed = Signal(str)  # error message
    image_processed = Signal(str)  # image name that was processed
    
    def __init__(self):
        super().__init__()
        self.queue = []
        self.image_handler = None
        self.is_running = True
        self.current_index = 0
        self.session_ctrl = None
        
    def setup(self, queue, image_handler, session_ctrl=None):
        """Set up the preprocessing queue
        
        Args:
            queue: List of image info dictionaries with 'name' and 'path' keys
            image_handler: ImageHandler instance
            session_ctrl: SessionController instance
        """
        self.queue = queue
        self.image_handler = image_handler
        self.session_ctrl = session_ctrl
        self.is_running = True
        
        # Always reset the current index when setting up a new queue
        self.current_index = 0
        
        # Check for saved state in session
        if session_ctrl and session_ctrl.current_session:
            if 'preprocessing_state' in session_ctrl.current_session.properties:
                state = session_ctrl.current_session.properties['preprocessing_state']
                
                # Only restore queue position if the queue hasn't changed
                if 'total_items' in state and state['total_items'] == len(queue):
                    # Restore queue position if valid
                    if 'current_index' in state and isinstance(state['current_index'], int):
                        saved_index = state['current_index']
                        
                        # Only use the saved index if it's valid
                        if 0 <= saved_index < len(queue):
                            self.current_index = saved_index
                            logger.info(f"Resumed preprocessing queue at index {saved_index}")
                    else:
                        logger.info(f"Queue size changed from {state.get('total_items', 'unknown')} to {len(queue)}, resetting index")
    
    def run(self):
        """Process the queue (runs in background thread)"""
        try:
            total = len(self.queue)
            logger.info(f"PreprocessingThread started: Processing {total} images")
            
            while self.is_running and self.current_index < total:
                image_info = self.queue[self.current_index]
                
                try:
                    # Process the image
                    image_path = image_info['path']
                    image_name = image_info['name']
                    
                    logger.info(f"Processing image {self.current_index+1}/{total}: {image_name}")
                    
                    # Generate masks for the image - this is the CPU intensive part
                    logger.info(f"Calling generate_masks on {image_path}")
                    masks_data = self.image_handler.generate_masks(image_path)
                    
                    if not masks_data:
                        # Emit failure signal
                        error_msg = f"Error processing image {image_name}: Failed to generate masks"
                        self.preprocessing_failed.emit(error_msg)
                        logger.error(error_msg)
                    else:
                        logger.info(f"Successfully generated masks for {image_name}")
                        # Now update the session with the masks data in the main thread
                        # via signals and slots
                        self._update_session_data(image_name, masks_data)
                        
                        # Signal that this image is processed
                        logger.info(f"Emitting image_processed signal for {image_name}")
                        self.image_processed.emit(image_name)
                        
                        # Emit the progress update
                        self.current_index += 1
                        logger.info(f"Updated progress: {self.current_index}/{total}")
                        self.progress_updated.emit(self.current_index, total)
                        
                        # Save state
                        logger.debug(f"Saving state with current_index={self.current_index}")
                        self._save_state()
                    
                except Exception as e:
                    error_msg = f"Error processing image {image_info['name']}: {str(e)}"
                    self.preprocessing_failed.emit(error_msg)
                    logger.error(error_msg)
                    
                    # Move to the next item even if there was an error
                    self.current_index += 1
                    self._save_state()
            
            if self.current_index >= total:
                logger.info("Preprocessing complete, emitting preprocessing_complete signal")
                self.preprocessing_complete.emit()
                
        except Exception as e:
            error_msg = f"Unexpected error in preprocessing queue: {str(e)}"
            self.preprocessing_failed.emit(error_msg)
            logger.exception(error_msg)
    
    def _update_session_data(self, image_name, masks_data):
        """Update session data in the main thread via signal"""
        if self.session_ctrl and self.session_ctrl.current_session:
            # This should be connected to a slot in the main thread
            self.image_processed.emit(image_name)
            
            # Use thread-safe method to update data
            self.session_ctrl.current_session.set_preprocessed_image(image_name, masks_data)
    
    def _save_state(self):
        """Save the current state to the session to resume later"""
        state = {
            'current_index': self.current_index,
            'timestamp': time.time(),
            'total_items': len(self.queue),
            'is_running': self.is_running
        }
        
        # Save to session if available
        if self.session_ctrl and self.session_ctrl.current_session:
            # Store in the session properties
            self.session_ctrl.current_session.properties['preprocessing_state'] = state
            
            # Session is saved in the main thread via signal/slot
        
        # Also save to file as backup (file I/O is thread-safe)
        try:
            with open('preprocessing_state.json', 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving state to file: {str(e)}")
    
    def stop(self):
        """Stop the preprocessing process"""
        self.is_running = False

class DNAOApplication(QStackedWidget, Ui_StackedWidget):
    """Main application window for DNA Annotation Tool"""
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # Make lbl_afm_img the direct widget of the scroll area
        self.scrollArea_afm_img.takeWidget()  # Remove dummy widget
        self.lbl_afm_img.setParent(None)
        self.scrollArea_afm_img.setWidget(self.lbl_afm_img)
        self.lbl_afm_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Initialize state variables
        self.current_image_path: Optional[str] = None 
        self.current_image_name: Optional[str] = None
        self.current_pixmap: Optional[QPixmap] = None
        self.right_click_position: Optional[QPoint] = None
        self.selection_overlay: Optional[QPixmap] = None  # For drawing selection rectangle
        
        # Create controllers
        self.session_ctrl = SessionController(self)
        self.annotation_ctrl = AnnotationController(self)
        
        # Connect to session controller signals
        self.session_ctrl.image_preprocessed.connect(self.mark_image_preprocessed)
        
        # Initialize preprocessing worker - don't set a parent
        self.preprocessing_worker = None
        
        # Set up connections
        self.setup_connections()
        
        # Ensure export button is connected
        self.btn_export_annotation.clicked.connect(self.export_annotation)
        
        # Initialize UI state
        self.session_ctrl.update_session_list()
        self.btn_start_annotation.setEnabled(False)
        
                # Set context menu policy for annotation image
        self.lbl_afm_img.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lbl_afm_img.customContextMenuRequested.connect(self.show_context_menu)
        
        # Enable mouse tracking for the image label
        self.lbl_afm_img.mousePressEvent = self.image_mouse_press_event
        self.lbl_afm_img.mouseMoveEvent = self.image_mouse_move_event
        self.lbl_afm_img.mouseReleaseEvent = self.image_mouse_release_event
        
        # Explicitly set the starting page to the main window (index 0)
        self.setCurrentIndex(0)
    
    def setup_connections(self):
        """Set up signal-slot connections"""
        # Connect session controller signals
        self.session_ctrl.session_changed.connect(self.on_session_changed)
        self.session_ctrl.session_saved.connect(self.on_session_saved)
        self.session_ctrl.session_loaded.connect(self.on_session_loaded)
        
        # Main page connections
        self.btn_create_session.clicked.connect(self.create_session)
        self.btn_load_session.clicked.connect(self.load_session)
        self.btn_delete_session.clicked.connect(self.delete_session)
        self.lst_sessions.clicked.connect(self.preview_session)
        
        # Session page connections
        self.btn_load_img.clicked.connect(self.load_images)
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.btn_start_annotation.clicked.connect(self.start_annotation)
        self.btn_delete_annotation.clicked.connect(self.delete_annotation)
        self.lst_session_img.clicked.connect(self.preview_image)
        self.btn_return_to_main.clicked.connect(self.return_to_main)
        
        # Annotation page connections
        self.btn_return_session.clicked.connect(self.return_to_session)
        self.btn_save_annotation.clicked.connect(self.save_annotation)
        self.btn_previous_dnao.clicked.connect(lambda: self.annotation_ctrl.select_previous_annotation(self.lst_dnao))
        self.btn_next_dnao.clicked.connect(lambda: self.annotation_ctrl.select_next_annotation(self.lst_dnao))
        self.btn_toggle_visualization.clicked.connect(self.toggle_annotations)
        
        # Connect list view selection to update image selection - we'll connect this when a model is available
        # We can't connect to selectionModel().selectionChanged directly here because it doesn't exist yet
        
        # Add connections for label buttons - now work with multiple selections
        self.btn_set_intact.clicked.connect(lambda: self._handle_label_button_click("intact"))
        self.btn_set_defect.clicked.connect(lambda: self._handle_label_button_click("damaged"))
        self.btn_set_invalid.clicked.connect(self.annotation_ctrl.delete_selected_annotations)
        
        # Set up keyboard shortcuts with context
        self.shortcut_prev = QShortcut(QKeySequence("Left"), self)
        self.shortcut_prev.setContext(Qt.WindowShortcut)  # Make shortcut work anywhere in window
        self.shortcut_prev.activated.connect(lambda: self.annotation_ctrl.select_previous_annotation(self.lst_dnao))
        
        self.shortcut_next = QShortcut(QKeySequence("Right"), self)
        self.shortcut_next.setContext(Qt.WindowShortcut)  # Make shortcut work anywhere in window
        self.shortcut_next.activated.connect(lambda: self.annotation_ctrl.select_next_annotation(self.lst_dnao))
        
        self.shortcut_toggle = QShortcut(QKeySequence("T"), self)
        self.shortcut_toggle.setContext(Qt.WindowShortcut)  # Make shortcut work anywhere in window
        self.shortcut_toggle.activated.connect(self.toggle_annotations)
        
        # Connect list selection change to update image selection
        self.lst_dnao.clicked.connect(lambda: self._sync_list_selection_with_image())
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        session = self.session_ctrl.current_session
        is_chirality_task = session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY
        
        if event.key() == Qt.Key_1:
            # Map label according to task type
            if is_chirality_task:
                label = "Z-shape"
            else:
                label = "intact"
            self.annotation_ctrl.set_current_label(label)
            logger.info(f"Label set to {label}")
        elif event.key() == Qt.Key_2:
            # Map label according to task type
            if is_chirality_task:
                label = "S-shape"
            else:
                label = "damaged"
            self.annotation_ctrl.set_current_label(label)
            logger.info(f"Label set to {label}")
        elif event.key() == Qt.Key_3:
            self.annotation_ctrl.delete_selected_annotations()
            logger.info("Selected annotations deleted")
        elif event.key() == Qt.Key_Delete:
            self.annotation_ctrl.delete_selected_annotations()
        elif event.key() == Qt.Key_Escape:
            self.annotation_ctrl.clear_selection()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        # Ensure we maintain focus after mouse clicks in annotation view
        if self.currentIndex() == 2:
            self.setFocus()
        super().mousePressEvent(event)

    def focusOutEvent(self, event):
        """Handle focus out events"""
        # If we're in annotation view, try to maintain focus
        if self.currentIndex() == 2:
            self.setFocus()
        super().focusOutEvent(event)
    
    def return_to_main(self):
        """Return to the main view from session view"""
        # Check if we have unsaved changes in the current session
        if self.session_ctrl.current_session and self.session_ctrl.has_unsaved_changes:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes in your session. Do you want to save before returning?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            
            if reply == QMessageBox.Yes:
                # Save the session
                success = self.save_session()
                if not success:
                    QMessageBox.warning(
                        self,
                        "Save Failed",
                        "Failed to save the session. Do you still want to return?"
                    )
                    return
        
        # Clear current session data
        self.session_ctrl.current_session = None
        self.session_ctrl.current_folder = None
        self.current_image_path = None
        self.current_image_name = None
        self.current_pixmap = None
        self.annotation_ctrl.annotations = []
        self.session_ctrl.has_unsaved_changes = False
        
        # Switch to main view
        self.setCurrentIndex(0)
        
        # Update the session list to reflect any changes
        self.session_ctrl.update_session_list()
        
        logger.info("Returned to main view")
    
    def create_session(self):
        """Create a new session"""
        # Show dialog to get session name
        session_name, ok = QInputDialog.getText(
            self, 
            "Create Session", 
            "Enter session name:",
            QLineEdit.Normal
        )
        
        if ok and session_name:
            # Show task type selection dialog
            task_types = [task.value for task in TaskType]
            task_type, ok = QInputDialog.getItem(
                self,
                "Select Task Type",
                "Choose the type of analysis:",
                task_types,
                0,  # Default to first item (YIELD)
                False  # Not editable
            )
            
            if not ok:
                return
            
            # Check if a session with this name already exists
            existing_sessions = self.session_ctrl.session_manager.list_sessions()
            if session_name in existing_sessions:
                # Ask user if they want to overwrite
                reply = QMessageBox.question(
                    self,
                    "Session Exists",
                    f"A session named '{session_name}' already exists. Do you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return
                    
                # Delete the existing session
                self.session_ctrl.session_manager.delete_session(session_name)
                
            # Create session using controller with selected task type
            self.session_ctrl.create_session(session_name, TaskType(task_type))
            
            # Clear current image data and preview
            self.current_image_path = None
            self.current_image_name = None
            self.current_pixmap = None
            self.lbl_img_preview.clear()
            
            # Update label buttons based on task type
            self.update_label_buttons()
            
            # Update session list
            self.session_ctrl.update_session_list()
            
            # Switch to session view
            self.setCurrentIndex(1)
            
            # Update session details
            if self.session_ctrl.current_session:
                self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
            
            self.update_label_buttons()
    
    def load_session(self):
        """Load an existing session"""
        # Get selected session
        indexes = self.lst_sessions.selectedIndexes()
        if not indexes:
            QMessageBox.warning(self, "No Selection", "Please select a session to load.")
            return
            
        session_name = indexes[0].data()
        
        # Load session using controller
        try:
            # Get the file path for the session
            session_file = os.path.join(self.session_ctrl.session_manager.sessions_dir, f"{session_name}.json")
            self.session_ctrl.load_session(session_file)
            
            # Clear current image data and preview
            self.current_image_path = None
            self.current_image_name = None
            self.current_pixmap = None
            self.lbl_img_preview.clear()
            
            # Switch to session view
            self.setCurrentIndex(1)
            
            # Update image list and session details
            self.update_image_list()
            if self.session_ctrl.current_session:
                self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
            
            # Update the statistics labels
            self.update_statistics_labels()
            
            # Display the first image if available
            if self.session_ctrl.current_session and self.session_ctrl.current_session.image_files:
                first_image = self.session_ctrl.current_session.image_files[0]
                
                # Select and preview the first image
                model = self.lst_session_img.model()
                if model and model.rowCount() > 0:
                    first_index = model.index(0, 0)
                    self.lst_session_img.setCurrentIndex(first_index)
                    self.preview_image(first_index)
                    
            # After loading, update annotation controller for the first image if available
            if self.session_ctrl.current_session and self.session_ctrl.current_session.image_files:
                first_image = self.session_ctrl.current_session.image_files[0]
                if first_image in self.session_ctrl.current_session.annotations:
                    self.annotation_ctrl.annotations = self.session_ctrl.current_session.annotations[first_image].copy()
                    logger.info(f"[load_session] Loaded {len(self.annotation_ctrl.annotations)} annotations for {first_image} from session.")
                else:
                    self.annotation_ctrl.annotations = []
                    logger.info(f"[load_session] No annotations found for {first_image}, starting empty list.")
                
            self.update_label_buttons()
            
        except Exception as e:
            logger.exception(f"Error loading session {session_name}: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load session: {str(e)}")
    
    def preview_session(self, index):
        """Display session details in the preview area"""
        if index.isValid():
            session_name = index.data()
            
            try:
                # Get the file path for the session
                session_file = os.path.join(self.session_ctrl.session_manager.sessions_dir, f"{session_name}.json")
                
                # Load session data without setting as current
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Create session from data
                session = Session.from_dict(session_data)
                
                # Display session stats
                self.session_ctrl.display_session_stats(session, self.lst_session_stats)
            except Exception as e:
                logger.error(f"Error previewing session: {e}")
                # Set an empty model instead of calling clear()
                model = QStandardItemModel()
                self.lst_session_stats.setModel(model)
    
    def update_image_list(self):
        """Update the image list display"""
        if not self.session_ctrl.current_session:
            return
        model = QStandardItemModel()
        for image_name in self.session_ctrl.current_session.image_files:
            item = QStandardItem(image_name)
            # Always allow annotation (disable preprocessing checks)
            item.setData(QColor(150, 220, 150), Qt.BackgroundRole)  # Green
            item.setData(QColor(0, 0, 0), Qt.ForegroundRole)  # Black text
            item.setToolTip("Ready to annotate")
            model.appendRow(item)
        self.lst_session_img.setModel(model)
        current_index = self.lst_session_img.currentIndex()
        if current_index.isValid():
            self.btn_start_annotation.setEnabled(True)
            image_name = model.data(current_index)
            has_annotations = (
                image_name in self.session_ctrl.current_session.annotations and 
                len(self.session_ctrl.current_session.annotations[image_name]) > 0
            )
            self.btn_start_annotation.setText(
                "Continue Annotation" if has_annotations else "Start Annotation"
            )
        else:
            self.btn_start_annotation.setEnabled(False)
            self.btn_start_annotation.setText("Start Annotation")
    
    def load_images(self):
        """Open file dialog and load images to the session"""
        if not self.session_ctrl.current_session:
            QMessageBox.warning(self, "No Session", "Please create or load a session first.")
            return
            
        # Open file dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select AFM Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
        )
        
        if not file_paths:
            return
            
        # Remember folder location
        self.session_ctrl.current_folder = os.path.dirname(file_paths[0])
        
        # Add images to session (this will now store the folder path)
        for file_path in file_paths:
            self.session_ctrl.current_session.add_image(file_path)
        
        # Save session
        self.save_session()
        
        # Update UI
        self.update_image_list()
        self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
        
        # Display the first image in the preview
        if file_paths:
            self.current_image_path = file_paths[0]
            self.current_image_name = os.path.basename(file_paths[0])
            self.current_pixmap = QPixmap(file_paths[0])
            self.update_preview_image()
            
            # Also update the button text based on annotations
            has_annotations = (
                self.current_image_name in self.session_ctrl.current_session.annotations and 
                len(self.session_ctrl.current_session.annotations[self.current_image_name]) > 0
            )
            
            self.btn_start_annotation.setText(
                "Continue Annotation" if has_annotations else "Start Annotation"
            )
            
            # Select the first item in the list
            first_index = self.lst_session_img.model().index(0, 0)
            self.lst_session_img.setCurrentIndex(first_index)
        
        # Start preprocessing the images
        # self.start_preprocessing()
        
        logger.info(f"Added {len(file_paths)} images to session {self.session_ctrl.current_session.name}")
    
    def preview_image(self, index):
        """Preview the selected image"""
        if not self.session_ctrl.current_session:
            return
            
        # Get the model and selected model index
        model = self.lst_session_img.model()
        if not model:
            return
            
        # Get the selected index
        if isinstance(index, QPoint):
            index = self.lst_session_img.indexAt(index)
        
        # If the index is not valid, return
        if not index.isValid():
            return
            
        # Get the selected image name
        item = model.itemFromIndex(index)
        if not item:
            return
            
        # Get the image name
        image_name = item.text()
        
        # Remember the current image name
        self.current_image_name = image_name
        
        # Get image path
        image_path = self.session_ctrl.get_image_path(image_name)
        if not image_path:
            self.lbl_img_preview.clear()
            self.current_pixmap = None
            self.current_image_path = None
            logger.error(f"Could not find image path for {image_name}")
            return
            
        # Update current image path
        self.current_image_path = image_path
        
        # Load image
        try:
            # Load as QPixmap
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                logger.error(f"Failed to load image {image_path}")
                return
                
            # Set the current pixmap
            self.current_pixmap = pixmap
            
            # Display preview
            self.update_preview_image()
            
            # --- UPDATE ANNOTATION CONTROLLER FROM SESSION ---
            # Always update the annotation controller's annotations from the session for this image
            if image_name in self.session_ctrl.current_session.annotations:
                self.annotation_ctrl.annotations = self.session_ctrl.current_session.annotations[image_name].copy()
                logger.info(f"[preview_image] Loaded {len(self.annotation_ctrl.annotations)} annotations for {image_name} from session.")
            else:
                self.annotation_ctrl.annotations = []
                logger.info(f"[preview_image] No annotations found for {image_name}, starting empty list.")
            
            # Always allow annotation (disable preprocessing checks)
            self.btn_start_annotation.setText("Start/Continue Annotation")
            self.btn_start_annotation.setEnabled(True)
            self.lbl_preprocessing_status.setText("Ready to annotate")
            self.lbl_preprocessing_status.setStyleSheet("color: green;")
            
            # Update label buttons based on task type
            self.update_label_buttons()
        except Exception as e:
            logger.error(f"Error previewing image {image_path}: {e}")
            self.lbl_img_preview.clear()
            self.current_pixmap = None
    
    def update_preview_image(self):
        if self.current_pixmap:
            # Check which view we're currently in
            current_index = self.currentIndex()
            
            if current_index == 1:  # Session view (pg_session)
                # Use the session view preview label
                available_size = self.lbl_img_preview.size()
                scaled_pixmap = self.current_pixmap.scaled(
                    available_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.lbl_img_preview.setPixmap(scaled_pixmap)
                self.lbl_img_preview.setAlignment(Qt.AlignCenter)
            elif current_index == 2:  # Annotation view (pg_img)
                # Use the annotation view scroll area and label
                available_size = self.scrollArea_afm_img.viewport().size()
                scaled_pixmap = self.current_pixmap.scaled(
                    available_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.lbl_afm_img.setPixmap(scaled_pixmap)
                self.lbl_afm_img.setAlignment(Qt.AlignCenter)

    def start_annotation(self):
        """Start or continue annotation for the selected image"""
        if not self.session_ctrl.current_session:
            QMessageBox.warning(
                self,
                "No Session",
                "Please create or load a session first."
            )
            return
            
        # Get the selected image
        current_index = self.lst_session_img.currentIndex()
        if not current_index.isValid():
            QMessageBox.warning(
                self,
                "No Image Selected",
                "Please select an image to annotate."
            )
            return
            
        # Get the image name from the model
        model = self.lst_session_img.model()
        if not model:
            QMessageBox.warning(
                self,
                "Error",
                "Failed to get image list model."
            )
            return
            
        image_name = model.data(current_index)
        if not image_name:
            QMessageBox.warning(
                self,
                "Error",
                "Failed to get image name."
            )
            return
            
        # Find the path to the image
        file_path = None
        
        # First, try the path stored in the session's folder_paths
        if image_name in self.session_ctrl.current_session.folder_paths:
            folder_path = self.session_ctrl.current_session.folder_paths[image_name]
            potential_path = os.path.join(folder_path, image_name)
            if os.path.exists(potential_path):
                file_path = potential_path
        # If we have a direct path from previous operations, check if it's the right image
        elif self.current_image_path and os.path.basename(self.current_image_path) == image_name:
            file_path = self.current_image_path
        # Otherwise try the current folder
        elif self.session_ctrl.current_folder:
            potential_path = os.path.join(self.session_ctrl.current_folder, image_name)
            if os.path.exists(potential_path):
                file_path = potential_path
                
        # If we still don't have a valid path, we need user input
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Image not found for annotation: {image_name}. Folder paths: {self.session_ctrl.current_session.folder_paths}")
            
            # Let's try to browse for the image
            QMessageBox.information(
                self, 
                "Image Not Found", 
                f"Image {image_name} not found. Please select the image file."
            )
            
            # Open file dialog to locate the image
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {image_name}",
                "",
                "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Update the folder path in the session
            self.session_ctrl.current_session.folder_paths[image_name] = os.path.dirname(file_path)
            self.session_ctrl.current_folder = os.path.dirname(file_path)
            self.save_session()
            
        # Load the image
        self.current_pixmap = QPixmap(file_path)
        if self.current_pixmap.isNull():
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load image: {file_path}"
            )
            return
            
        # Set focus to the window when entering annotation view
        self.setFocus()
        
        # Switch to annotation view
        self.setCurrentIndex(2)
        
        # Force layout update
        self.grp_afm_img.updateGeometry()
        self.lbl_afm_img.updateGeometry()
        
        # Scale the image to fit the label
        scaled_pixmap = scale_pixmap_to_label(self.current_pixmap, self.lbl_afm_img)
        self.lbl_afm_img.setPixmap(scaled_pixmap)
        
        # --- NEW: Use segmentation pipeline if no existing annotations ---
        if image_name in self.session_ctrl.current_session.annotations and \
           len(self.session_ctrl.current_session.annotations[image_name]) > 0:
            # Load existing annotations, do NOT overwrite
            self.annotation_ctrl.annotations = self.session_ctrl.current_session.annotations[image_name].copy()
            logger.info(f"Loaded {len(self.annotation_ctrl.annotations)} existing annotations for {image_name}")
        else:
            # No existing annotations, generate from segmentation pipeline
            logger.info(f"Running segmentation pipeline for {image_name}")
            masks_data = run_pipeline(file_path, self.session_ctrl.current_session)
            if not masks_data:
                logger.error(f"Failed to generate masks for {image_name} using segmentation pipeline")
                QMessageBox.warning(
                    self,
                    "Processing Error",
                    "Failed to generate segmentations. Some annotations may be missing."
                )
            success = self.annotation_ctrl.process_masks_from_preprocessed(masks_data, file_path)
            if not success:
                logger.error(f"Failed to process masks for {image_name}")
                QMessageBox.warning(
                    self,
                    "Processing Error",
                    "Failed to process segmentation results. Some annotations may be missing."
                )
            # Save these as the initial annotations
            self.session_ctrl.current_session.annotations[image_name] = self.annotation_ctrl.annotations.copy()
            self.save_session()
            
            # Train KMeans for chirality classification if we have enough annotations and it's a chirality task
            session = self.session_ctrl.current_session
            is_chirality_task = session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY
            
            if is_chirality_task:
                try:
                    # First, assign chirality labels to the current image's annotations (the ones displayed in UI)
                    if len(self.annotation_ctrl.annotations) >= 2:
                        logger.info(f"Training KMeans for chirality classification with {len(self.annotation_ctrl.annotations)} annotations")
                        kmeans, features = fit_kmeans_chirality(self.annotation_ctrl.annotations)
                        
                        # Store KMeans model in session
                        self.session_ctrl.current_session._kmeans_model = kmeans
                        
                        # Assign chirality labels to the annotations that are displayed in the UI
                        assign_chirality_labels_kmeans(kmeans, self.annotation_ctrl.annotations, features)
                        
                        # Update the session annotations with the new labels
                        self.session_ctrl.current_session.annotations[image_name] = self.annotation_ctrl.annotations.copy()
                        
                        logger.info("KMeans chirality classification completed successfully")
                    else:
                        logger.info(f"Not enough annotations ({len(self.annotation_ctrl.annotations)}) to train KMeans for chirality classification")
                except Exception as e:
                    logger.warning(f"Failed to train KMeans for chirality classification: {e}")
            else:
                logger.info("Skipping KMeans training for non-chirality task")
        
        # Update current image info to ensure it's correct
        self.current_image_path = file_path
        self.current_image_name = image_name
        
                # Always embed the image for manual annotation features
        try:
            if not self.annotation_ctrl.image_handler.embed_image(file_path):
                logger.warning("Failed to embed image in SAM2 model. Manual annotation features may not work.")
                QMessageBox.warning(
                    self,
                    "Server Connection Error",
                    "Failed to connect to the server. Manual annotation features will not be available.\n"
                    "Please check your server connection and try again."
                )
        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            QMessageBox.warning(
                self,
                "Server Connection Error",
                "Failed to connect to the server. Manual annotation features will not be available.\n"
                "Please check your server connection and try again."
            )
        
        # Ensure annotations are visible by default
        self.annotation_ctrl.show_annotations = True
        self.btn_toggle_visualization.setText("Hide Annotations (t)")
        
        # Update the annotation list
        self.annotation_ctrl.update_annotations_list(self.lst_dnao)
        
        # Update the statistics labels
        self.update_statistics_labels()
        
        # Update label buttons based on task type
        self.update_label_buttons()
        
        # Enable the save button
        self.btn_save_annotation.setEnabled(True)
        
        # Enable the export button
        self.btn_export_annotation.setEnabled(True)
        
        # Refresh the display to show annotations
        self.refresh_annotation_display()
        
           
    def refresh_annotation_display(self):
        """Update the annotation display and list"""
        # Get the currently selected annotation index
        current_index = self.lst_dnao.currentIndex()
        selected_index = current_index.row() if current_index.isValid() else -1
        
        # Store the current background color if it exists
        current_bg_color = None
        if hasattr(self.lst_dnao, 'palette'):
            current_bg_color = self.lst_dnao.palette().color(self.lst_dnao.backgroundRole())
        
        # Display annotations with the selected index
        result_pixmap = self.annotation_ctrl.display_annotations(
            self.current_pixmap, 
            self.lbl_afm_img,
            selected_index
        )
        
        if result_pixmap:
            # Scale the pixmap to fit the scroll area viewport, keeping aspect ratio
            viewport_size = self.scrollArea_afm_img.viewport().size()
            scaled_pixmap = result_pixmap.scaled(
                viewport_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.lbl_afm_img.setPixmap(scaled_pixmap)
            self.lbl_afm_img.adjustSize()
            # Update the annotations list
            self.annotation_ctrl.update_annotations_list(self.lst_dnao)
            # Restore the background color if it existed
            if current_bg_color:
                palette = self.lst_dnao.palette()
                palette.setColor(self.lst_dnao.backgroundRole(), current_bg_color)
                self.lst_dnao.setPalette(palette)
        
        # --- CLOSEUP LOGIC ---
        # Show closeup of selected annotation in lbl_dnao_closeup
        closeup_pixmap = None
        if (
            selected_index is not None and selected_index >= 0 and 
            selected_index < len(self.annotation_ctrl.annotations)
        ):
            ann = self.annotation_ctrl.annotations[selected_index]
            contour = ann.contour
            if contour is not None and len(contour) > 0:
                try:
                    # Prefer cropping from the loaded QPixmap
                    if self.current_pixmap is not None:
                        x, y, w, h = cv2.boundingRect(contour)
                        pad = 10
                        x1 = max(x - pad, 0)
                        y1 = max(y - pad, 0)
                        x2 = min(x + w + pad, self.current_pixmap.width())
                        y2 = min(y + h + pad, self.current_pixmap.height())
                        crop_w = x2 - x1
                        crop_h = y2 - y1
                        if crop_w > 0 and crop_h > 0:
                            closeup_pixmap = self.current_pixmap.copy(x1, y1, crop_w, crop_h)
                            # Draw contour if toggled on
                            if self.annotation_ctrl.show_annotations:
                                painter = QPainter(closeup_pixmap)
                                # Get session task type to determine color scheme
                                session = self.session_ctrl.current_session if hasattr(self, 'session_ctrl') else None
                                is_chirality_task = session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY
                                
                                # Color logic based on task type
                                if is_chirality_task:
                                    # Chirality colors
                                    if ann.class_label == "Z-shape":
                                        pen_color = QColor(255, 140, 0)  # Orange
                                    elif ann.class_label == "S-shape":
                                        pen_color = QColor(0, 0, 200)  # Blue
                                    else:
                                        pen_color = QColor(64, 64, 64)  # Gray
                                else:
                                    # Yield colors
                                    if ann.class_label == "intact":
                                        pen_color = QColor(0, 128, 0)  # Green
                                    elif ann.class_label == "damaged":
                                        pen_color = QColor(128, 0, 0)  # Red
                                    else:
                                        pen_color = QColor(64, 64, 64)  # Gray
                                pen = QPen(pen_color, 2)
                                painter.setPen(pen)
                                # Shift contour to crop coordinates
                                points = [QPoint(int(p[0]) - x1, int(p[1]) - y1) for p in contour]
                                if len(points) >= 3:
                                    painter.drawPolygon(QPolygon(points))
                                elif len(points) == 2:
                                    painter.drawLine(points[0], points[1])
                                painter.end()
                            closeup_pixmap = closeup_pixmap.scaled(
                                self.lbl_dnao_closeup.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                            )
                    # Fallback: if QPixmap is not available, use cv2.imread
                    elif self.current_image_path and os.path.exists(self.current_image_path):
                        img = cv2.imread(self.current_image_path)
                        if img is not None:
                            x, y, w, h = cv2.boundingRect(contour)
                            pad = 10
                            x1 = max(x - pad, 0)
                            y1 = max(y - pad, 0)
                            x2 = min(x + w + pad, img.shape[1])
                            y2 = min(y + h + pad, img.shape[0])
                            crop = img[y1:y2, x1:x2]
                            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            h_crop, w_crop, ch = crop.shape
                            bytes_per_line = ch * w_crop
                            qimg = QImage(crop.data, w_crop, h_crop, bytes_per_line, QImage.Format_RGB888)
                            closeup_pixmap = QPixmap.fromImage(qimg)
                            # Draw contour if toggled on
                            if self.annotation_ctrl.show_annotations:
                                painter = QPainter(closeup_pixmap)
                                # Get session task type to determine color scheme
                                session = self.session_ctrl.current_session if hasattr(self, 'session_ctrl') else None
                                is_chirality_task = session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY
                                
                                # Color logic based on task type
                                if is_chirality_task:
                                    # Chirality colors
                                    if ann.class_label == "Z-shape":
                                        pen_color = QColor(255, 140, 0)  # Orange
                                    elif ann.class_label == "S-shape":
                                        pen_color = QColor(0, 0, 200)  # Blue
                                    else:
                                        pen_color = QColor(64, 64, 64)  # Gray
                                else:
                                    # Yield colors
                                    if ann.class_label == "intact":
                                        pen_color = QColor(0, 128, 0)  # Green
                                    elif ann.class_label == "damaged":
                                        pen_color = QColor(128, 0, 0)  # Red
                                    else:
                                        pen_color = QColor(64, 64, 64)  # Gray
                                pen = QPen(pen_color, 2)
                                painter.setPen(pen)
                                # Shift contour to crop coordinates
                                points = [QPoint(int(p[0]) - x1, int(p[1]) - y1) for p in contour]
                                if len(points) >= 3:
                                    painter.drawPolygon(QPolygon(points))
                                elif len(points) == 2:
                                    painter.drawLine(points[0], points[1])
                                painter.end()
                            closeup_pixmap = closeup_pixmap.scaled(
                                self.lbl_dnao_closeup.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                            )
                except Exception as e:
                    logger.warning(f"Failed to create closeup: {e}")
        # Set or clear the closeup label
        if closeup_pixmap:
            self.lbl_dnao_closeup.setPixmap(closeup_pixmap)
            self.lbl_dnao_closeup.setAlignment(Qt.AlignCenter)
        else:
            self.lbl_dnao_closeup.clear()

    def return_to_session(self):
        """Return to the session view"""
        # Ask to save only if there are unsaved annotations
        if self.annotation_ctrl.has_unsaved_changes and self.annotation_ctrl.annotations and self.current_image_name:
            reply = QMessageBox.question(
                self, 
                "Save Annotations", 
                "Do you want to save your annotations before returning?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            
            if reply == QMessageBox.Yes:
                self.save_annotation()
        
        # Switch back to session view
        self.setCurrentIndex(1)
        
        # Update the image list to show latest preprocessing status
        self.update_image_list()
        
        # Update the statistics labels
        self.update_statistics_labels()
        
        # Continue preprocessing in the background if needed
        if self.session_ctrl.current_session:
            # self.start_preprocessing()
            pass
    
    def save_annotation(self):
        """Save the current annotations to the session"""
        if not self.annotation_ctrl.annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                "There are no annotations to save."
            )
            return
        
        if not self.session_ctrl.current_session or not self.current_image_name:
            QMessageBox.warning(
                self,
                "Error",
                "No session or image selected to save annotations for."
            )
            return
        
        # Skip saving if no changes were made
        if not self.annotation_ctrl.has_unsaved_changes:
            logger.info("No changes to save")
            return
        
        try:
            # Update the session with the current annotations
            self.session_ctrl.current_session.annotations[self.current_image_name] = self.annotation_ctrl.annotations.copy()
            
            # Save the session
            success = self.save_session()

            # --- Classifier retraining logic ---
            session = self.session_ctrl.current_session
            if success and session:
                session.increment_retrain_counter()
            if success and session and session.should_retrain_classifier():
                try:
                    # Gather all features and labels from all images in the session
                    features = []
                    labels = []
                    for img_name, anns in session.annotations.items():
                        # Try to get the image path for this image
                        img_path = session.get_image_path(img_name)
                        for ann in anns:
                            # For now, we only use contour-based features (no image needed)
                            mask_dict = {'contours': [ann.contour], 'area': ann.properties.get('area', 0.0), 'bbox': ann.properties.get('bbox', [0,0,0,0])}
                            features.append(extract_mask_features(mask_dict))
                            labels.append(ann.class_label)
                    if features and labels:
                        features = np.stack(features)
                        # Use or create the classifier instance
                        if not hasattr(session, '_classifier_instance'):
                            session._classifier_instance = SVMClassifier()
                        classifier = session._classifier_instance
                        classifier.fit(features, labels)
                        # Save classifier to disk
                        classifier_path = session.get_classifier_path()
                        classifier.save(classifier_path)
                        session.reset_retrain_counter()
                        logger.info(f"Classifier retrained and saved to {classifier_path}")
                except Exception as e:
                    logger.warning(f"Classifier retraining failed: {e}")

            if success:
                # Reset the unsaved changes flag
                self.annotation_ctrl.has_unsaved_changes = False
                
                # Update the session statistics display
                self.session_ctrl.display_session_stats(
                    self.session_ctrl.current_session, 
                    self.lst_session_stats
                )
                
                # Update the statistics labels
                self.update_statistics_labels()
                
                QMessageBox.information(
                    self,
                    "Save Complete",
                    f"Annotations saved for {self.current_image_name}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Save Failed",
                    "Failed to save annotations."
                )
                
        except Exception as e:
            logger.exception(f"Error saving annotations: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Error saving annotations: {str(e)}"
            )

    def show_context_menu(self, position):
        """Show the context menu for annotations"""
        if self.currentIndex() != 2:  # Only show menu in annotation view
            return
            
        menu = QMenu()
        
        # Add DNAO action
        add_action = menu.addAction("Add DNAO")
        add_action.triggered.connect(self.add_dnao_at_position)
        
        # Remove DNAO action (only if we have annotations)
        if self.annotation_ctrl.annotations:
            remove_action = menu.addAction("Remove DNAO")
            remove_action.triggered.connect(self.remove_dnao_at_position)
            
            # Add fuse action if multiple annotations are selected
            if len(self.annotation_ctrl.selected_indices) >= 2:
                fuse_action = menu.addAction("Fuse Selected DNAOs")
                fuse_action.triggered.connect(self.fuse_selected_dnaos)
        
        menu.exec_(self.lbl_afm_img.mapToGlobal(position))
    
    def add_dnao_at_position(self):
        """Add a new DNAO annotation at the clicked position"""
        if not self.current_image_path:
            return
            
        # Always make sure selection rectangle is correctly updated and we have a valid position
        position = self.right_click_position
        if position is None:
            # Use center of image as fallback
            position = QPoint(self.lbl_afm_img.width() // 2, self.lbl_afm_img.height() // 2)
            
        logger.info(f"Adding DNAO at position: {position} with selection rectangle: {self.annotation_ctrl.selection_rect}")
        
        success = self.annotation_ctrl.add_annotation_at_position(
            position,
            self.current_image_path,
            self.lbl_afm_img,
            self.current_pixmap
        )
        
        if success:
            self.refresh_annotation_display()
            # Update session stats display if we're in session view
            if self.currentIndex() == 1 and self.session_ctrl.current_session:
                self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
        else:
            pass
            # QMessageBox.warning(
            #     self,
            #     "Add Failed",
            #     "Failed to add annotation at the selected position."
            # )
    
    def remove_dnao_at_position(self):
        """Remove DNAO at the right-clicked position"""
        if self.right_click_position:
            if self.annotation_ctrl.remove_selected_annotations():
                self.refresh_annotation_display()
                self.annotation_ctrl.update_annotations_list(self.lst_dnao)
                # Update session stats display if we're in session view
                if self.currentIndex() == 1 and self.session_ctrl.current_session:
                    self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
    
    def fuse_selected_dnaos(self):
        """Fuse selected DNAOs into one"""
        if self.annotation_ctrl.fuse_selected_annotations():
            self.refresh_annotation_display()
            self.annotation_ctrl.update_annotations_list(self.lst_dnao)
            # Update session stats display if we're in session view
            if self.currentIndex() == 1 and self.session_ctrl.current_session:
                self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)

    def delete_session(self):
        """Delete the selected session"""
        indexes = self.lst_sessions.selectedIndexes()
        if not indexes:
            QMessageBox.warning(self, "No Selection", "Please select a session to delete.")
            return
            
        session_name = indexes[0].data()
        
        # Delete session using controller
        success = self.session_ctrl.delete_session(session_name)
        
        if success:
            # Check if the current session was deleted
            if self.session_ctrl.current_session and self.session_ctrl.current_session.name == session_name:
                # Reset everything
                self.current_image_path = None
                self.current_image_name = None
                self.current_pixmap = None
                self.lbl_img_preview.clear()
                self.lbl_img_preview.setText("No image selected")
                self.annotation_ctrl.annotations = []
                self.session_ctrl.current_session = None
                self.session_ctrl.current_folder = None
                self.session_ctrl.has_unsaved_changes = False
                
                # Clear the session image list
                model = QStandardItemModel()
                self.lst_session_img.setModel(model)
                
                # Clear the session stats
                self.lst_session_stats.clear()
                
                # Return to main view
                self.setCurrentIndex(0)
            
            # Force refresh the session list
            self.session_ctrl.update_session_list()
            
            # Clear any cached data
            self.session_ctrl.session_manager.clear_cache()
            
            # Reset all UI elements that might retain state
            self.btn_start_annotation.setEnabled(False)
            self.btn_start_annotation.setText("Start Annotation")
            self.btn_save_annotation.setEnabled(False)
            self.btn_export_annotation.setEnabled(False)
            
            # Clear any remaining image data
            self.lbl_afm_img.clear()
            self.lbl_afm_img.setText("No image loaded")
            
            # Reset annotation list
            model = QStandardItemModel()
            self.lst_dnao.setModel(model)
            
            # Reset statistics labels
            self.lbl_intact_dnao.setText("Intact: 0")
            self.lbl_defect_dnao.setText("Defect: 0")
            self.lbl_total_dnao.setText("Total: 0")
            self.lbl_yield_dnao.setText("Yield: 0.0%")
            
            # Clear the session details display in the main window
            model = QStandardItemModel()
            self.lst_session_stats.setModel(model)
            
            # Reset preview statistics labels
            self.lbl_intact_preview.setText("Intact: 0")
            self.lbl_defect_preview.setText("Defect: 0")
            self.lbl_total_preview.setText("Total: 0")
            self.lbl_yield_preview.setText("Yield: 0.0%")
            
            # Always clear the image list view, regardless of whether it was the current session
            model = QStandardItemModel()
            self.lst_session_img.setModel(model)
            self.lst_session_img.clearSelection()

    def delete_annotation(self):
        """Delete the selected image and its annotations from the current session"""
        if not self.session_ctrl.current_session:
            QMessageBox.warning(self, "No Session", "Please load a session first.")
            return
            
        indexes = self.lst_session_img.selectedIndexes()
        if not indexes:
            QMessageBox.warning(self, "No Selection", "Please select an image to delete.")
            return
            
        image_name = indexes[0].data()
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete '{image_name}' and all its annotations?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove the image from the session
            self.session_ctrl.current_session.remove_image(image_name)
            
            # Save the changes
            if self.save_session():
                # Always clear the preview if this was the current image or not
                if self.current_image_name == image_name:
                    self.current_image_name = None
                    self.current_image_path = None
                    self.current_pixmap = None
                    self.lbl_img_preview.clear()
                    self.lbl_img_preview.setText("No image selected")
                
                # Update the UI
                self.update_image_list()
                self.session_ctrl.display_session_stats(
                    self.session_ctrl.current_session, 
                    self.lst_session_stats
                )
                
                # If there are other images available, select the first one
                model = self.lst_session_img.model()
                if model and model.rowCount() > 0:
                    first_index = model.index(0, 0)
                    self.lst_session_img.setCurrentIndex(first_index)
                    self.preview_image(first_index)
                
                # Restart preprocessing to ensure the queue is updated
                # self.start_preprocessing()
                
                QMessageBox.information(self, "Success", f"Image '{image_name}' has been deleted.")
            else:
                QMessageBox.critical(self, "Error", f"Failed to delete image '{image_name}'.")

    def image_mouse_press_event(self, event):
        """Handle mouse press events on the image"""
        if self.currentIndex() != 2:  # Only handle events in annotation view
            return
            
        # Store the current position for potential menu operations
        self.right_click_position = event.pos()
        
        # Left button for selection
        if event.button() == Qt.LeftButton:
            # Store the initial press position for drag detection
            self.drag_start_pos = event.pos()
            self.is_dragging = False  # Start with no drag
            
            # Let the annotation controller handle the event
            self.annotation_ctrl.image_mouse_press_event(event)
            
            # Refresh the display to show selection effects
            self.refresh_annotation_display()
            
            # Ensure window has focus after interaction
            self.setFocus()
            
        # Right button for context menu
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.pos())

    def image_mouse_move_event(self, event):
        """Handle mouse move events on the image"""
        if self.currentIndex() != 2:  # Only handle events in annotation view
            return
            
        if hasattr(self, 'drag_start_pos'):
            # Calculate distance moved
            dx = event.pos().x() - self.drag_start_pos.x()
            dy = event.pos().y() - self.drag_start_pos.y()
            distance = (dx * dx + dy * dy) ** 0.5
            
            # If we've moved more than a small threshold, start the selection
            if distance > 5:  # 5 pixel threshold
                if not self.is_dragging:
                    logger.info("Starting selection due to drag")
                    self.is_dragging = True
                    # Start the selection at the initial press position
                    self.annotation_ctrl.start_selection(self.drag_start_pos)
                
                # Update the selection
                self.annotation_ctrl.update_selection(event.pos())
                self.refresh_annotation_display()

    def image_mouse_release_event(self, event):
        """Handle mouse release events on the image"""
        if self.currentIndex() != 2:  # Only handle events in annotation view
            return
            
        if event.button() == Qt.LeftButton:
            if hasattr(self, 'is_dragging') and self.is_dragging:
                logger.info("End selection on mouse release")
                self.annotation_ctrl.end_selection()
                
                # Refresh display to show the selection and selection rectangle
                self.refresh_annotation_display()
                
                # Update the list selection to match image selection
                self._sync_list_selection_with_image()
            
            # Clean up drag state
            if hasattr(self, 'drag_start_pos'):
                delattr(self, 'drag_start_pos')
            if hasattr(self, 'is_dragging'):
                delattr(self, 'is_dragging')

    def toggle_annotations(self):
        """Toggle the visibility of annotations"""
        is_visible = self.annotation_ctrl.toggle_annotations()
        self.btn_toggle_visualization.setText("Show Annotations (t)" if not is_visible else "Hide Annotations (t)")
        self.refresh_annotation_display()

    def export_annotation(self):
        """Export the current annotated image with legend"""
        logger.info("Export annotation method called")
        
        if not self.current_pixmap or not self.current_image_name:
            logger.warning("No image loaded for export")
            QMessageBox.warning(
                self,
                "No Image",
                "No image is currently loaded to export."
            )
            return
            
        # Ask for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Annotated Image",
            f"{os.path.splitext(self.current_image_name)[0]}_annotated.png",
            "PNG Images (*.png);;All Files (*)"
        )
        
        if not file_path:
            logger.info("Export cancelled by user")
            return
            
        try:
            logger.info(f"Exporting annotation to {file_path}")
            
            # Get the current display with annotations
            logger.info("Calling display_annotations method")
            result_pixmap = self.annotation_ctrl.display_annotations(
                self.current_pixmap, 
                self.lbl_afm_img,
                -1  # No selected annotation
            )
            
            if not result_pixmap:
                logger.error("Failed to generate annotated image")
                QMessageBox.warning(
                    self,
                    "Export Failed",
                    "Failed to generate annotated image."
                )
                return
                
            logger.info("Creating export pixmap with legend")
            # Create a new pixmap with space for the legend
            legend_height = 150  # Increased height for the legend
            total_height = result_pixmap.height() + legend_height
            export_pixmap = QPixmap(result_pixmap.width(), total_height)
            
            # Use a light gray background instead of white
            export_pixmap.fill(QColor(240, 240, 240))
            
            # Create painter for the export pixmap
            painter = QPainter(export_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw the annotated image at the bottom
            painter.drawPixmap(0, legend_height, result_pixmap)
            
            # Draw the legend at the top
            font = painter.font()
            font.setPointSize(12)
            painter.setFont(font)
            
            # Draw legend title
            painter.setPen(Qt.black)
            painter.drawText(10, 25, "DNAO Annotation Legend")
            
            # Count annotations by type (excluding invalid)
            session = self.session_ctrl.current_session
            if session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY:
                z_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "Z-shape")
                s_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "S-shape")
                total_count = z_count + s_count
                legend_items = [
                    ("Z-shape", QColor(0, 128, 255), z_count),
                    ("S-shape", QColor(255, 128, 0), s_count)
                ]
                yield_percentage = None
            else:
                intact_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "intact")
                damaged_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "damaged")
                total_count = intact_count + damaged_count
                legend_items = [
                    ("Intact", QColor(0, 255, 0), intact_count),
                    ("Damaged", QColor(255, 0, 0), damaged_count)
                ]
                yield_percentage = (intact_count / total_count * 100) if total_count > 0 else 0
            
            logger.info(f"Annotation counts: Intact={intact_count}, Damaged={damaged_count}, Total={total_count}")
            
            # Draw legend items with counts
            x_pos = 10
            y_pos = 50
            box_size = 20
            
            for i, (label, color, count) in enumerate(legend_items):
                # Draw color box
                painter.setPen(Qt.black)
                painter.setBrush(QBrush(color))
                painter.drawRect(x_pos, y_pos, box_size, box_size)
                
                # Draw label and count
                painter.drawText(x_pos + box_size + 10, y_pos + 15, f"{label}: {count}")
                
                # Move to next position
                x_pos += 150
                
                # If we've drawn 2 items, move to next row
                if i == 1:
                    x_pos = 10
                    y_pos += 30
            
            # Draw statistics
            y_pos += 20  # Add some space before statistics
            
            # Draw total count
            painter.setPen(Qt.black)
            painter.drawText(10, y_pos + 15, f"Total: {total_count}")
            
            # Draw yield percentage
            if yield_percentage is not None:
                painter.drawText(200, y_pos + 15, f"Yield: {yield_percentage:.1f}%")
            
            # End painting
            painter.end()
            
            # Save the pixmap
            logger.info(f"Saving pixmap to {file_path}")
            export_pixmap.save(file_path)
            
            logger.info("Export completed successfully")
            QMessageBox.information(
                self,
                "Export Complete",
                f"Annotated image saved to {file_path}"
            )
            
        except Exception as e:
            logger.exception(f"Error exporting annotation: {str(e)}")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Error exporting annotation: {str(e)}"
            )

    def update_statistics_labels(self):
        """Update the statistics labels with current annotation counts"""
        session = self.session_ctrl.current_session
        if not session:
            return
            
        if session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY:
            # Update the labels based on current view
            if self.currentIndex() == 2:  # Image view - show current image statistics
                z_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "Z-shape")
                s_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "S-shape")
                total_count = z_count + s_count
                self.lbl_intact_dnao.setText(f"Z-shape: {z_count}")
                self.lbl_defect_dnao.setText(f"S-shape: {s_count}")
                self.lbl_total_dnao.setText(f"Total: {total_count}")
                self.lbl_yield_dnao.setText("")
            elif self.currentIndex() == 1:  # Session view - show session-wide statistics
                z_count = session.get_z_shape_count()
                s_count = session.get_s_shape_count()
                total_count = z_count + s_count
                self.lbl_intact_preview.setText(f"Z-shape: {z_count}")
                self.lbl_defect_preview.setText(f"S-shape: {s_count}")
                self.lbl_total_preview.setText(f"Total: {total_count}")
                self.lbl_yield_preview.setText("")
        else:
            # Update the labels based on current view
            if self.currentIndex() == 2:  # Image view - show current image statistics
                intact_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "intact")
                damaged_count = sum(1 for ann in self.annotation_ctrl.annotations if ann.class_label == "damaged")
                total_count = intact_count + damaged_count
                # Calculate yield (percentage of intact DNAOs)
                yield_percentage = (intact_count / total_count * 100) if total_count > 0 else 0
                self.lbl_intact_dnao.setText(f"Intact: {intact_count}")
                self.lbl_defect_dnao.setText(f"Defect: {damaged_count}")
                self.lbl_total_dnao.setText(f"Total: {total_count}")
                self.lbl_yield_dnao.setText(f"Yield: {yield_percentage:.1f}%")
            elif self.currentIndex() == 1:  # Session view - show session-wide statistics
                intact_count = session.get_intact_count()
                damaged_count = session.get_damaged_count()
                total_count = intact_count + damaged_count
                # Calculate yield (percentage of intact DNAOs)
                yield_percentage = (intact_count / total_count * 100) if total_count > 0 else 0
                self.lbl_intact_preview.setText(f"Intact: {intact_count}")
                self.lbl_defect_preview.setText(f"Defect: {damaged_count}")
                self.lbl_total_preview.setText(f"Total: {total_count}")
                self.lbl_yield_preview.setText(f"Yield: {yield_percentage:.1f}%")

    def load_folder(self):
        """Open folder dialog and load all images from the folder to the session"""
        if not self.session_ctrl.current_session:
            QMessageBox.warning(self, "No Session", "Please create or load a session first.")
            return
            
        # Open folder dialog
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with AFM Images",
            ""
        )
        
        if not folder_path:
            return
            
        # Remember folder location
        self.session_ctrl.current_folder = folder_path
        
        # Get all image files in the folder
        image_files = self.annotation_ctrl.image_handler.get_image_files_in_folder(folder_path)
        
        if not image_files:
            QMessageBox.warning(self, "No Images", "No image files found in the selected folder.")
            return
        
        # Add images to session
        for image_name in image_files:
            file_path = os.path.join(folder_path, image_name)
            self.session_ctrl.current_session.add_image(file_path)
        
        # Save session
        self.save_session()
        
        # Update UI
        self.update_image_list()
        self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
        
        # Display the first image in the preview if available
        if image_files:
            first_image_path = os.path.join(folder_path, image_files[0])
            self.current_image_path = first_image_path
            self.current_image_name = image_files[0]
            self.current_pixmap = QPixmap(first_image_path)
            self.update_preview_image()
            
            # Select the first item in the list
            first_index = self.lst_session_img.model().index(0, 0)
            self.lst_session_img.setCurrentIndex(first_index)
        
        # Start preprocessing the images
        # self.start_preprocessing()
        
        logger.info(f"Added {len(image_files)} images from folder {folder_path} to session {self.session_ctrl.current_session.name}")
        
        # Precompute embeddings asynchronously in a simple background thread
        def _precompute_embeddings(paths, session_name, handler):
            logger.info(f"Starting embedding precompute thread for {len(paths)} images in session {session_name}")
            logger.info(f"Handler session bound: {hasattr(handler, 'session') and handler.session is not None}")
            if hasattr(handler, 'session') and handler.session:
                logger.info(f"Handler session name: {handler.session.name}")
            total = len(paths)
            current = 0
            for img_name in paths:
                fp = os.path.join(folder_path, img_name)
                logger.info(f"Processing image {img_name} ({current+1}/{total})")
                try:
                    ok = handler.embed_image_with_id(fp, session_name)
                    logger.info(f"Embedding result for {img_name}: {ok}")
                    current += 1
                    # Update progress bar in UI thread
                    try:
                        # Use a simpler approach - just log progress for now
                        logger.info(f"Progress: {current}/{total} images processed")
                    except Exception as e2:
                        logger.debug(f"Progress invoke failed: {e2}")
                    # Mark image as preprocessed (coloring list item) via signal
                    try:
                        QMetaObject.invokeMethod(
                            self.session_ctrl,
                            'image_preprocessed_safe',
                            Qt.QueuedConnection,
                            Q_ARG(str, img_name)
                        )
                    except Exception as e3:
                        logger.debug(f"Image preprocessed signal failed: {e3}")
                except Exception as e:
                    logger.warning(f"Embedding failed for {fp}: {e}")
                    logger.exception(f"Full exception for {fp}:")
            logger.info(f"Embedding precompute thread completed. Processed {current}/{total} images.")
        
        # Create a simple wrapper that doesn't access UI elements
        def _precompute_wrapper():
            try:
                _precompute_embeddings(image_files, self.session_ctrl.current_session.name, self.annotation_ctrl.image_handler)
            except Exception as e:
                logger.exception(f"Error in embedding wrapper: {e}")
        
        try:
            session_name = self.session_ctrl.current_session.name
            # Ensure session is bound to image handler for identifier-based API
            self.annotation_ctrl.rebind_session()
            logger.info(f"Session binding status:")
            logger.info(f"  - Current session: {self.session_ctrl.current_session.name}")
            logger.info(f"  - Image handler session: {getattr(self.annotation_ctrl.image_handler, 'session', None)}")
            if hasattr(self.annotation_ctrl.image_handler, 'session') and self.annotation_ctrl.image_handler.session:
                logger.info(f"  - Bound session name: {self.annotation_ctrl.image_handler.session.name}")
            logger.info(f"Starting embedding thread for session: {session_name}")
            # Initialize progress UI
            try:
                self.progress_preprocessing.setMaximum(len(image_files))
                self.progress_preprocessing.setValue(0)
                self.lbl_preprocessing_status.setText(f"Preprocessing images... (0/{len(image_files)})")
            except Exception:
                pass
            t = threading.Thread(target=_precompute_wrapper, daemon=True)
            logger.info(f"Created embedding thread: {t}")
            t.start()
            logger.info(f"Started embedding thread: {t.is_alive()}")
            # Wait a moment and check if thread is still alive
            import time
            time.sleep(0.1)
            logger.info(f"Thread alive after 0.1s: {t.is_alive()}")
            logger.info(f"Thread name: {t.name}")
            logger.info(f"Thread daemon: {t.daemon}")
        except Exception as e:
            logger.warning(f"Could not start embedding precompute thread: {e}")
            logger.exception("Full exception when starting embedding thread:")
        
        # Also try a simple synchronous test first (segmentation runs in-process
        # now -- there's no server to connection-test anymore)
        logger.info("=== Testing embedding synchronously first ===")
        try:
            test_image = os.path.join(folder_path, image_files[0])
            logger.info(f"Testing with first image: {test_image}")
            result = self.annotation_ctrl.image_handler.embed_image_with_id(test_image, session_name)
            logger.info(f"Sync test result: {result}")
        except Exception as e:
            logger.exception("Sync test failed:")

    def start_preprocessing(self):
        """Start preprocessing images in the background"""
        # Disabled: Do nothing for now
        logger.info("Preprocessing thread is disabled. No background preprocessing will be started.")
        return

    def stop_preprocessing(self):
        """Stop the preprocessing process"""
        if self.preprocessing_worker and self.preprocessing_worker.isRunning():
            logger.info("Stopping preprocessing thread")
            self.preprocessing_worker.stop()
            
            # Wait for thread to finish with a timeout
            if not self.preprocessing_worker.wait(3000):  # 3 second timeout
                logger.warning("Preprocessing thread did not exit cleanly, terminating")
                self.preprocessing_worker.terminate()
                
            self.lbl_preprocessing_status.setText("Preprocessing stopped.")
            logger.info("Preprocessing thread stopped")

    def update_preprocessing_progress(self, current, total):
        """Update the preprocessing progress bar"""
        self.progress_preprocessing.setValue(current)
        self.lbl_preprocessing_status.setText(f"Preprocessing images... ({current}/{total})")
        
    def mark_image_preprocessed(self, image_name):
        """Mark an image as preprocessed in the UI"""
        logger.info(f"Marking image as preprocessed: {image_name}")
        
        # Save the session - this needs to be done in the main thread
        if self.session_ctrl and self.session_ctrl.current_session:
            try:
                self.session_ctrl.save_current_session()
                logger.info(f"Saved session after preprocessing {image_name}")
            except Exception as e:
                logger.error(f"Error saving session after preprocessing {image_name}: {e}")
        
        # Find the item in the list
        model = self.lst_session_img.model()
        if not model:
            logger.error("No model found for lst_session_img when trying to mark as preprocessed")
            return
            
        # Find the item with this image name
        found = False
        
        for row in range(model.rowCount()):
            item = model.item(row)
            if item.text() == image_name:
                # Set the background color to darker green
                item.setData(QColor(150, 220, 150), Qt.BackgroundRole)
                item.setData(QColor(0, 0, 0), Qt.ForegroundRole)  # Black text
                item.setToolTip("Preprocessing complete - Ready to annotate")
                found = True
                logger.info(f"Found and updated item in the list: {image_name}")
                break
        
        if not found:
            logger.warning(f"Could not find image in list: {image_name}")
        
        # Force the list to refresh - don't use update() directly
        # This will trigger the dataChanged signal and redraw the list view
        self.update_image_list()
        
        # Update button and status for current image if needed
        if self.current_image_name == image_name:
            self.update_current_image_status()

    def on_preprocessing_complete(self):
        """Called when preprocessing is complete"""
        logger.info("Preprocessing completed")
        self.lbl_preprocessing_status.setText("Preprocessing complete")
        self.lbl_preprocessing_status.setStyleSheet("color: green;")
        
        # Update the button and status for the currently selected image
        self.update_current_image_status()
        
        # Save the session
        self.save_session()
    
    def update_current_image_status(self):
        """Update the button and status for the currently selected image"""
        if not self.session_ctrl.current_session or not self.current_image_name:
            return
            
        # Check if the current image has been preprocessed
        if self.current_image_name in self.session_ctrl.current_session.preprocessed_images:
            # Enable annotation button
            self.btn_start_annotation.setText("Start/Continue Annotation")
            self.btn_start_annotation.setEnabled(True)
            self.lbl_preprocessing_status.setText("Preprocessing complete")
            self.lbl_preprocessing_status.setStyleSheet("color: green;")
            logger.info(f"Updated button state for current image: {self.current_image_name}")
            
            # Update the statistics
            self.update_statistics_labels()

    def mark_image_preprocessing_failed(self, error_msg):
        """Mark an image as failed preprocessing in the UI"""
        if not ":" in error_msg:
            logger.error(f"Invalid error message format: {error_msg}")
            return
            
        # Extract the image name from the error message
        parts = error_msg.split(":")
        image_name = parts[0].replace("Error processing image ", "").strip()
        
        # Find the item in the list
        model = self.lst_session_img.model()
        if not model:
            return
            
        # Find the item with this image name
        for row in range(model.rowCount()):
            item = model.item(row)
            if item.text() == image_name:
                # Set the background color to darker red
                item.setData(QColor(220, 150, 150), Qt.BackgroundRole)
                item.setData(QColor(0, 0, 0), Qt.ForegroundRole)  # Black text
                item.setToolTip(f"Preprocessing failed: {parts[1].strip()}")
                break
        
        # Update UI if this is the selected image
        current_index = self.lst_session_img.currentIndex()
        if current_index.isValid():
            current_name = model.data(current_index)
            if current_name == image_name:
                self.btn_start_annotation.setText("Preprocessing Failed")
                self.btn_start_annotation.setEnabled(False)
                self.lbl_preprocessing_status.setText(f"❌ Preprocessing failed: {parts[1].strip()}")
                self.lbl_preprocessing_status.setStyleSheet("color: red;")
                
        # Show error message
        QMessageBox.warning(
            self,
            "Preprocessing Failed",
            f"Failed to preprocess image '{image_name}': {parts[1].strip()}"
        )

    def save_session(self):
        """Save the current session"""
        if not self.session_ctrl.current_session:
            return False
            
        try:
            # Get the file path for the session
            session_file = os.path.join(self.session_ctrl.session_manager.sessions_dir, f"{self.session_ctrl.current_session.name}.json")
            self.session_ctrl.save_session(session_file)
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save session: {str(e)}")
            return False

    def on_session_changed(self):
        """Handle session changed signal"""
        self.update_image_list()
        if self.session_ctrl.current_session:
            self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
            
            # Stop any existing preprocessing thread
            if self.preprocessing_worker and self.preprocessing_worker.isRunning():
                logger.info("Stopping existing preprocessing thread due to session change")
                self.stop_preprocessing()
            
            # Start/continue preprocessing in the background
            # self.start_preprocessing()
            
        self.update_statistics_labels()
        self.update_label_buttons()  # Update label button texts based on task type

    def on_session_saved(self):
        """Handle session saved signal"""
        self.session_ctrl.has_unsaved_changes = False
        logger.info("Session saved successfully")

    def on_session_loaded(self):
        """Handle session loaded signal"""
        # Update UI elements that depend on session state
        self.update_image_list()
        if self.session_ctrl.current_session:
            self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
        self.update_statistics_labels()
        self.update_label_buttons()  # Update label button texts based on task type
        # Restore this session's previously-trained classifier, if any -
        # otherwise it silently started untrained again on every reopen.
        self.annotation_ctrl.reload_classifier()

    def _handle_label_button_click(self, label: str):
        """Handle label button clicks with logging and task-type awareness"""
        logger.info(f"Label button clicked: {label}")
        session = self.session_ctrl.current_session
        # Map label according to task type
        if session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY:
            if label == "intact":
                mapped_label = "Z-shape"
            elif label == "damaged":
                mapped_label = "S-shape"
            elif label == "invalid":
                mapped_label = "Invalid"
            else:
                mapped_label = label
        else:
            if label == "intact":
                mapped_label = "intact"
            elif label == "damaged":
                mapped_label = "damaged"
            elif label == "invalid":
                mapped_label = "invalid"
            else:
                mapped_label = label
        # Check if we have selected annotations
        if not self.annotation_ctrl.selected_indices:
            logger.warning("No annotations selected for label change")
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select one or more annotations to change their label."
            )
            return
        # Change the label for all selected annotations
        success = self.annotation_ctrl.change_selected_annotations_label(mapped_label)
        if success:
            logger.info(f"Successfully changed label to {mapped_label} for {len(self.annotation_ctrl.selected_indices)} annotations")
            # Refresh the display to show the updated labels
            self.refresh_annotation_display()
            # Update the annotations list
            self.annotation_ctrl.update_annotations_list(self.lst_dnao)
            # Update statistics
            self.update_statistics_labels()
            # Update session stats display if we're in session view
            if self.currentIndex() == 1 and self.session_ctrl.current_session:
                self.session_ctrl.display_session_stats(self.session_ctrl.current_session, self.lst_session_stats)
        else:
            logger.error(f"Failed to change label to {mapped_label}")
            QMessageBox.warning(
                self,
                "Label Change Failed",
                f"Failed to change label to {mapped_label}."
            )

    def _sync_list_selection_with_image(self):
        """Synchronize the list view selection with the image selection"""
        if not self.annotation_ctrl.selected_indices:
            # If no annotations are selected, clear the list selection
            self.lst_dnao.clearSelection()
            return
            
        # Get the model
        model = self.lst_dnao.model()
        if not model:
            return
            
        # Select the first selected annotation in the list
        # (we can only select one item at a time in the list view)
        first_selected_idx = self.annotation_ctrl.selected_indices[0]
        if first_selected_idx < model.rowCount():
            self.lst_dnao.setCurrentIndex(model.index(first_selected_idx, 0))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_annotation_display()

    def update_label_buttons(self):
        """Update the label button texts based on the session's task type."""
        session = self.session_ctrl.current_session
        if session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY:
            self.btn_set_intact.setText("Z-shape (1)")
            self.btn_set_defect.setText("S-shape (2)")
            self.btn_set_invalid.setText("Invalid (3)")
        else:
            self.btn_set_intact.setText("Intact (1)")
            self.btn_set_defect.setText("Defect (2)")
            self.btn_set_invalid.setText("Invalid (3)")

    def update_session_details(self):
        """Update the session details in the main menu to reflect the current session and task type."""
        session = self.session_ctrl.current_session
        if not session:
            return
        model = QStandardItemModel()
        model.appendRow(QStandardItem(f"Session: {session.name}"))
        model.appendRow(QStandardItem(f"Created: {session.created[:10]}"))
        model.appendRow(QStandardItem(f"Images: {len(session.image_files)}"))
        model.appendRow(QStandardItem(f"Total DNAO: {session.get_annotation_count()}"))
        # Task type user-friendly
        if hasattr(session, 'task_type'):
            if session.task_type == TaskType.CHIRALITY:
                task_str = "Chirality (Z-shape/S-shape)"
            elif session.task_type == TaskType.YIELD:
                task_str = "Yield/Quality (Intact/Defect)"
            else:
                task_str = str(session.task_type)
            model.appendRow(QStandardItem(f"Task: {task_str}"))
        self.lst_session_stats.setModel(model)