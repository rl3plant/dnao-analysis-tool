import logging
import os
import sys
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import cv2
from PySide6.QtWidgets import QMessageBox, QDialog
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor, QPainter, QPen, QBrush, QPixmap
from PySide6.QtCore import Qt, QPoint
import matplotlib.pyplot as plt  # Add at the top with other imports

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handling.data_models import Annotation, TaskType
from network.image_handler import ImageHandler
from .image_utils import convert_to_image_coordinates, is_point_in_any_annotation, find_closest_annotation
from .segmentation_selection_dialog import SegmentationSelectionDialog
from src.client.pipeline.instance_segmentation.segmentation_classification.svm_classifier import SVMClassifier

logger = logging.getLogger(__name__)

class AnnotationController:
    """Handles annotation operations"""
    
    def __init__(self, parent_window):
        self.parent = parent_window
        self.image_handler = ImageHandler()
        self.annotations: List[Annotation] = []
        self.has_unsaved_changes: bool = False
        self.show_annotations: bool = True  # Add flag for visualization toggle
        self.selected_indices: List[int] = []  # Track selected annotation indices
        self.selection_rect: Optional[tuple] = None  # Track selection rectangle (start_x, start_y, end_x, end_y)
        self.is_selecting: bool = False  # Track if we're currently selecting
        # Bind session for identifier-based API
        if hasattr(self.parent, 'session_ctrl') and self.parent.session_ctrl.current_session:
            self.image_handler.session = self.parent.session_ctrl.current_session
    
    def rebind_session(self):
        """Rebind the current session to the image handler for identifier-based API calls."""
        if hasattr(self.parent, 'session_ctrl') and self.parent.session_ctrl.current_session:
            self.image_handler.session = self.parent.session_ctrl.current_session
            logger.info(f"Rebound session {self.parent.session_ctrl.current_session.name} to image handler")
        else:
            logger.warning("No current session available for rebinding")
    
    def reload_classifier(self):
        """
        Reload the classifier from the current session's classifier path.
        """
        session = self.parent.session_ctrl.current_session
        if not session:
            logger.warning("No current session available for classifier reload")
            return False
        classifier_path = session.get_classifier_path()
        if not hasattr(session, '_classifier_instance'):
            session._classifier_instance = SVMClassifier()
        classifier = session._classifier_instance
        try:
            classifier.load(classifier_path)
            logger.info("Classifier reloaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not reload classifier: {e}")
            return False
    
    def generate_masks(self, image_path):
        """Generate masks using the server and update annotations"""
        if not hasattr(self, 'image_handler'):
            self.image_handler = ImageHandler()
        
        # Ensure the image handler has the current session bound
        if hasattr(self.parent, 'session_ctrl') and self.parent.session_ctrl.current_session:
            self.image_handler.session = self.parent.session_ctrl.current_session
            logger.info(f"Bound session {self.parent.session_ctrl.current_session.name} to image handler for mask generation")
        else:
            logger.warning("No current session available for mask generation")
        
        # Send image to server for processing
        response = self.image_handler.generate_masks(image_path)
        
        if response:
            # Process the response into annotations
            self.annotations = self.image_handler.process_masks(response, image_path)
            
            # Sort annotations after loading
            self._sort_annotations()
            
            # Mark that we have unsaved changes
            self.has_unsaved_changes = True
            
            logger.info(f"Generated {len(self.annotations)} annotations")
            return True
        else:
            logger.error("Failed to generate masks from server")
            return False
    
    def display_annotations(self, pixmap, label, selected_index: int = -1):
        """Display the current annotations on the image"""
        if not pixmap:
            logger.warning("No pixmap available for annotation display")
            return None
            
        # Create a new pixmap from the original image for drawing
        try:
            # Create a copy of the pixmap for drawing
            result_pixmap = pixmap.copy()
            painter = QPainter(result_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw annotations if they exist and are visible
            if self.annotations and self.show_annotations:
                logger.info(f"Displaying {len(self.annotations)} annotations")
                
                # Draw annotations
                for i, annotation in enumerate(self.annotations):
                    # Convert contour points to QPoint list
                    points = [QPoint(int(p[0]), int(p[1])) for p in annotation.contour]
                    
                    # Get session task type to determine color scheme
                    session = self.parent.session_ctrl.current_session if hasattr(self.parent, 'session_ctrl') else None
                    is_chirality_task = session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY
                    
                    # Debug: Log the annotation being processed
                    logger.debug(f"Displaying annotation {i}: class_label='{annotation.class_label}', is_chirality_task={is_chirality_task}")
                    if session:
                        logger.debug(f"Session task_type: {session.task_type}, type: {type(session.task_type)}")
                    
                    # Set color based on class label, selection state, and task type
                    if i in self.selected_indices:
                        # Selected annotations are highlighted with a thicker border
                        if is_chirality_task:
                            # Chirality colors
                            if annotation.class_label == "Z-shape":
                                pen_color = QColor(255, 165, 0)  # Orange
                                fill_color = QColor(255, 165, 0, 50)  # Transparent orange
                            elif annotation.class_label == "S-shape":
                                pen_color = QColor(0, 0, 255)  # Blue
                                fill_color = QColor(0, 0, 255, 50)  # Transparent blue
                            else:
                                pen_color = QColor(200, 200, 200)  # Bright gray
                                fill_color = QColor(200, 200, 200, 50)  # Transparent gray
                        else:
                            # Yield colors
                            if annotation.class_label == "intact":
                                pen_color = QColor(0, 255, 0)  # Bright green
                                fill_color = QColor(0, 255, 0, 50)  # Transparent green
                            elif annotation.class_label == "damaged":
                                pen_color = QColor(255, 0, 0)  # Bright red
                                fill_color = QColor(255, 0, 0, 50)  # Transparent red
                            else:  # invalid
                                pen_color = QColor(200, 200, 200)  # Bright gray
                                fill_color = QColor(200, 200, 200, 50)  # Transparent gray
                    else:
                        # Normal annotations
                        if is_chirality_task:
                            # Chirality colors
                            if annotation.class_label == "Z-shape":
                                pen_color = QColor(255, 140, 0)  # Dark orange
                                fill_color = QColor(255, 140, 0, 30)  # Transparent orange
                            elif annotation.class_label == "S-shape":
                                pen_color = QColor(0, 0, 200)  # Dark blue
                                fill_color = QColor(0, 0, 200, 30)  # Transparent blue
                            else:
                                pen_color = QColor(64, 64, 64)  # Dark gray
                                fill_color = QColor(64, 64, 64, 30)  # Transparent gray
                        else:
                            # Yield colors
                            if annotation.class_label == "intact":
                                pen_color = QColor(0, 128, 0)  # Dark green
                                fill_color = QColor(0, 128, 0, 30)  # Transparent green
                            elif annotation.class_label == "damaged":
                                pen_color = QColor(128, 0, 0)  # Dark red
                                fill_color = QColor(128, 0, 0, 30)  # Transparent red
                            else:  # invalid
                                pen_color = QColor(64, 64, 64)  # Dark gray
                                fill_color = QColor(64, 64, 64, 30)  # Transparent gray
                    
                    # Set up the painter for this annotation
                    painter.setPen(QPen(pen_color, 3 if i in self.selected_indices else 2))
                    painter.setBrush(QBrush(fill_color))
                    
                    # Draw the filled polygon
                    if len(points) >= 3:
                        painter.drawPolygon(points)
            
            # Draw selection rectangle if active
            if self.selection_rect:
                logger.info(f"Drawing selection rectangle: {self.selection_rect}")
                painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))  # Yellow dashed line
                
                # Get the rectangle coordinates
                x1, y1, x2, y2 = self.selection_rect
                
                # Normalize the rectangle coordinates (ensure x1 < x2, y1 < y2)
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Calculate the width and height
                rect_width = x2 - x1
                rect_height = y2 - y1
                
                # Draw rectangle in image coordinates
                painter.drawRect(x1, y1, rect_width, rect_height)
                logger.info(f"Drew rectangle at ({x1}, {y1}) with size {rect_width}x{rect_height}")
            
            painter.end()
            return result_pixmap
            
        except Exception as e:
            logger.exception(f"Error displaying annotations: {str(e)}")
            return None
    
    def update_annotations_list(self, list_widget):
        """Update the list of annotations"""
        # Create list model
        model = QStandardItemModel()
        
        # Store current selection
        current_index = list_widget.currentIndex()
        current_row = current_index.row() if current_index.isValid() else -1
        
        # Store the current background color if it exists
        current_bg_color = None
        if hasattr(list_widget, 'palette'):
            current_bg_color = list_widget.palette().color(list_widget.backgroundRole())
        
        for i, annotation in enumerate(self.annotations):
            item = QStandardItem(f"DNAO {i+1}: {annotation.class_label}")
            
            # Get session task type to determine color scheme
            session = self.parent.session_ctrl.current_session if hasattr(self.parent, 'session_ctrl') else None
            is_chirality_task = session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY
            
            # Debug logging
            if session:
                logger.debug(f"Annotation {i}: class_label='{annotation.class_label}', session.task_type='{session.task_type}', is_chirality_task={is_chirality_task}")
            else:
                logger.debug(f"Annotation {i}: class_label='{annotation.class_label}', no session")
            
            # Set color based on class label and task type
            if is_chirality_task:
                # Chirality colors
                if annotation.class_label == "Z-shape":
                    item.setForeground(QColor(255, 140, 0))  # Orange
                elif annotation.class_label == "S-shape":
                    item.setForeground(QColor(0, 0, 200))  # Blue
                else:
                    item.setForeground(QColor(64, 64, 64))  # Gray
            else:
                # Yield colors
                if annotation.class_label == "intact":
                    item.setForeground(QColor(0, 128, 0))  # Green
                elif annotation.class_label == "damaged":
                    item.setForeground(QColor(255, 0, 0))  # Red
                elif annotation.class_label == "invalid":
                    item.setForeground(QColor(64, 64, 64))  # Gray
                
            model.appendRow(item)
            
        list_widget.setModel(model)

        # Restore selection if it was valid
        if current_row >= 0 and current_row < model.rowCount():
            list_widget.setCurrentIndex(model.index(current_row, 0))
        # If no selection or invalid selection, select the first item if available
        elif model.rowCount() > 0:
            list_widget.setCurrentIndex(model.index(0, 0))
            
        # Restore the background color if it was previously set
        if current_bg_color:
            palette = list_widget.palette()
            palette.setColor(list_widget.backgroundRole(), current_bg_color)
            list_widget.setPalette(palette)
            
        # Synchronize with selected_indices
        if self.selected_indices:
            # Select the first selected annotation in the list
            first_selected_idx = self.selected_indices[0]
            if first_selected_idx < model.rowCount():
                list_widget.setCurrentIndex(model.index(first_selected_idx, 0))
        
        # If the parent has a _connect_list_selection_changed method, call it
        if hasattr(self.parent, '_connect_list_selection_changed'):
            self.parent._connect_list_selection_changed()
    
    def _calculate_center(self, contour: np.ndarray) -> tuple:
        """Calculate the center coordinates of a contour"""
        if contour is None or len(contour) == 0:
            return (0, 0)
        return tuple(np.mean(contour, axis=0))

    def _sort_annotations(self):
        """Sort annotations by their center coordinates from top-left to bottom-right"""
        def sort_key(annotation):
            # Use y-coordinate as primary key (top to bottom)
            # Use x-coordinate as secondary key (left to right)
            return (annotation.center_of_mass[1], annotation.center_of_mass[0])
        
        self.annotations.sort(key=sort_key)

    def add_annotation_at_position(self, position, image_path, label, pixmap):
        """Add a new DNAO annotation at the clicked position or within the selection rectangle"""
        if not image_path:
            logger.warning("Cannot add DNAO: No image path")
            return False
        
        try:
            # Check if we have an active selection rectangle
            if self.selection_rect is not None:
                logger.info(f"Selection rectangle is active: {self.selection_rect}")
                # Get selection rectangle coordinates
                x1, y1, x2, y2 = self.selection_rect
                
                # Get image dimensions from parent's current_pixmap
                if hasattr(self.parent, 'current_pixmap') and self.parent.current_pixmap:
                    image_width = self.parent.current_pixmap.width()
                    image_height = self.parent.current_pixmap.height()
                    
                    logger.info(f"Image dimensions: {image_width}x{image_height}")
                    
                    # Make sure x1 < x2 and y1 < y2
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    logger.info(f"Normalized coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                        
                    # Clamp coordinates to image boundaries
                    x1 = max(0, min(x1, image_width))
                    y1 = max(0, min(y1, image_height))
                    x2 = max(0, min(x2, image_width))
                    y2 = max(0, min(y2, image_height))
                    
                    logger.info(f"Clamped coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                    
                    # Create bounding box dictionary
                    bbox = {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                    
                    # Only proceed if the bounding box has a reasonable size
                    if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
                        logger.warning(f"Bounding box too small: {bbox}, dimensions: {abs(x2-x1)}x{abs(y2-y1)}")
                        # Fallback to point-based segmentation
                        image_position = convert_to_image_coordinates(position, label, pixmap)
                        logger.info(f"Falling back to point-based segmentation at {image_position}")
                        result = self.image_handler.request_segmentation(
                            image_path,
                            int(image_position.x()),
                            int(image_position.y())
                        )
                    else:
                        logger.info(f"Using bounding box for segmentation: {bbox}, dimensions: {abs(x2-x1)}x{abs(y2-y1)}")
                        # Call the segment_bbox endpoint with the bounding box
                        result = self.image_handler.request_bbox_segmentation(
                            image_path,
                            bbox
                        )
                else:
                    logger.warning("No current pixmap available, falling back to point-based segmentation")
                    # Fall back to point-based segmentation
                    image_position = convert_to_image_coordinates(position, label, pixmap)
                    result = self.image_handler.request_segmentation(
                        image_path,
                        int(image_position.x()),
                        int(image_position.y())
                    )
            else:
                # Convert position from widget coordinates to image coordinates
                image_position = convert_to_image_coordinates(position, label, pixmap)
                
                # Check if there's already a DNAO at or near this position
                is_inside, _ = is_point_in_any_annotation(self.annotations, image_position)
                if is_inside:
                    logger.info(f"Point is inside an existing annotation contour")
                    QMessageBox.information(
                        self.parent,
                        "DNAO Already Exists",
                        "This point is inside an existing DNAO annotation."
                    )
                    return False
                
                logger.info(f"Adding DNAO at position: {image_position}")
                
                # Call the segment endpoint with the point coordinates
                result = self.image_handler.request_segmentation(
                    image_path,
                    int(image_position.x()),
                    int(image_position.y())
                )
            
            if result:
                logger.info(f"Received segmentation result: {result.keys() if isinstance(result, dict) else 'not a dict'}")
                
                # Check if we have multiple predictions
                predictions = result.get("predictions", [])[0]['masks']
                if isinstance(predictions, list) and len(predictions) > 1:
                    image_position = convert_to_image_coordinates(position, label, pixmap)
                    # Show the selection dialog
                    dialog = SegmentationSelectionDialog(self.parent, image_path, predictions, image_position)
                    if dialog.exec_() == QDialog.Accepted:
                        # Get the selected prediction
                        selected_prediction = dialog.get_selected_prediction()
                        if selected_prediction:
                            # Create a new result dict with just the selected prediction
                            result = {"prediction": [selected_prediction]}
                        else:
                            logger.warning("No prediction selected")
                            return False
                
                # Process the segmentation result and add to annotations
                new_annotation = self._process_segmentation_result(result, image_path)
                
                if new_annotation:
                    logger.info(f"Created new annotation with contour of {len(new_annotation.contour) if new_annotation.contour is not None else 'None'} points")
                    
                    # Get session task type to determine default label
                    session = self.parent.session_ctrl.current_session if hasattr(self.parent, 'session_ctrl') else None
                    is_chirality_task = session and hasattr(session, 'task_type') and session.task_type == TaskType.CHIRALITY
                    
                    # Downstream chirality classification using KMeans (only for chirality tasks)
                    if is_chirality_task:
                        if session and hasattr(session, '_kmeans_model') and session._kmeans_model is not None:
                            from src.client.pipeline.downstream_tasks.chirality_classifier import assign_chirality_labels_kmeans
                            assign_chirality_labels_kmeans(session._kmeans_model, [new_annotation])
                            logger.info(f"Assigned chirality label '{new_annotation.class_label}' to new annotation")
                        else:
                            logger.info("No KMeans model available for chirality classification")
                            # For chirality tasks, assign a default label if no model is available
                            new_annotation.class_label = "Z-shape"  # Default to Z-shape
                            logger.info(f"Assigned default chirality label '{new_annotation.class_label}' to new annotation")
                    else:
                        # For yield tasks, assign default yield label
                        new_annotation.class_label = "intact"  # Default to intact for yield tasks
                        logger.info(f"Assigned default yield label '{new_annotation.class_label}' to new annotation")
                    
                    # Add to annotations list and sort
                    self.annotations.append(new_annotation)
                    self._sort_annotations()
                    
                    # Mark that we have unsaved changes
                    self.has_unsaved_changes = True
                    
                    # Update the list view without resetting its appearance
                    if hasattr(self.parent, 'lst_dnao'):
                        self.update_annotations_list(self.parent.lst_dnao)
                    
                    # Refresh the display to show the new annotation with correct colors
                    if hasattr(self.parent, 'refresh_annotation_display'):
                        self.parent.refresh_annotation_display()
                    
                    # Clear the selection rectangle after adding the annotation
                    logger.info("Clearing selection rectangle after adding annotation")
                    self.selection_rect = None
                    
                    return True
                else:
                    logger.error("Failed to create annotation from segmentation result")
            else:
                logger.error("Failed to get a valid segmentation result")
                
            return False
                
        except Exception as e:
            logger.exception(f"Error adding DNAO: {str(e)}")
            return False
    
    def remove_annotation_at_position(self, position, label, pixmap):
        """Remove DNAO annotation at or near the clicked position"""
        if not self.annotations:
            logger.warning("Cannot remove DNAO: No annotations")
            return False
        
        try:
            # Convert position from widget coordinates to image coordinates
            image_position = convert_to_image_coordinates(position, label, pixmap)
            
            logger.info(f"Attempting to remove DNAO at position: {image_position}")
            
            # Check if point is inside any annotation's contour
            is_inside, idx = is_point_in_any_annotation(self.annotations, image_position)
            if is_inside and idx is not None:
                # Remove the annotation
                self.annotations.pop(idx)
                logger.info(f"Removed annotation at index {idx}")
                
                # Mark that we have unsaved changes
                self.has_unsaved_changes = True
                return True
                
            # If no annotation contains the point, fall back to closest annotation approach
            closest_idx, min_distance = find_closest_annotation(self.annotations, image_position)
            
            if closest_idx is not None and min_distance < 10:  # Use reasonable threshold
                # Remove the annotation
                self.annotations.pop(closest_idx)
                logger.info(f"Removed closest annotation at index {closest_idx} (distance: {min_distance:.2f})")
                
                # Mark that we have unsaved changes
                self.has_unsaved_changes = True
                return True
            
            return False
                
        except Exception as e:
            logger.exception(f"Error removing DNAO: {str(e)}")
            return False
    
    def _process_segmentation_result(self, result, image_path):
        """Process segmentation result into an Annotation object"""
        if not result:
            logger.warning("No segmentation result received")
            return None
        
        # The server returns a response with 'point' or 'bbox' and 'prediction' keys
        predictions = result.get("prediction", [])
        if not predictions:
            logger.warning("No prediction in segmentation result")
            return None
        
        # Determine segmentation type
        segmentation_type = "point"
        if "bbox" in result:
            segmentation_type = "bbox"
        elif "contour" in result:
            segmentation_type = "contour"
        
        logger.info(f"Processing segmentation result of type: {segmentation_type}")
        
        # Use the first prediction with highest score if multiple are returned
        if isinstance(predictions, list) and len(predictions) > 0:
            # Sort by score if available, otherwise take the first one
            prediction = sorted(predictions, key=lambda x: x.get("score", 0), reverse=True)[0]
        else:
            prediction = predictions
        
        # Create contour from prediction
        contours_data = prediction.get("contours", [])
        if not contours_data or len(contours_data) == 0:
            logger.warning("No contours in prediction")
            return None
            
        try:
            # Check if contours_data has multiple contours and get the first one
            # This removes the batch dimension if it exists
            if isinstance(contours_data[0], list) and len(contours_data) > 0:
                contour_points = contours_data[0]  # Get the first contour
            else:
                contour_points = contours_data  # Use as is if no batch dimension
            
            # Convert to numpy array
            contour_np = np.array(contour_points, dtype=np.int32)
            
            # Create properties dictionary with segmentation type
            properties = {
                "area": prediction.get("area", 0),
                "from_segmentation": True,
                "segmentation_type": segmentation_type
            }
            
            # Add point or bbox data to properties
            if segmentation_type == "point" and "point" in result:
                properties["prompt_point"] = result["point"]
            elif segmentation_type == "bbox" and "bbox" in result:
                properties["prompt_bbox"] = result["bbox"]
            elif segmentation_type == "contour" and "contour" in result:
                properties["prompt_contour"] = "used_contour"
                
            # Create new annotation
            annotation = Annotation(
                id=f"seg_{segmentation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_image=image_path,
                contour=contour_np,  # Pass the processed contour array
                shape_tag="auto",
                class_label="intact",  # Default to intact
                confidence=prediction.get("score", 1.0),
                properties=properties
            )
            
            return annotation
            
        except Exception as e:
            logger.exception(f"Error creating annotation from prediction: {str(e)}")
            return None

    def select_previous_annotation(self, list_widget):
        """Select the previous annotation in the list"""
        current_index = list_widget.currentIndex()
        if not current_index.isValid():
            # If no selection, select the last item
            last_row = list_widget.model().rowCount() - 1
            if last_row >= 0:
                list_widget.setCurrentIndex(list_widget.model().index(last_row, 0))
                # Update selected_indices
                self.selected_indices = [last_row]
        else:
            # Select previous item
            prev_row = current_index.row() - 1
            if prev_row >= 0:
                list_widget.setCurrentIndex(list_widget.model().index(prev_row, 0))
                # Update selected_indices
                self.selected_indices = [prev_row]
            
        # Notify the parent window to refresh the annotation display
        if hasattr(self.parent, 'refresh_annotation_display'):
            self.parent.refresh_annotation_display()

    def select_next_annotation(self, list_widget):
        """Select the next annotation in the list"""
        current_index = list_widget.currentIndex()
        if not current_index.isValid():
            # If no selection, select the first item
            if list_widget.model().rowCount() > 0:
                list_widget.setCurrentIndex(list_widget.model().index(0, 0))
                # Update selected_indices
                self.selected_indices = [0]
        else:
            # Select next item
            next_row = current_index.row() + 1
            if next_row < list_widget.model().rowCount():
                list_widget.setCurrentIndex(list_widget.model().index(next_row, 0))
                # Update selected_indices
                self.selected_indices = [next_row]
            
        # Notify the parent window to refresh the annotation display
        if hasattr(self.parent, 'refresh_annotation_display'):
            self.parent.refresh_annotation_display()

    def change_selected_annotation_label(self, list_widget, new_label):
        """Change the label of the selected annotation"""
        # Get the current selection
        current_index = list_widget.currentIndex()
        if not current_index.isValid():
            QMessageBox.warning(
                self.parent,
                "No Selection",
                "Please select an annotation to label."
            )
            return
            
        # Get the annotation index
        annotation_idx = current_index.row()
        if annotation_idx < 0 or annotation_idx >= len(self.annotations):
            logger.error(f"Invalid annotation index: {annotation_idx}")
            return
            
        # Update the annotation label
        self.annotations[annotation_idx].class_label = new_label
        
        # Mark that we have unsaved changes
        self.has_unsaved_changes = True
        
        # Update the list display
        self.update_annotations_list(list_widget)
        
        # Update the statistics labels
        self.parent.update_statistics_labels()
        
        # Refresh the annotation display
        self.parent.refresh_annotation_display()
        
        # Select the next annotation
        self.select_next_annotation(list_widget)

    def toggle_annotations(self):
        """Toggle the visibility of annotations"""
        self.show_annotations = not self.show_annotations
        return self.show_annotations

    def start_annotation(self, image_path):
        """Start annotation for a new image"""
        if not image_path:
            logger.warning("Cannot start annotation: No image path")
            return False
            
        try:
            # Generate masks from server
            success = self.generate_masks(image_path)
            
            if success:
                # Sort annotations after loading
                self._sort_annotations()
                
                # Update the display
                if hasattr(self.parent, 'refresh_annotation_display'):
                    self.parent.refresh_annotation_display()
                    
                # Update the annotations list
                if hasattr(self.parent, 'lst_dnao'):
                    self.update_annotations_list(self.parent.lst_dnao)
                    
                return True
                
            return False
            
        except Exception as e:
            logger.exception(f"Error starting annotation: {str(e)}")
            return False

    def process_masks_from_preprocessed(self, masks_data, image_path):
        """Process masks from preprocessed data
        
        Args:
            masks_data: Preprocessed masks data
            image_path: Path to the source image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure we have valid masks data
            if masks_data is None:
                logger.error("No masks data provided")
                return False
                
            # Handle different formats of preprocessed data
            if isinstance(masks_data, dict):
                # If masks_data is a dictionary, it might contain a 'masks' key
                if 'masks' in masks_data:
                    # Use the 'masks' list
                    masks = masks_data['masks']
                    logger.info(f"Using 'masks' from dict, found {len(masks)} masks")
                else:
                    # Or it might be a single mask
                    masks = [masks_data]
                    logger.info("Using entire dict as a single mask")
            elif isinstance(masks_data, list):
                # Already a list of masks
                masks = masks_data
                logger.info(f"Using list of masks, found {len(masks)} masks")
            else:
                # Unknown format
                logger.error(f"Unexpected masks data format: {type(masks_data)}")
                return False
                
            # Prepare server-like response format for the image handler
            server_response = {'masks': masks}
            
            # Process the masks into annotations
            self.annotations = self.image_handler.process_masks(server_response, image_path)
            
            # Sort annotations after loading
            self._sort_annotations()
            
            # Mark that we have unsaved changes
            self.has_unsaved_changes = True
            
            logger.info(f"Processed {len(self.annotations)} annotations from preprocessed data")
            return True
        except Exception as e:
            logger.exception(f"Error processing preprocessed masks: {e}")
            return False

    def start_selection(self, position: QPoint):
        """Start a selection rectangle"""
        if not hasattr(self.parent, 'current_pixmap') or not self.parent.current_pixmap:
            logger.warning("No current pixmap available for selection")
            return
        
        # Convert position to image coordinates
        image_pos = convert_to_image_coordinates(position, self.parent.lbl_afm_img, self.parent.current_pixmap)
        
        # Only start selection if we're not already selecting
        if not self.is_selecting:
            self.is_selecting = True
            self.selection_rect = (image_pos.x(), image_pos.y(), image_pos.x(), image_pos.y())
            logger.info(f"Started selection at image coordinates: ({image_pos.x()}, {image_pos.y()})")

    def update_selection(self, position: QPoint):
        """Update the selection rectangle"""
        if not self.is_selecting or not self.selection_rect:
            return
        
        if not hasattr(self.parent, 'current_pixmap') or not self.parent.current_pixmap:
            logger.warning("No current pixmap available for selection update")
            return
        
        # Convert position to image coordinates
        image_pos = convert_to_image_coordinates(position, self.parent.lbl_afm_img, self.parent.current_pixmap)
        
        # Update the selection rectangle with the new end point
        self.selection_rect = (self.selection_rect[0], self.selection_rect[1], image_pos.x(), image_pos.y())
        logger.info(f"Updated selection rectangle to: {self.selection_rect}")

    def end_selection(self):
        """End the selection process and process selected annotations"""
        if not self.is_selecting:
            return

        # Get selection rectangle coordinates
        if self.selection_rect is not None:
            # Get selection rectangle coordinates
            x1, y1, x2, y2 = self.selection_rect
            
            # Get image dimensions from parent's current_pixmap
            if hasattr(self.parent, 'current_pixmap') and self.parent.current_pixmap:
                # Make sure x1 < x2 and y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Ensure rectangle is within image boundaries
                image_width = self.parent.current_pixmap.width()
                image_height = self.parent.current_pixmap.height()
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                # Update the selection rectangle with the normalized coordinates
                self.selection_rect = (x1, y1, x2, y2)
                
                # Calculate width and height to determine if it's a valid selection
                width = x2 - x1
                height = y2 - y1
                
                # Only update selection if the rectangle has a meaningful size
                if width > 5 and height > 5:
                    # Update the selected indices based on the rectangle
                    # (this will select annotations within the rectangle)
                    self._update_selected_indices()
                    logger.info(f"End selection: rectangle size {width}x{height}, selected {len(self.selected_indices)} annotations")
                else:
                    logger.info(f"Selection rectangle too small ({width}x{height}), ignoring")
            else:
                logger.warning("No current pixmap available, cannot process selection rectangle")
        
        # Reset selection state but KEEP the rectangle
        self.is_selecting = False

    def _calculate_overlap_ratio(self, contour, rect):
        """Calculate the ratio of the annotation's area that overlaps with the selection rectangle."""
        if contour is None or len(contour) < 3:
            return 0.0
        # Get image dimensions from parent pixmap
        if not hasattr(self.parent, 'current_pixmap') or self.parent.current_pixmap is None:
            return 0.0
        img_width = self.parent.current_pixmap.width()
        img_height = self.parent.current_pixmap.height()
        x1, y1, x2, y2 = rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        # Create full image-sized mask for annotation
        contour = np.array(contour, dtype=np.int32)
        ann_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.drawContours(ann_mask, [contour], -1, 1, -1)
        # Create full image-sized mask for rectangle
        rect_mask = np.zeros_like(ann_mask)
        cv2.rectangle(rect_mask, (x1, y1), (x2, y2), 1, -1)
        # Calculate intersection
        intersection = np.logical_and(ann_mask, rect_mask)
        overlap_area = np.sum(intersection)
        ann_area = np.sum(ann_mask)
        if ann_area == 0:
            return 0.0
        ratio = overlap_area / ann_area
        return ratio

    def _update_selected_indices(self):
        """Update selected indices based on selection rectangle with >90% overlap criterion"""
        if not self.selection_rect:
            return
        logger.info("Updating selected indices from selection rectangle")
        x1, y1, x2, y2 = self.selection_rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        self.selected_indices = []
        for i, ann in enumerate(self.annotations):
            if ann.contour is None or len(ann.contour) == 0:
                continue
            overlap_ratio = self._calculate_overlap_ratio(ann.contour, (x1, y1, x2, y2))
            if overlap_ratio > 0.99:
                self.selected_indices.append(i)
                logger.info(f"Selected annotation {i} with >90% overlap (ratio={overlap_ratio:.2f})")
        logger.info(f"Selected {len(self.selected_indices)} annotations within rectangle (>90% overlap)")
        if hasattr(self.parent, 'refresh_annotation_display'):
            self.parent.refresh_annotation_display()

    def toggle_selection(self, index: int, ctrl_alt_pressed: bool = False):
        """Toggle selection of a single annotation"""
        if ctrl_alt_pressed:
            if index in self.selected_indices:
                self.selected_indices.remove(index)
            else:
                self.selected_indices.append(index)
        else:
            self.selected_indices = [index]
        self._update_selected_indices()

    def _annotation_in_rect(self, annotation, start_pos, end_pos):
        """Check if an annotation intersects with the selection rectangle"""
        if annotation.contour is None or len(annotation.contour) == 0:
            return False
            
        # Get the bounding box of the annotation
        x, y, w, h = cv2.boundingRect(annotation.contour)
        
        # Check if the bounding box intersects with the selection rectangle
        rect_left = min(start_pos.x(), end_pos.x())
        rect_right = max(start_pos.x(), end_pos.x())
        rect_top = min(start_pos.y(), end_pos.y())
        rect_bottom = max(start_pos.y(), end_pos.y())
        
        # Check if any part of the annotation is within the rectangle
        # This tests for rectangle intersection
        intersection = not (x + w < rect_left or x > rect_right or y + h < rect_top or y > rect_bottom)
        
        # For a more accurate test, we could check if any contour point is inside the rectangle
        # But the bounding box intersection is usually sufficient and faster
        return intersection

    def fuse_selected_annotations(self):
        """Fuse multiple selected annotations into one using convex hull and SAM2 refinement"""
        if len(self.selected_indices) < 2:
            return False

        try:
            # Get the selected annotations
            selected_annotations = [self.annotations[i] for i in self.selected_indices]
            
            # Validate contours
            valid_contours = []
            for ann in selected_annotations:
                if ann.contour is not None and len(ann.contour) > 0:
                    # Ensure contour is in the correct format (N,2)
                    contour = ann.contour.reshape(-1, 2)
                    if contour.shape[1] == 2:  # Verify it has x,y coordinates
                        valid_contours.append(contour)
            
            if not valid_contours:
                logger.error("No valid contours found in selected annotations")
                return False
            
            # Combine all contours into a single array of points
            all_points = np.vstack(valid_contours)
            
            # Compute the convex hull of all points
            hull = cv2.convexHull(all_points)
            
            # Ensure hull is in the correct format (N,2)
            hull = hull.reshape(-1, 2)
            
            # Create a new annotation with the convex hull as the initial contour
            new_annotation = Annotation(
                id=f"fused_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                source_image=selected_annotations[0].source_image,
                contour=hull,
                class_label=selected_annotations[0].class_label,  # Use the label of the first annotation
                confidence=selected_annotations[0].confidence,  # Use the confidence of the first annotation
                center_of_mass=selected_annotations[0].center_of_mass,  # Will be updated
                shape_tag="fused",
                properties={
                    "fused_from": [ann.id for ann in selected_annotations],
                    "fused_at": datetime.now().isoformat()
                }
            )
            
            # Try to refine the contour using SAM2 if the server is available
            if hasattr(self, 'image_handler') and self.parent.current_image_path:
                try:
                    # Use the new contour-based segmentation method
                    result = self.image_handler.request_contour_segmentation(
                        self.parent.current_image_path,
                        hull
                    )
                    
                    if result:
                        # Process the segmentation result
                        refined_annotation = self._process_segmentation_result(result, self.parent.current_image_path)
                        
                        if refined_annotation and refined_annotation.contour is not None:
                            # Ensure refined contour is in correct format
                            refined_contour = refined_annotation.contour.reshape(-1, 2)
                            if refined_contour.shape[1] == 2:
                                new_annotation.contour = refined_contour
                                # Keep the original class label and confidence
                                new_annotation.class_label = selected_annotations[0].class_label
                                new_annotation.confidence = selected_annotations[0].confidence
                            else:
                                logger.warning("Refined contour has invalid format, using convex hull")
                        else:
                            logger.warning("Failed to get valid refined contour, using convex hull")
                except Exception as e:
                    logger.exception(f"Error during contour refinement: {str(e)}")
                    # Continue with the convex hull if refinement fails
            
            # Remove the old annotations (in reverse order to maintain correct indices)
            for idx in sorted(self.selected_indices, reverse=True):
                self.annotations.pop(idx)
            
            # Add the new fused annotation
            self.annotations.append(new_annotation)
            
            # Sort annotations and clear selection
            self._sort_annotations()
            self.selected_indices = []
            
            # Mark that we have unsaved changes
            self.has_unsaved_changes = True
            return True
            
        except Exception as e:
            logger.exception(f"Error fusing annotations: {str(e)}")
            return False

    def remove_selected_annotations(self):
        """Remove all selected annotations"""
        if not self.selected_indices:
            return False

        try:
            # Remove annotations in reverse order to maintain correct indices
            for idx in sorted(self.selected_indices, reverse=True):
                self.annotations.pop(idx)
            
            # Clear selection
            self.selected_indices = []
            
            # Mark that we have unsaved changes
            self.has_unsaved_changes = True
            return True
            
        except Exception as e:
            logger.exception(f"Error removing annotations: {str(e)}")
            return False

    def change_selected_annotations_label(self, new_label: str):
        """Change the label of all selected annotations"""
        if not self.selected_indices:
            logger.warning("Attempted to change labels but no annotations are selected")
            return False

        try:
            logger.info(f"Changing label to {new_label} for {len(self.selected_indices)} annotations")
            for idx in self.selected_indices:
                old_label = self.annotations[idx].class_label
                self.annotations[idx].class_label = new_label
                logger.info(f"Changed annotation {idx} label from {old_label} to {new_label}")
            
            # Mark that we have unsaved changes
            self.has_unsaved_changes = True
            
            # Update the list display
            if hasattr(self.parent, 'lst_dnao'):
                self.update_annotations_list(self.parent.lst_dnao)
            
            # Update the statistics labels
            if hasattr(self.parent, 'update_statistics_labels'):
                self.parent.update_statistics_labels()
            
            # Refresh the annotation display
            if hasattr(self.parent, 'refresh_annotation_display'):
                self.parent.refresh_annotation_display()
            
            return True
            
        except Exception as e:
            logger.exception(f"Error changing annotation labels: {str(e)}")
            return False

    def set_current_label(self, label: str):
        """Set the label for selected annotations"""
        logger.info(f"Setting label to {label} for selected annotations")
        if self.selected_indices:
            return self.change_selected_annotations_label(label)
        else:
            logger.warning(f"No annotations selected to label as {label}")
            return False

    def image_mouse_press_event(self, event):
        """Handle mouse press events on the image"""
        # Only proceed if we have a valid pixmap
        if not hasattr(self.parent, 'current_pixmap') or not self.parent.current_pixmap:
            logger.warning("No current pixmap available for mouse event")
            return False
            
        # Check if Ctrl+Alt is pressed for multiple selection
        ctrl_alt_pressed = event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)
            
        if event.button() == Qt.LeftButton:
            # Convert click position to image coordinates for annotation detection
            image_pos = convert_to_image_coordinates(
                event.pos(), 
                self.parent.lbl_afm_img, 
                self.parent.current_pixmap
            )
            
            # First check if we clicked on an annotation
            is_inside, idx = is_point_in_any_annotation(self.annotations, image_pos)
            if is_inside and idx is not None:
                logger.info(f"Clicked on annotation {idx} at {image_pos.x()}, {image_pos.y()}")
                
                if ctrl_alt_pressed:
                    # Toggle the selection with Ctrl+Alt
                    if idx in self.selected_indices:
                        self.selected_indices.remove(idx)
                        logger.info(f"Removed annotation {idx} from selection")
                    else:
                        self.selected_indices.append(idx)
                        logger.info(f"Added annotation {idx} to selection")
                else:
                    # Select only this annotation
                    self.selected_indices = [idx]
                    logger.info(f"Selected only annotation {idx}")
                    # Clear the selection rectangle when directly selecting an annotation
                    self.selection_rect = None
                    
                # Update the list view selection to match the image selection
                if hasattr(self.parent, '_sync_list_selection_with_image'):
                    self.parent._sync_list_selection_with_image()
                    
                return True
            
            # If we didn't click on an annotation, check if we clicked within existing selection rectangle
            if self.selection_rect is not None:
                x1, y1, x2, y2 = self.selection_rect
                # Make sure coordinates are in order
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Check if click is inside the selection rectangle
                if (x1 <= event.pos().x() <= x2 and y1 <= event.pos().y() <= y2):
                    logger.info(f"Clicked inside selection rectangle at {event.pos().x()}, {event.pos().y()}")
                    # If rectangle clicked, don't clear it, just update selections
                    self._update_selected_indices()
                    return True
            
            # If we clicked on empty space and not in a selection rectangle
            logger.info(f"Clicked on empty space at {event.pos().x()}, {event.pos().y()}")
            self.selection_rect = None  # Clear any existing selection rectangle
            self.selected_indices = []  # Clear selected indices
            return True
                
        return False

    def clear_selection(self):
        """Clear the selection rectangle and selected annotations"""
        self.selection_rect = None
        self.selected_indices = []
        self.is_selecting = False
        
        logger.info("Selection cleared")
        
        # Refresh the annotation display if possible
        if hasattr(self.parent, 'refresh_annotation_display'):
            self.parent.refresh_annotation_display()

    def delete_selected_annotations(self):
        """Delete all selected annotations and select the next one"""
        if not self.selected_indices:
            logger.warning("No annotations selected for deletion")
            return False
            
        logger.info(f"Deleting {len(self.selected_indices)} selected annotations")
        
        # Store the index of the first selected annotation
        first_selected_idx = self.selected_indices[0]
        
        # Remove annotations in reverse order to maintain correct indices
        success = self.remove_selected_annotations()
        
        if success and self.annotations:
            # Calculate the index of the next annotation to select
            next_idx = min(first_selected_idx, len(self.annotations) - 1)
            
            # Select the next annotation
            self.selected_indices = [next_idx]
            
            # Update the list view selection
            if hasattr(self.parent, 'lst_dnao'):
                self.update_annotations_list(self.parent.lst_dnao)
            
            # Refresh the display
            if hasattr(self.parent, 'refresh_annotation_display'):
                self.parent.refresh_annotation_display()
        
        return success