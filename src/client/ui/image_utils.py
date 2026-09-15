import logging
import numpy as np
import cv2
from typing import Optional, Tuple
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel

logger = logging.getLogger(__name__)

def convert_to_image_coordinates(pos: QPoint, label: QLabel, pixmap: QPixmap) -> QPoint:
    """Convert coordinates from widget to image space"""
    if not pixmap or pixmap.isNull():
        return pos
        
    # Get the label's size and the pixmap's size
    label_size = label.size()
    pixmap_size = pixmap.size()
    
    # Calculate the scaled dimensions of the image in the label
    # This maintains aspect ratio
    scaled_width = label_size.width()
    scaled_height = int(scaled_width * (pixmap_size.height() / pixmap_size.width()))
    
    # If the scaled height is too large, scale based on height instead
    if scaled_height > label_size.height():
        scaled_height = label_size.height()
        scaled_width = int(scaled_height * (pixmap_size.width() / pixmap_size.height()))
    
    # Calculate the offset to center the image in the label
    offset_x = (label_size.width() - scaled_width) / 2
    offset_y = (label_size.height() - scaled_height) / 2
    
    # Get the position relative to the label
    x = pos.x()
    y = pos.y()
    
    # Adjust the position by the offset
    x = x - offset_x
    y = y - offset_y
    
    # Calculate scaling factors based on the actual scaled dimensions
    scale_x = pixmap_size.width() / scaled_width
    scale_y = pixmap_size.height() / scaled_height
    
    # Convert to image coordinates
    image_x = int(x * scale_x)
    image_y = int(y * scale_y)
    
    # Ensure coordinates are within image bounds
    image_x = max(0, min(image_x, pixmap_size.width() - 1))
    image_y = max(0, min(image_y, pixmap_size.height() - 1))
    
    return QPoint(image_x, image_y)

def convert_to_widget_coordinates(pos: QPoint, label: QLabel, pixmap: QPixmap) -> QPoint:
    """Convert coordinates from image space to widget space"""
    if not pixmap or pixmap.isNull():
        return pos
        
    # Get the label's size and the pixmap's size
    label_size = label.size()
    pixmap_size = pixmap.size()
    
    # Calculate scaling factors
    scale_x = label_size.width() / pixmap_size.width()
    scale_y = label_size.height() / pixmap_size.height()
    
    # Get the position in image coordinates
    x = pos.x()
    y = pos.y()
    
    # Convert to widget coordinates
    widget_x = int(x * scale_x)
    widget_y = int(y * scale_y)
    
    return QPoint(widget_x, widget_y)

def scale_pixmap_to_label(pixmap: QPixmap, label: QLabel) -> QPixmap:
    """Scale a pixmap to fit a label while maintaining aspect ratio"""
    if not pixmap or pixmap.isNull():
        return pixmap
        
    # Get the label's size and the pixmap's size
    label_size = label.size()
    pixmap_size = pixmap.size()
    
    # Scale to fit the label
    scaled_width = label_size.width()
    scaled_height = int(scaled_width * (pixmap_size.height() / pixmap_size.width()))
    
    if scaled_height > label_size.height():
        scaled_height = label_size.height()
        scaled_width = int(scaled_height * (pixmap_size.width() / pixmap_size.height()))
    
    return pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def find_closest_annotation(annotations, position):
    """Find the closest annotation to the given position by checking all contour vertices"""
    closest_idx = None
    min_distance = float('inf')
    
    # Create position array for broadcasting
    pos_array = np.array([position.x(), position.y()])
    
    for i, annotation in enumerate(annotations):
        # Get the contour vertices
        contour = annotation.contour
        
        if contour is not None and len(contour) > 0:
            # Vectorized distance calculation to all vertices at once
            distances = np.sqrt(np.sum((contour - pos_array)**2, axis=1))
            
            # Find the minimum distance for this contour
            annotation_min_distance = np.min(distances)
            
            # Update if closer than current minimum
            if annotation_min_distance < min_distance:
                min_distance = annotation_min_distance
                closest_idx = i
    
    return closest_idx, min_distance

def is_point_in_any_annotation(annotations, point):
    """Check if a point is inside any annotation"""
    test_point = (float(point.x()), float(point.y()))
    
    for annotation in annotations:
        contour = annotation.contour
        if contour is not None and len(contour) > 0:
            result = cv2.pointPolygonTest(contour, test_point, False)
            if result >= 0:  # Point is inside or on the contour
                return True, annotations.index(annotation)
    
    return False, None
