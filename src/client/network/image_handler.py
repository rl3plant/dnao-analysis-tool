import os
import logging
import json
import numpy as np
from typing import List, Dict, Optional, Any
import cv2
from datetime import datetime
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from PySide6.QtCore import Qt, QPointF, QMetaObject, Q_ARG, Slot

# Fix the import path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_handling.data_models import Annotation
from annotation.shape_modeling import MaskFilter
from utils.config import get_settings
from pipeline.instance_segmentation import run_pipeline
from local_sam_service import get_shared_service

logger = logging.getLogger(__name__)
settings = get_settings()

class ImageHandler:
    """Handles image processing and (now in-process, formerly HTTP) segmentation.

    Nothing here makes network requests -- see local_sam_service.py.
    """

    def __init__(self):
        logger.info("Image handler initialized (in-process SAM2 segmentation)")
    
    def get_image_files_in_folder(self, folder_path: str) -> List[str]:
        """Get all image files in the given folder
        
        Args:
            folder_path: Path to the folder
            
        Returns:
            List of image filenames
        """
        # Common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        
        # Get all files in the folder
        files = []
        try:
            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    files.append(filename)
            
            files.sort()  # Sort alphabetically
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {folder_path}: {e}")
            return []
    
    def generate_masks(self, image_path: str) -> Optional[Dict]:
        """Request mask generation from server or load precomputed contours
        
        Args:
            image_path: Path to the image
            
        Returns:
            Server response or None if the request failed
        """
        # Check if we should use precomputed contours
        if settings.get('USE_PRECOMPUTED_CONTOURS', False):
            logger.info(f"Using precomputed contours for {image_path}")
            return self.load_precomputed_contours(image_path)
        
        try:
            # Get current session if available
            session = None
            if hasattr(self, 'session'):
                session = self.session
            
            # Run pipeline
            masks = run_pipeline(image_path, session)
            
            # Return in same format as server response
            return {'masks': masks}
            
        except Exception as e:
            logger.exception(f"Error generating masks: {e}")
            return None
    
    def request_segmentation(self, image_path: str, x: int, y: int) -> Optional[Dict]:
        """Segment at a point prompt (in-process SAM2) or find in precomputed contours.

        Args:
            image_path: Path to the image
            x: X coordinate of the point
            y: Y coordinate of the point

        Returns:
            A dict with 'predictions' (as before) and also 'prediction' (singular)
            set to that single point's mask list -- _process_segmentation_result()
            reads the singular key, but only the SegmentationSelectionDialog path
            used to set it, so a single-candidate point click always fell through
            to "Failed to create annotation" before. Fixed here at the source
            instead of touching the parsing code in annotation_controller.py.
        """
        if settings.get('USE_PRECOMPUTED_CONTOURS', False):
            logger.info(f"Using precomputed contours for point segmentation at ({x}, {y})")
            return self.find_contour_at_point(image_path, x, y)

        try:
            response = get_shared_service().segment_points(image_path, [(x, y)])
            if not response:
                logger.error("Segmentation failed")
                return None
            response['prediction'] = response['predictions'][0]['masks']
            return response
        except Exception as e:
            logger.exception(f"Error requesting segmentation: {e}")
            return None

    def request_contour_segmentation(self, image_path: str, contour: np.ndarray) -> Optional[Dict]:
        """Refine a contour (e.g. after fusing several annotations) via box-prompt SAM2.

        The old tool posted to a `/segment-mask` server route that never
        existed (checked src/server/app.py's routes) -- this always failed
        and fell back to the raw convex hull. Now actually works.
        """
        try:
            response = get_shared_service().segment_contour(image_path, contour)
            if not response:
                return None
            response['prediction'] = response['predictions'][0]['masks']
            return response
        except Exception as e:
            logger.exception(f"Error requesting contour segmentation: {e}")
            return None

    def request_bbox_segmentation(self, image_path: str, bbox: Dict[str, int]) -> Optional[Dict]:
        """Segment within a bounding box prompt (in-process SAM2)."""
        try:
            response = get_shared_service().segment_bbox(image_path, bbox)
            if not response:
                logger.error("Bbox segmentation failed")
                return None
            response['prediction'] = response['predictions'][0]['masks']
            logger.info("Successfully received bounding box segmentation response")
            return response
        except Exception as e:
            logger.exception(f"Error requesting bbox segmentation: {e}")
            return None
    
    def process_masks(self, server_response: Dict, image_path: str) -> List[Annotation]:
        """Process mask data from server into Annotation objects
        
        Args:
            server_response: Response from the server
            image_path: Path to the source image
            
        Returns:
            List of Annotation objects
        """
        annotations = []
        
        try:
            # Extract masks from the response
            masks = server_response.get('masks', [])
            
            if not masks:
                logger.warning("No masks in server response")
                return []
            
            # Apply filters to the masks
            # Get filter settings from config or use defaults
            area_threshold = settings.get('MASK_AREA_THRESHOLD', 0.5)
            
            # Create and apply filters
            filters = [
                MaskFilter.create_area_filter(threshold_factor=area_threshold)
            ]
            
            # Apply all filters
            filtered_masks = MaskFilter.apply_filters(masks, filters)
            logger.info(f"Filtered {len(masks)} masks to {len(filtered_masks)} masks")
                
            # Process each mask
            for i, mask in enumerate(filtered_masks):
                # Extract contour data
                contours_data = mask.get('contours', [])
                
                if not contours_data or len(contours_data) == 0:
                    logger.warning(f"No contour data for mask {i}")
                    continue
                
                # The server returns a list of contours, but we only use the first one
                if contours_data and len(contours_data) > 0:
                    contour_points = contours_data[0]  # Get the first contour
                    
                    # Convert to numpy array
                    contour_np = np.array(contour_points, dtype=np.int32)
                    
                    # Create annotation
                    annotation = Annotation(
                        id=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                        source_image=image_path,
                        contour=contour_np,
                        shape_tag="auto",
                        class_label="intact",  # Default to intact
                        confidence=1.0,
                        properties={
                            "area": mask.get('area', 0),
                            "from_server": True
                        }
                    )
                    
                    annotations.append(annotation)
                
            logger.info(f"Processed {len(annotations)} annotations from server response")
            return annotations
            
        except Exception as e:
            logger.exception(f"Error processing masks: {e}")
            return []
    
    def draw_annotations(self, pixmap: QPixmap, annotations: List[Annotation], selected_index: int = -1) -> QPixmap:
        """Draw annotations on a pixmap
        
        Args:
            pixmap: The source pixmap
            annotations: List of annotations to draw
            selected_index: Index of the currently selected annotation
            
        Returns:
            New pixmap with annotations drawn
        """
        if not annotations:
            logger.warning("No annotations to draw")
            return pixmap
            
        # Create a copy of the pixmap to draw on
        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw each annotation
        for i, annotation in enumerate(annotations):
            try:
                # Set color based on class label
                if annotation.class_label == "intact":
                    color = QColor(0, 255, 0, 128)  # Green (semi-transparent)
                elif annotation.class_label == "damaged":
                    color = QColor(255, 0, 0, 128)  # Red (semi-transparent)
                elif annotation.class_label == "invalid":
                    color = QColor(128, 128, 128, 128)  # Gray (semi-transparent)
                else:
                    color = QColor(0, 0, 255, 128)  # Blue (semi-transparent)
                
                # Draw contour if available
                if annotation.contour is not None and len(annotation.contour) > 0:
                    # Set pen for contour
                    contour_pen = QPen(color.darker(150), 3)
                    painter.setPen(contour_pen)
                    
                    # Create QPoint array from contour points
                    points = []
                    for point in annotation.contour:
                        if len(point) >= 2:
                            points.append(QPointF(point[0], point[1]))
                    
                    # Draw the polygon
                    if len(points) > 2:
                        # Draw the outline
                        painter.drawPolyline(points)
                        # Close the contour
                        painter.drawLine(points[-1], points[0])
                        
                        # Fill with a semi-transparent version of the color
                        fill_color = QColor(color)
                        fill_color.setAlpha(50)  # Very transparent
                        painter.setBrush(QBrush(fill_color))
                        painter.setPen(Qt.NoPen)  # No outline for the fill
                        painter.drawPolygon(points)
                        
                        # Draw center point for the selected annotation
                        if i == selected_index:
                            # Calculate center point
                            center_x = sum(p.x() for p in points) / len(points)
                            center_y = sum(p.y() for p in points) / len(points)
                            
                            # Draw blue center point
                            center_color = QColor(255, 255, 255)  # White
                            painter.setPen(QPen(center_color, 5))
                            painter.setBrush(QBrush(center_color))
                            painter.drawEllipse(QPointF(center_x, center_y), 5, 5)
                        
            except Exception as e:
                logger.exception(f"Error drawing annotation {i}: {str(e)}")
        
        painter.end()
        return result
    
    def load_precomputed_contours(self, image_path: str) -> Optional[Dict]:
        """Load precomputed contours from JSON files in the ../shapes directory
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with contours in the same format as server response
        """
        try:
            # Get image directory and name
            image_dir = os.path.dirname(image_path)
            image_name = os.path.basename(image_path)
            image_name_without_ext = os.path.splitext(image_name)[0]
            
            # Get shapes directory (../shapes relative to image directory)
            shapes_dir = os.path.join(image_dir, "..", "shapes")
            
            if not os.path.exists(shapes_dir):
                logger.error(f"Shapes directory not found: {shapes_dir}")
                return None
                
            logger.info(f"Looking for shape files in {shapes_dir} matching {image_name_without_ext}")
            
            # Find all matching shape files
            shape_files = []
            for filename in os.listdir(shapes_dir):
                if filename.startswith(image_name_without_ext) and "_shape_" in filename and filename.endswith(".json"):
                    shape_files.append(os.path.join(shapes_dir, filename))
            
            if not shape_files:
                logger.warning(f"No shape files found for {image_name}")
                return None
                
            logger.info(f"Found {len(shape_files)} shape files")
            
            # Load all shape files and extract contours
            masks = []
            for i, shape_file in enumerate(shape_files):
                try:
                    with open(shape_file, 'r') as f:
                        shape_data = json.load(f)
                    
                    # If the shape file contains a full mask entry, use it directly
                    if isinstance(shape_data, dict) and 'contours' in shape_data:
                        masks.append(shape_data)
                    else:
                        # Otherwise, assume the file contains contour points
                        # and create a mask entry
                        contour_points = shape_data
                        mask = {
                            'contours': contour_points,
                            'area': self.calculate_contour_area(contour_points),
                            'score': 1.0,  # Default score
                            'id': i
                        }
                        masks.append(mask)
                        
                except Exception as e:
                    logger.error(f"Error loading shape file {shape_file}: {e}")
            
            logger.info(f"Loaded {len(masks)} masks from shape files")
            
            # Return in same format as server response
            return {'masks': masks}
                
        except Exception as e:
            logger.exception(f"Error loading precomputed contours: {e}")
            return None
    
    def calculate_contour_area(self, contour_points: List) -> float:
        """Calculate the area of a contour
        
        Args:
            contour_points: List of contour points [[x1, y1], [x2, y2], ...]
            
        Returns:
            Area of the contour
        """
        try:
            contour_np = np.array(contour_points, dtype=np.int32)
            area = cv2.contourArea(contour_np)
            return float(area)
        except Exception as e:
            logger.error(f"Error calculating contour area: {e}")
            return 0.0
    
    def find_contour_at_point(self, image_path: str, x: int, y: int) -> Optional[Dict]:
        """Find a precomputed contour that contains the given point
        
        Args:
            image_path: Path to the image
            x: X coordinate of the point
            y: Y coordinate of the point
            
        Returns:
            Dictionary containing the contour that contains the point
        """
        try:
            # Load all precomputed contours
            contours_data = self.load_precomputed_contours(image_path)
            if not contours_data or 'masks' not in contours_data:
                return None
                
            # Check each contour to see if it contains the point
            for mask in contours_data['masks']:
                contours = mask.get('contours', [])
                if not contours:
                    continue
                    
                # Check the first contour (main boundary)
                contour_points = contours[0]
                contour_np = np.array(contour_points, dtype=np.int32)
                
                # Check if point is inside contour
                if cv2.pointPolygonTest(contour_np, (float(x), float(y)), False) >= 0:
                    # Found a contour containing the point
                    logger.info(f"Found contour containing point ({x}, {y})")
                    
                    # Return in format compatible with server response
                    return {
                        'point': [float(x), float(y)],
                        'prediction': mask
                    }
            
            logger.warning(f"No contour found containing point ({x}, {y})")
            return None
                
        except Exception as e:
            logger.exception(f"Error finding contour at point: {e}")
            return None
    
    def embed_image(self, image_path: str) -> bool:
        """Compute (and cache) the SAM2 embedding for an image, in-process.

        No longer actually needs a session/image id (that was for keying the
        embedding cache on the old server, keyed by path here instead), but
        embed_image_with_id() keeps that parameter for call-site compatibility.
        """
        try:
            return get_shared_service().embed(image_path)
        except Exception as e:
            logger.exception(f"Error embedding image: {e}")
            return False

    def embed_image_with_id(self, image_path: str, session_name: str) -> bool:
        """Same as embed_image(); session_name is unused now (see embed_image)."""
        return self.embed_image(image_path)
            
    def preprocess_image(self, image_path: str) -> Optional[Dict]:
        """Preprocess an image by generating masks
        
        Args:
            image_path: Path to the image
            
        Returns:
            Server response or None if the request failed
        """
        # This method is no longer used in the main workflow
        # We keep it for backwards compatibility but log a warning
        logger.warning("preprocess_image is deprecated - using direct generate_masks instead")
        return self.generate_masks(image_path)
