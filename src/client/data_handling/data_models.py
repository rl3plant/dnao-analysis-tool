from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import os
import json
from datetime import datetime
from threading import Lock
from enum import Enum
import cv2
import logging

class TaskType(Enum):
    """Enum for different types of annotation tasks"""
    YIELD = "yield"  # Default task for yield analysis
    CHIRALITY = "chirality"  # Task for chirality analysis

@dataclass
class Annotation:
    """Data class representing a single DNAO annotation"""
    id: str
    source_image: str
    contour: np.ndarray
    shape_tag: str = "unknown"
    class_label: str = "intact"  # intact, damaged, etc.
    confidence: float = 1.0
    is_valid: bool = True
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    appearance: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    center_of_mass: Optional[tuple] = None  # Store pre-computed center coordinates
    
    def __post_init__(self):
        """Calculate center of mass after initialization if not provided"""
        if self.center_of_mass is None and self.contour is not None:
            self.update_center_of_mass()
    
    def update_center_of_mass(self):
        """Update the center of mass based on the current contour"""
        if self.contour is not None and len(self.contour) > 0:
            # contour is normally OpenCV's (N, 1, 2) point format (e.g. straight
            # from cv2.findContours/SAM), not (N, 2) - reshape first so this is a
            # real 2D centroid (mean_x, mean_y), not np.mean(..., axis=0) over
            # the wrong axis (which silently returns a 1-element array instead).
            centroid = np.mean(self.contour.reshape(-1, 2), axis=0)
            self.center_of_mass = (float(centroid[0]), float(centroid[1]))
        else:
            self.center_of_mass = (0.0, 0.0)
    
    def to_dict(self) -> Dict:
        """Convert annotation to dictionary for serialization"""
        result = {
            "id": self.id,
            "source_image": self.source_image,
            "contour": self.contour.tolist() if self.contour is not None else None,
            "shape_tag": self.shape_tag,
            "class_label": self.class_label,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "created": self.created,
            "modified": self.modified,
            "properties": self.properties,
            "center_of_mass": self.center_of_mass,
        }
        # Only include appearance if it exists
        if self.appearance is not None:
            result["appearance"] = self.appearance.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Annotation':
        """Create annotation from dictionary"""
        # Convert contour to numpy array if present
        contour = data.get('contour')
        if contour is not None:
            contour = np.array(contour, dtype=np.int32)
            
        # Convert appearance to numpy array if present
        appearance = data.get('appearance')
        if appearance is not None:
            appearance = np.array(appearance)
            
        return cls(
            id=data.get('id'),
            source_image=data.get('source_image'),
            contour=contour,
            shape_tag=data.get('shape_tag', 'unknown'),
            class_label=data.get('class_label', 'intact'),
            confidence=data.get('confidence', 1.0),
            is_valid=data.get('is_valid', True),
            created=data.get('created', ''),
            modified=data.get('modified', ''),
            appearance=appearance,
            properties=data.get('properties', {}),
            center_of_mass=data.get('center_of_mass')
        )

@dataclass
class ImageStatistics:
    """Statistics for a single image"""
    median_contour_area: float = 0.0
    mean_contour_area: float = 0.0
    std_contour_area: float = 0.0
    corrections_added: int = 0
    corrections_removed: int = 0
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update_contour_stats(self, contours: List[np.ndarray]) -> None:
        """Update contour statistics"""
        if not contours:
            return
            
        areas = [cv2.contourArea(contour) for contour in contours]
        self.median_contour_area = float(np.median(areas))
        self.mean_contour_area = float(np.mean(areas))
        self.std_contour_area = float(np.std(areas))
        self.last_modified = datetime.now().isoformat()
    
    def increment_corrections(self, added: bool = True) -> None:
        """Increment correction counter"""
        if added:
            self.corrections_added += 1
        else:
            self.corrections_removed += 1
        self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "median_contour_area": self.median_contour_area,
            "mean_contour_area": self.mean_contour_area,
            "std_contour_area": self.std_contour_area,
            "corrections_added": self.corrections_added,
            "corrections_removed": self.corrections_removed,
            "last_modified": self.last_modified
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageStatistics':
        return cls(
            median_contour_area=data.get('median_contour_area', 0.0),
            mean_contour_area=data.get('mean_contour_area', 0.0),
            std_contour_area=data.get('std_contour_area', 0.0),
            corrections_added=data.get('corrections_added', 0),
            corrections_removed=data.get('corrections_removed', 0),
            last_modified=data.get('last_modified', datetime.now().isoformat())
        )

@dataclass
class Session:
    """Data class representing an annotation session"""
    name: str
    dnao_types: List[str] = field(default_factory=list)
    image_files: List[str] = field(default_factory=list)
    folder_paths: Dict[str, str] = field(default_factory=dict)  # Map image names to their folder paths
    annotations: Dict[str, List[Annotation]] = field(default_factory=dict)
    preprocessed_images: Dict[str, Dict] = field(default_factory=dict)  # Store preprocessed image data
    properties: Dict[str, Any] = field(default_factory=dict)  # General purpose properties storage
    task_type: TaskType = TaskType.YIELD  # Default to yield analysis
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    image_statistics: Dict[str, ImageStatistics] = field(default_factory=dict)
    
    # Classifier-related fields
    classifier_type: str = "svm_classifier"  # Type of classifier to use
    retrain_frequency: int = 0  # Retrain every k images (default: every image)
    images_since_retrain: int = 0  # Counter for retraining
    classifier_path: Optional[str] = None  # Path to saved classifier model
    # KMeans model for chirality classification (not serialized)
    _kmeans_model: Any = field(default=None, repr=False, compare=False)
    
    def __post_init__(self):
        # Add a lock for thread safety
        self._lock = Lock()
    
    def add_image(self, image_path: str) -> None:
        """Add an image to the session"""
        image_name = os.path.basename(image_path)
        if image_name not in self.image_files:
            self.image_files.append(image_name)
            self.annotations[image_name] = []
            # Store the folder path for this specific image
            self.folder_paths[image_name] = os.path.dirname(image_path)
            self.modified = datetime.now().isoformat()
    
    def get_image_path(self, image_name: str) -> Optional[str]:
        """Get the full path for an image"""
        if image_name in self.folder_paths:
            return os.path.join(self.folder_paths[image_name], image_name)
        return None
    
    def remove_image(self, image_name: str) -> None:
        """Remove an image from the session"""
        if image_name in self.image_files:
            self.image_files.remove(image_name)
            if image_name in self.annotations:
                del self.annotations[image_name]
            if image_name in self.preprocessed_images:
                del self.preprocessed_images[image_name]
            if image_name in self.folder_paths:
                del self.folder_paths[image_name]
            self.modified = datetime.now().isoformat()
    
    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation to the session"""
        image_name = os.path.basename(annotation.source_image)
        if image_name not in self.image_files:
            self.image_files.append(image_name)
        
        if image_name not in self.annotations:
            self.annotations[image_name] = []
            
        self.annotations[image_name].append(annotation)
        self.modified = datetime.now().isoformat()
    
    def remove_annotation(self, image_name: str, annotation_id: str) -> bool:
        """Remove an annotation from the session"""
        if image_name in self.annotations:
            for i, ann in enumerate(self.annotations[image_name]):
                if ann.id == annotation_id:
                    self.annotations[image_name].pop(i)
                    self.modified = datetime.now().isoformat()
                    return True
        return False
    
    def get_intact_count(self) -> int:
        """Get count of intact DNAO in the session"""
        count = 0
        for image_anns in self.annotations.values():
            count += sum(1 for ann in image_anns if ann.class_label == "intact")
        return count
    
    def get_damaged_count(self) -> int:
        """Get count of damaged DNAO in the session"""
        count = 0
        for image_anns in self.annotations.values():
            count += sum(1 for ann in image_anns if ann.class_label == "damaged")
        return count
    
    def get_z_shape_count(self) -> int:
        """Get count of Z-shape DNAO in the session"""
        count = 0
        for image_anns in self.annotations.values():
            count += sum(1 for ann in image_anns if ann.class_label == "Z-shape")
        return count
    
    def get_s_shape_count(self) -> int:
        """Get count of S-shape DNAO in the session"""
        count = 0
        for image_anns in self.annotations.values():
            count += sum(1 for ann in image_anns if ann.class_label == "S-shape")
        return count
    
    def get_annotation_count(self) -> int:
        """Get total count of annotations in the session"""
        return sum(len(anns) for anns in self.annotations.values())
    
    def get_preprocessed_image(self, image_name: str) -> Optional[Dict]:
        """Thread-safe access to preprocessed image data"""
        with self._lock:
            return self.preprocessed_images.get(image_name)
    
    def set_preprocessed_image(self, image_name: str, data: Dict) -> None:
        """Thread-safe setting of preprocessed image data"""
        with self._lock:
            self.preprocessed_images[image_name] = data
            self.modified = datetime.now().isoformat()
    
    def update_image_statistics(self, image_name: str) -> None:
        """Update statistics for a specific image"""
        if image_name not in self.annotations:
            return
            
        if image_name not in self.image_statistics:
            self.image_statistics[image_name] = ImageStatistics()
            
        contours = [ann.contour for ann in self.annotations[image_name] if ann.contour is not None]
        self.image_statistics[image_name].update_contour_stats(contours)
        self.modified = datetime.now().isoformat()
    
    def track_correction(self, image_name: str, added: bool = True) -> None:
        """Track a correction made to an image"""
        if image_name not in self.image_statistics:
            self.image_statistics[image_name] = ImageStatistics()
            
        self.image_statistics[image_name].increment_corrections(added)
        self.modified = datetime.now().isoformat()
    
    def get_image_statistics(self, image_name: str) -> Optional[ImageStatistics]:
        """Get statistics for a specific image"""
        return self.image_statistics.get(image_name)
    
    def should_retrain_classifier(self) -> bool:
        """
        Check if the classifier should be retrained based on retrain frequency.
        
        Returns:
            True if classifier should be retrained
        """
        return self.images_since_retrain >= self.retrain_frequency
    
    def increment_retrain_counter(self) -> None:
        """Increment the retrain counter and check if retraining is needed."""
        self.images_since_retrain += 1
        self.modified = datetime.now().isoformat()
    
    def reset_retrain_counter(self) -> None:
        """Reset the retrain counter after training."""
        self.images_since_retrain = 0
        self.modified = datetime.now().isoformat()
    
    def _safe_classifier_filename(self) -> str:
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in self.name)
        return f"classifier_{safe_name}.pkl"

    def get_classifier_path(self) -> Optional[str]:
        """
        Get the path where the classifier should be saved/loaded.

        Falls back to a bare filename (relative to the current working
        directory) if nothing has called set_default_classifier_path() yet -
        SessionController does that on create/load so this fallback is really
        only hit by code (e.g. tests) that builds a Session directly.

        Returns:
            Path to classifier file or None if not set
        """
        if not self.classifier_path:
            self.classifier_path = self._safe_classifier_filename()

        return self.classifier_path

    def set_default_classifier_path(self, directory: str) -> None:
        """Point the classifier file at `directory` (normally the session
        manager's storage dir), so it lives next to the session's own data
        instead of wherever the app happens to be run from."""
        self.classifier_path = os.path.join(directory, self._safe_classifier_filename())

    def set_classifier_path(self, path: str) -> None:
        """
        Set the path for the classifier model.
        
        Args:
            path: Path to classifier file
        """
        self.classifier_path = path
        self.modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Thread-safe conversion to dictionary"""
        with self._lock:
            annotations_dict = {}
            for img, anns in self.annotations.items():
                annotations_dict[img] = [ann.to_dict() for ann in anns]
                
            return {
                "name": self.name,
                "dnao_types": self.dnao_types,
                "image_files": self.image_files,
                "folder_paths": self.folder_paths,
                "annotations": annotations_dict,
                "preprocessed_images": dict(self.preprocessed_images),  # Create a copy
                "properties": dict(self.properties),  # Create a copy
                "task_type": self.task_type.value,
                "created": self.created,
                "modified": self.modified,
                "image_statistics": {
                    img: stats.to_dict() 
                    for img, stats in self.image_statistics.items()
                },
                "classifier_type": self.classifier_type,
                "retrain_frequency": self.retrain_frequency,
                "images_since_retrain": self.images_since_retrain,
                "classifier_path": self.classifier_path
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Session':
        """Create session from dictionary"""
        # Debug logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Creating session from dict, task_type from data: {data.get('task_type', 'Not found')}")
        
        session = cls(
            name=data.get('name', 'Unnamed Session'),
            dnao_types=data.get('dnao_types', []),
            image_files=data.get('image_files', []),
            folder_paths=data.get('folder_paths', {}),  # Restore all folder paths
            preprocessed_images=data.get('preprocessed_images', {}),
            properties=data.get('properties', {}),  # Restore properties
            task_type=TaskType(data.get('task_type', 'yield')),
            created=data.get('created', ''),
            modified=data.get('modified', ''),
            classifier_type=data.get('classifier_type', 'svm_classifier'),
            retrain_frequency=data.get('retrain_frequency', 1),
            images_since_retrain=data.get('images_since_retrain', 0),
            classifier_path=data.get('classifier_path')
        )
        
        # Debug logging after creation
        logger.debug(f"Created session with task_type: {session.task_type}, type: {type(session.task_type)}")
        
        # Load annotations
        annotations_dict = data.get('annotations', {})
        for img, anns_data in annotations_dict.items():
            # Create annotations and ensure they have center_of_mass
            annotations = [Annotation.from_dict(ann) for ann in anns_data]
            # Sort annotations by center of mass
            annotations.sort(key=lambda a: (a.center_of_mass[1], a.center_of_mass[0]))
            session.annotations[img] = annotations
        
        # Load image statistics
        session.image_statistics = {
            img: ImageStatistics.from_dict(stats)
            for img, stats in data.get('image_statistics', {}).items()
        }
            
        return session
