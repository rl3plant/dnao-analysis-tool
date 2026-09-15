"""
Base classifier interface for segmentation classification.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseClassifier(ABC):
    """
    Base interface for all segmentation classifiers.
    
    This abstract class defines the contract that all classifiers must implement.
    """
    
    def __init__(self, name: str = "base_classifier"):
        """
        Initialize the base classifier.
        
        Args:
            name: Name identifier for this classifier
        """
        self.name = name
        self.is_trained = False
        self.training_samples = 0
    
    @abstractmethod
    def fit(self, features: np.ndarray, labels: List[str]) -> None:
        """
        Train the classifier with features and labels.
        
        Args:
            features: 2D numpy array of feature vectors (n_samples, n_features)
            labels: List of class labels corresponding to features
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict class and confidence score for given features.
        
        Args:
            features: 1D or 2D numpy array of feature vectors
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        pass
    
    @abstractmethod
    def score(self, features: np.ndarray) -> float:
        """
        Get a quality score (0-1) for filtering purposes.
        Higher score means better quality mask.
        
        Args:
            features: 1D or 2D numpy array of feature vectors
            
        Returns:
            Quality score between 0 and 1
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load the model from
        """
        pass
    
    def can_predict(self) -> bool:
        """
        Check if the classifier is ready to make predictions.
        
        Returns:
            True if classifier is trained and ready
        """
        return self.is_trained and self.training_samples > 0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the classifier.
        
        Returns:
            Dictionary with classifier information
        """
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "training_samples": self.training_samples
        } 