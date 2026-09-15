"""
SVM-based classifier for segmentation classification.
"""

import numpy as np
import joblib
import logging
from typing import List, Tuple
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

class SVMClassifier(BaseClassifier):
    """
    One-Class SVM-based outlier detector for segmentation mask filtering.
    Trained only on 'valid' (good) segmentations. Predicts if a new segmentation is valid (inlier) or invalid (outlier).
    """
    def __init__(self, name: str = "oneclass_svm_classifier"):
        super().__init__(name)
        self.model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # nu is the expected outlier fraction
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_samples = 0

    def fit(self, features: np.ndarray, labels: List[str] = None) -> None:
        """
        Train the One-Class SVM on 'valid' segmentations only.
        Args:
            features: 2D numpy array of feature vectors (n_samples, n_features)
            labels: (optional, ignored) for compatibility
        """
        if len(features) == 0:
            logger.warning("No training data provided")
            return
        try:
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled)
            self.is_trained = True
            self.training_samples = len(features)
            logger.info(f"Trained One-Class SVM with {len(features)} samples")
        except Exception as e:
            logger.error(f"Error training One-Class SVM: {e}")
            raise

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict if the given features are 'valid' (inlier) or 'invalid' (outlier).
        Args:
            features: 1D or 2D numpy array of feature vectors
        Returns:
            Tuple of ("valid"/"invalid", confidence_score)
        """
        if not self.can_predict():
            logger.warning("Classifier not trained, returning default prediction")
            return "valid", 0.5
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            pred = self.model.predict(features_scaled)[0]  # 1 for inlier, -1 for outlier
            score = self.model.decision_function(features_scaled)[0]
            label = "valid" if pred == 1 else "invalid"
            # Confidence: scale decision function to [0,1] using a sigmoid
            confidence = 1 / (1 + np.exp(-score))
            return label, float(confidence)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "valid", 0.5

    def score(self, features: np.ndarray) -> float:
        """
        Get outlier score for filtering purposes (higher = more likely valid).
        Args:
            features: 1D or 2D numpy array of feature vectors
        Returns:
            Score between 0 and 1 (sigmoid of decision function)
        """
        if not self.can_predict():
            logger.warning("Classifier not trained, returning default score")
            return 0.5
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            score = self.model.decision_function(features_scaled)[0]
            confidence = 1 / (1 + np.exp(-score))
            return float(confidence)
        except Exception as e:
            logger.error(f"Error computing score: {e}")
            return 0.5

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return
        
        try:
            # Save both model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'training_samples': self.training_samples,
                'name': self.name
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Saved SVM classifier to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load the model from
        """
        try:
            # Load model data
            model_data = joblib.load(path)
            
            # Restore components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.training_samples = model_data['training_samples']
            self.name = model_data.get('name', self.name)
            
            # Update training status
            self.is_trained = True
            
            logger.info(f"Loaded SVM classifier from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise 