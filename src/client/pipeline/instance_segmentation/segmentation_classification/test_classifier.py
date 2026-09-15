"""
Test script for the segmentation classification module.
"""

import numpy as np
import logging
from typing import Dict, List
from src.client.pipeline.features_shared import extract_mask_features, extract_features_batch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_mask(area: float = 100.0, num_vertices: int = 5) -> Dict:
    """Create a test mask for testing."""
    # Create a simple polygon contour
    angles = np.linspace(0, 2*np.pi, num_vertices, endpoint=False)
    radius = np.sqrt(area / np.pi)  # Approximate radius for given area
    
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    contour = np.column_stack([x, y]).astype(np.float32)
    
    # Calculate bounding box
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)
    bbox = [x_min, y_min, x_max, y_max]
    
    return {
        'contours': [contour],
        'area': area,
        'bbox': bbox
    }

def test_feature_extraction():
    """Test feature extraction."""
    logger.info("Testing feature extraction...")
    
    # Create test masks
    test_masks = [
        create_test_mask(100.0, 5),  # Small, few vertices
        create_test_mask(200.0, 8),  # Medium, more vertices
        create_test_mask(50.0, 3),   # Very small, few vertices
    ]
    
    for i, mask in enumerate(test_masks):
        features = extract_mask_features(mask)
        logger.info(f"Mask {i+1} features: {features}")
        logger.info(f"Feature shape: {features.shape}")
    
    logger.info("Feature extraction test completed!")

def test_svm_classifier():
    """Test SVM classifier."""
    logger.info("Testing SVM classifier...")
    
    from .svm_classifier import SVMClassifier
    
    # Create test data
    test_masks = [
        create_test_mask(100.0, 5),
        create_test_mask(200.0, 8),
        create_test_mask(50.0, 3),
        create_test_mask(150.0, 6),
    ]
    
    # Extract features
    features = extract_features_batch(test_masks)
    logger.info(f"Extracted features shape: {features.shape}")
    
    # Create labels (simulate user annotations)
    labels = ["good", "good", "bad", "good"]
    
    # Create and train classifier
    classifier = SVMClassifier("test_classifier")
    classifier.fit(features, labels)
    
    logger.info(f"Classifier info: {classifier.get_info()}")
    
    # Test prediction
    test_features = extract_mask_features(create_test_mask(120.0, 6))
    prediction, confidence = classifier.predict(test_features)
    score = classifier.score(test_features)
    
    logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
    logger.info(f"Score: {score:.3f}")
    
    logger.info("SVM classifier test completed!")

def test_session_integration():
    """Test session integration."""
    logger.info("Testing session integration...")
    
    from data_handling.data_models import Session
    
    # Create a test session
    session = Session(name="test_session")
    
    # Test retrain logic
    logger.info(f"Initial retrain counter: {session.images_since_retrain}")
    logger.info(f"Should retrain: {session.should_retrain_classifier()}")
    
    # Increment counter
    session.increment_retrain_counter()
    logger.info(f"After increment: {session.images_since_retrain}")
    logger.info(f"Should retrain: {session.should_retrain_classifier()}")
    
    # Reset counter
    session.reset_retrain_counter()
    logger.info(f"After reset: {session.images_since_retrain}")
    
    # Test classifier path
    path = session.get_classifier_path()
    logger.info(f"Classifier path: {path}")
    
    logger.info("Session integration test completed!")

if __name__ == "__main__":
    logger.info("Starting classifier tests...")
    
    try:
        test_feature_extraction()
        test_svm_classifier()
        test_session_integration()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise 