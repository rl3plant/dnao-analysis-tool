import os
import numpy as np
import logging
from src.client.pipeline.features_shared import extract_features
from .model import AnnotationClassifier
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AnnotationTrainer:
    def __init__(self, session_path=None):
        """
        Initialize the trainer.
        
        Args:
            session_path: Path to the current session directory
        """
        self.session_path = session_path
        self.classifier = None
        
        # Initialize model path
        if session_path:
            self.model_path = os.path.join(session_path, "models", "annotation_model.pkl")
            
            # Load existing model if available
            if os.path.exists(self.model_path):
                self.classifier = AnnotationClassifier(self.model_path)
            else:
                self.classifier = AnnotationClassifier()
                # Apply warm start to new model
                self.warm_start()
        else:
            self.classifier = AnnotationClassifier()
            # Apply warm start to new model
            self.warm_start()
    
    def warm_start(self):
        """
        Initialize the model with default training data.
        This provides a basic starting point for classification.
        """
        logger.info("Applying warm start to classifier")
        
        # Create synthetic training data for the three classes
        # We'll create many more samples for intact DNAOs since they're much more common
        
        # For intact DNAOs (class 0) - create many more samples
        intact_features_list = []
        for _ in range(20):  # Create 20 intact samples (increased from 10)
            intact_features = [
                np.random.normal(0.8, 0.1, 128),  # SIFT-like features (128 dimensions)
                np.random.normal(0.7, 0.1, 20)    # Fourier-like features (20 dimensions)
            ]
            intact_features = np.concatenate(intact_features)
            intact_features_list.append(intact_features)
        
        # For damaged DNAOs (class 1) - fewer samples
        damaged_features_list = []
        for _ in range(2):  # Create 2 damaged samples (reduced from 3)
            damaged_features = [
                np.random.normal(0.3, 0.1, 128),  # SIFT-like features (128 dimensions)
                np.random.normal(0.4, 0.1, 20)    # Fourier-like features (20 dimensions)
            ]
            damaged_features = np.concatenate(damaged_features)
            damaged_features_list.append(damaged_features)
        
        # For invalid DNAOs (class 2) - fewer samples
        invalid_features_list = []
        for _ in range(1):  # Create 1 invalid sample (reduced from 2)
            invalid_features = [
                np.random.normal(0.5, 0.2, 128),  # SIFT-like features (128 dimensions)
                np.random.normal(0.5, 0.2, 20)    # Fourier-like features (20 dimensions)
            ]
            invalid_features = np.concatenate(invalid_features)
            invalid_features_list.append(invalid_features)
        
        # Combine all features
        X = np.array(intact_features_list + damaged_features_list + invalid_features_list)
        
        # Create labels with more intact samples
        y = np.array(
            ["intact"] * len(intact_features_list) + 
            ["damaged"] * len(damaged_features_list) + 
            ["invalid"] * len(invalid_features_list)
        )
        
        # Set class weights to reflect prior knowledge
        # This will make the model much more likely to predict "intact" when uncertain
        self.classifier.model.set_params(class_weight={
            "intact": 1.0,      # Normal weight for intact
            "damaged": 3.0,     # Higher weight for damaged (increased from 2.0)
            "invalid": 4.0      # Even higher weight for invalid (increased from 2.5)
        })
        
        # Train the model
        self.classifier.update(X, y)
        logger.info("Warm start applied to classifier with stronger bias toward intact predictions")
    
    def process_annotations(self, image, contours, labels=None):
        """
        Process a set of annotations, either for prediction or training.
        
        Args:
            image: numpy array of the image
            contours: list of contour points
            labels: optional list of labels for training
            
        Returns:
            If labels is None: list of (prediction, probability) tuples
            If labels is provided: None
        """
        features_list = []
        
        # Extract features for each contour with progress bar
        print("Extracting features from annotations...")
        for contour in tqdm(contours, desc="Feature extraction"):
            features = extract_features(image, contour)
            features_list.append(features)
        
        if labels is None:
            # Prediction mode
            predictions = []
            print("Making predictions...")
            for features in tqdm(features_list, desc="Prediction"):
                pred, prob = self.classifier.predict(features)
                predictions.append((pred, prob))
            return predictions
        else:
            # Training mode
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels)
            
            # Train the model
            print("Training model...")
            self.classifier.update(X, y)
            logger.info(f"Trained model on {len(X)} samples")
            print("\n" + "="*50)
            print(f"✅ CLASSIFIER RETRAINING COMPLETE! Processed {len(X)} samples.")
            print("="*50 + "\n")
            return None
    
    def save_model(self):
        """
        Save the current model state.
        """
        if not self.session_path:
            return
            
        # Ensure models directory exists
        models_dir = os.path.join(self.session_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        print(f"Saving model to {self.model_path}...")
        self.classifier.save_model(self.model_path)
        logger.info(f"Saved model to {self.model_path}")
    
    def load_model(self, session_path):
        """
        Load a model state for a specific session.
        
        Args:
            session_path: path to the session directory
        """
        self.session_path = session_path
        self.model_path = os.path.join(session_path, "models", "annotation_model.pkl")
        
        # Load model
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.classifier = AnnotationClassifier(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        else:
            print("Initializing new model...")
            self.classifier = AnnotationClassifier()
            # Apply warm start to new model
            self.warm_start()
            logger.info("Initialized new model with warm start") 