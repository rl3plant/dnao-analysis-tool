import numpy as np
from sklearn.svm import SVC
import joblib
import logging

logger = logging.getLogger(__name__)

class AnnotationClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Optional path to load a pre-trained model
        """
        # Use balanced class weights to handle imbalanced data
        self.model = SVC(probability=True, class_weight='balanced')
        if model_path:
            self.load_model(model_path)
    
    def predict(self, features):
        """
        Predict the class of a feature vector.
        
        Args:
            features: numpy array of features
            
        Returns:
            predicted class and probability
        """
        # Check if model is trained
        if not hasattr(self.model, 'classes_'):
            return "intact", 0.5  # Default prediction if model not trained
            
        # Check if features are empty or invalid
        if features is None or len(features) == 0:
            return "intact", 0.5  # Default prediction for empty features
            
        # Ensure features is a 2D array
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Make prediction
        try:
            # Get probability for each class
            probas = self.model.predict_proba(features)[0]
            
            # Get the predicted class
            pred = self.model.predict(features)[0]
            
            # Get the probability of the predicted class
            prob = np.max(probas)
            
            # If the model is uncertain (low probability) and "intact" is not the prediction,
            # but "intact" has a reasonable probability, prefer "intact"
            if prob < 0.6:  # Threshold for uncertainty
                # Find the index of "intact" in the classes
                intact_idx = np.where(self.model.classes_ == "intact")[0]
                if len(intact_idx) > 0:
                    intact_idx = intact_idx[0]
                    intact_prob = probas[intact_idx]
                    
                    # If "intact" has a reasonable probability, prefer it
                    if intact_prob > 0.3:  # Threshold for reasonable probability
                        pred = "intact"
                        prob = intact_prob
                        logger.info("Preferring 'intact' due to model uncertainty")
            
            return pred, prob
        except Exception as e:
            # Log the error and return a default prediction
            logger.error(f"Error during prediction: {str(e)}")
            return "intact", 0.5
    
    def update(self, features_list, labels):
        """
        Update the model with new training data.
        
        Args:
            features_list: list of feature vectors
            labels: list of corresponding labels
        """
        X = np.array(features_list)
        y = np.array(labels)
        
        # If model is not trained yet, fit it
        if not hasattr(self.model, 'classes_'):
            self.model.fit(X, y)
        else:
            # Otherwise, update with new data
            # Note: This is a simple implementation. For production,
            # you might want to implement proper online learning
            self.model.fit(X, y)
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path: path to save the model
        """
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """
        Load a model from disk.
        
        Args:
            path: path to load the model from
        """
        self.model = joblib.load(path) 