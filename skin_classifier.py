import os
import joblib

class SkinClassifier:
    def __init__(self, model_path="skin_condition_classifier.joblib"):
        """
        Initialize the skin condition classifier
        
        Args:
            model_path: Path to the trained classifier model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Unknown']
        self.load_model()
    
    def load_model(self):
        """Load the trained classifier model"""
        if not os.path.exists(self.model_path):
            print(f"WARNING: Classifier not found at {self.model_path}")
            print("Classification features will be limited.")
            return False
        
        try:
            classifier_info = joblib.load(self.model_path)
            self.model = classifier_info['model']
            self.class_names = classifier_info['class_names']
            print(f"Model loaded successfully with {len(self.class_names)} classes:")
            print(", ".join(self.class_names))
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return False
    
    def classify(self, features):
        """
        Classify skin condition based on extracted features
        
        Args:
            features: Features extracted from skin image
            
        Returns:
            tuple: (condition, confidence)
        """
        if self.model is None or features is None:
            return "Unknown", 0
        
        # Predict using classifier
        condition = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])[0]
        condition_idx = list(self.model.classes_).index(condition)
        confidence = proba[condition_idx] * 100
        
        return condition, confidence