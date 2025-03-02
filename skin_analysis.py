import cv2
import numpy as np

class SkinAnalyzer:
    def __init__(self):
        """Initialize skin analyzer for properties like hydration and oiliness"""
        pass
    
    def analyze_skin(self, face_region):
        """
        Analyze skin properties
        
        Args:
            face_region: BGR image of the face
            
        Returns:
            dict: Dictionary with skin analysis results
        """
        if face_region is None or face_region.size == 0:
            return {
                "hydration": "Unknown",
                "oiliness": "Unknown"
            }
        
        # Convert to grayscale for brightness analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness for hydration assessment
        brightness = np.mean(gray_face)
        hydration = "Dry Skin" if brightness < 100 else "Well-Hydrated"
        
        # Assess oiliness using image difference method
        blurred = cv2.GaussianBlur(face_region, (15, 15), 0)
        diff = cv2.absdiff(face_region, blurred)
        oil_score = np.mean(diff)
        oiliness = "Oily Skin" if oil_score > 20 else "Normal Skin"
        
        return {
            "hydration": hydration,
            "oiliness": oiliness,
            "brightness": brightness,
            "oil_score": oil_score
        }
    
    def classify_outbreak(self, acne_count):
        """
        Classify the severity of acne outbreak
        
        Args:
            acne_count: Number of acne detections
            
        Returns:
            str: Outbreak status description
        """
        if acne_count == 0:
            return "Clear Skin"
        elif acne_count <= 2:
            return "Minor Breakout"
        elif acne_count <= 5:
            return "Moderate Breakout"
        else:
            return "Severe Breakout"