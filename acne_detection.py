import cv2
import numpy as np
from ultralytics import YOLO

class AcneDetector:
    def __init__(self, model_path="acne_detection_yolo.pt", facial_landmarks=None):
        """
        Initialize YOLO model for acne detection
        
        Args:
            model_path: Path to the trained YOLO model
            facial_landmarks: FacialLandmarks instance for eye/nose region detection
        """
        self.model = None
        self.model_path = model_path
        self.facial_landmarks = facial_landmarks
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model for acne detection"""
        try:
            self.model = YOLO(self.model_path)
            print("YOLOv8 acne detection model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_acne(self, face_region):
        """
        Detect acne using the YOLOv8 model, filtering out false positives
        
        Args:
            face_region: BGR image of the face
            
        Returns:
            list: List of acne detections as (x, y, radius, confidence, label)
        """
        if self.model is None or face_region is None or face_region.size == 0:
            return []
        
        face_height, face_width = face_region.shape[:2]
        
        # Get facial landmarks if available
        landmarks = None
        if self.facial_landmarks:
            landmarks = self.facial_landmarks.get_landmarks(face_region)
        
        # Run YOLOv8 inference
        results = self.model(face_region, conf=0.35)  # Increased confidence threshold
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get box coordinates
                confidence = float(box.conf[0].cpu().numpy())  # Get confidence
                
                # Calculate center and radius for circle drawing
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                radius = int(max(x2 - x1, y2 - y1) / 2)
                
                # Filter out detections in eye and nose regions if landmarks available
                skip_detection = False
                if self.facial_landmarks and landmarks:
                    skip_detection = self.facial_landmarks.is_in_eye_or_nose_region(
                        center_x, center_y, landmarks, face_width, face_height
                    )
                
                if not skip_detection:
                    # Additional size-based filtering - acne should be relatively small
                    if radius < face_width * 0.15:  # Acne shouldn't be larger than 15% of face width
                        detections.append((center_x, center_y, radius, confidence * 100, "Acne"))
        
        return detections
    
    def draw_detections(self, face_region, detections, face_coords=None, frame=None):
        """
        Draw acne detections on the image
        
        Args:
            face_region: BGR image of the face
            detections: List of detections from detect_acne()
            face_coords: (x, y, width, height) of face in original frame
            frame: Original frame (optional, if provided will draw on it instead)
            
        Returns:
            image with detections drawn
        """
        if frame is not None and face_coords is not None:
            # Draw on the original frame
            x, y, _, _ = face_coords
            image = frame
            offset_x, offset_y = x, y
        else:
            # Draw on the face region
            image = face_region.copy()
            offset_x, offset_y = 0, 0
        
        for ox, oy, r, conf, condition in detections:
            # Adjusting coordinates to the target image
            center = (offset_x + ox, offset_y + oy)
            # Color based on confidence: more red = higher confidence
            color = (0, int(255 * (1 - conf/100)), int(255 * (conf/100)))
            cv2.circle(image, center, r, color, 2)  # Draw circle
            
            # Show condition and confidence
            text = f"{condition}: {int(conf)}%"
            cv2.putText(image, text, (center[0]+r, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        return image