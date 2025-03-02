import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_detection_confidence=0.7):
        """Initialize MediaPipe face detection"""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_face(self, frame):
        """
        Detect face in the given frame
        
        Args:
            frame: BGR image frame
            
        Returns:
            tuple: (face_region, face_coordinates) where face_coordinates is (x, y, width, height)
                  or (None, None) if no face is detected
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect Face
        results = self.face_detection.process(frame_rgb)
        
        h, w = frame.shape[:2]
        if results.detections:
            # Get the first face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Extract face region
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + width), min(h, y + height)
            face_region = frame[y1:y2, x1:x2]
            
            # Return face region and coordinates
            return face_region, (x, y, width, height)
        
        return None, None
    
    def draw_face_box(self, frame, face_coords):
        """Draw bounding box around the face"""
        if face_coords:
            x, y, width, height = face_coords
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        return frame