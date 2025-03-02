import cv2
import numpy as np
import mediapipe as mp

class FacialLandmarks:
    def __init__(self, max_num_faces=1):
        """Initialize MediaPipe face mesh for facial landmarks"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define key points for eyes and nose (MediaPipe indices)
        self.left_eye_indices = [33, 133, 160, 159, 158, 144, 153, 157]
        self.right_eye_indices = [362, 263, 387, 386, 385, 373, 380, 384]
        self.nose_indices = [1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 98]
    
    def get_landmarks(self, face_region):
        """
        Get facial landmarks using MediaPipe Face Mesh
        
        Args:
            face_region: BGR image containing a face
            
        Returns:
            facial landmarks or None if no landmarks detected
        """
        if face_region is None or face_region.size == 0:
            return None
        
        # Convert to RGB
        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        
        # Process the image to find face landmarks
        results = self.face_mesh.process(face_region_rgb)
        
        # Check if landmarks detected
        if not results.multi_face_landmarks:
            return None
        
        # Return the first face's landmarks
        return results.multi_face_landmarks[0]
    
    def is_in_eye_or_nose_region(self, x, y, landmarks, face_width, face_height):
        """
        Check if a point is in eye or nose region using facial landmarks
        
        Args:
            x, y: Point coordinates
            landmarks: Facial landmarks from MediaPipe
            face_width, face_height: Dimensions of the face region
            
        Returns:
            bool: True if point is in eye or nose region
        """
        if landmarks is None:
            return False
        
        # Extract landmark coordinates and convert to pixel space
        left_eye_points = [(int(landmarks.landmark[idx].x * face_width), 
                           int(landmarks.landmark[idx].y * face_height)) 
                           for idx in self.left_eye_indices]
        
        right_eye_points = [(int(landmarks.landmark[idx].x * face_width), 
                            int(landmarks.landmark[idx].y * face_height)) 
                            for idx in self.right_eye_indices]
        
        nose_points = [(int(landmarks.landmark[idx].x * face_width), 
                       int(landmarks.landmark[idx].y * face_height)) 
                       for idx in self.nose_indices]
        
        # Create masks for the regions
        mask = np.zeros((face_height, face_width), dtype=np.uint8)
        
        # Fill eye and nose regions
        cv2.fillPoly(mask, [np.array(left_eye_points)], 255)
        cv2.fillPoly(mask, [np.array(right_eye_points)], 255)
        cv2.fillPoly(mask, [np.array(nose_points)], 255)
        
        # Add padding to these regions (expand by 10 pixels)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Check if the point is in any of these regions
        if 0 <= y < face_height and 0 <= x < face_width:
            return mask[y, x] > 0
        
        return False
    
    def draw_landmarks(self, face_region):
        """Draw facial landmarks on the face region for visualization"""
        if face_region is None or face_region.size == 0:
            return face_region
        
        landmarks = self.get_landmarks(face_region)
        if landmarks:
            face_region_copy = face_region.copy()
            self.mp_drawing.draw_landmarks(
                image=face_region_copy,
                landmark_list=landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            return face_region_copy
        
        return face_region