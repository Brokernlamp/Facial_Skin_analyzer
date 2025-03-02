import cv2
import numpy as np

# Import all modules
from face_detection import FaceDetector
from facial_landmarks import FacialLandmarks
from feature_extraction import FeatureExtractor
from skin_classifier import SkinClassifier
from acne_detection import AcneDetector
from skin_analysis import SkinAnalyzer
from utils import resize_frame, add_text_with_background

def main():
    # Initialize all components
    face_detector = FaceDetector(min_detection_confidence=0.7)
    facial_landmarks = FacialLandmarks(max_num_faces=1)
    feature_extractor = FeatureExtractor()
    skin_classifier = SkinClassifier(model_path="skin_condition_classifier.joblib")
    acne_detector = AcneDetector(model_path="acne_detection_yolo.pt", facial_landmarks=facial_landmarks)
    skin_analyzer = SkinAnalyzer()
    
    # Check for required models
    if not hasattr(acne_detector, 'model') or acne_detector.model is None:
        print("ERROR: YOLO model not loaded. Acne detection will not be available.")
        print("Make sure 'acne_detection_yolo.pt' is in the same directory as this script")
        return
    
    # Start camera
    cap = cv2.VideoCapture(0)  # Change to 0 if this is your default camera
    
    if not cap.isOpened():
        print("ERROR: Could not open camera. Try changing the camera index.")
        return
        
    print("Skin Analysis started. Press 'q' to quit.")
    print("Press 'd' to toggle debug visualization.")
    
    debug_mode = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Retrying...")
            continue
            
        # Downscale the frame for faster processing
        frame = resize_frame(frame, width=640)
        
        # Step 1: Detect face
        face_region, face_coords = face_detector.detect_face(frame)
        
        # Draw face box
        if face_coords:
            frame = face_detector.draw_face_box(frame, face_coords)
        
        if face_region is not None:
            # Step 2: Extract features for classification
            features = feature_extractor.extract_features(face_region)
            
            # Step 3: Classify skin condition
            condition, condition_confidence = skin_classifier.classify(features)
            
            # Step 4: Detect acne
            acne_detections = acne_detector.detect_acne(face_region)
            
            # Step 5: Draw acne detections on the frame
            frame = acne_detector.draw_detections(face_region, acne_detections, face_coords, frame)
            
            # Step 6: Analyze skin (hydration, oiliness)
            skin_properties = skin_analyzer.analyze_skin(face_region)
            
            # Step 7: Determine outbreak status
            outbreak_count = len(acne_detections)
            outbreak_status = skin_analyzer.classify_outbreak(outbreak_count)
            
            # Calculate average confidence of detections
            avg_confidence = np.mean([conf for _, _, _, conf, _ in acne_detections]) if acne_detections else 0
            
            # Prepare result text
            results = f"Hydration: {skin_properties['hydration']} | Oiliness: {skin_properties['oiliness']} | "
            results += f"Acne: {outbreak_status} ({outbreak_count})"
            
            if outbreak_count > 0:
                results += f" | Condition: {condition}"
            
            results += f" | Avg. Confidence: {avg_confidence:.1f}%"
            
            # Show Analysis on Screen
            frame = add_text_with_background(frame, results, (20, 30), 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                             (0, 0, 0, 128), 5)
            
            # Debug mode - show facial landmarks
            if debug_mode:
                landmark_face = facial_landmarks.draw_landmarks(face_region)
                if landmark_face is not None:
                    x, y, w, h = face_coords
                    frame[y:y+landmark_face.shape[0], x:x+landmark_face.shape[1]] = landmark_face
        else:
            # No face detected
            frame = add_text_with_background(frame, "No face detected", (20, 30), 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                             (0, 0, 0, 128), 5)
        
        cv2.imshow("Skin Analysis with YOLOv8", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()