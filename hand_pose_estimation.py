"""
Lightweight 3D Hand Pose Estimation Module
Implements real-time hand tracking and 3D pose estimation using MediaPipe
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, Tuple, List


# Hand landmark connections for drawing
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
])


class HandPoseEstimator:
    """
    Lightweight 3D hand pose estimation using MediaPipe Hands.
    Provides real-time hand tracking with 21 3D landmarks per hand.
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand pose estimator.
        
        Args:
            static_image_mode: If True, treats input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        # Create HandLandmarker options
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        
        running_mode = vision.RunningMode.VIDEO if not static_image_mode else vision.RunningMode.IMAGE
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.running_mode = running_mode
        self.frame_timestamp_ms = 0
        
    def draw_landmarks(self, image: np.ndarray, hand_landmarks: List, color=(0, 255, 0)):
        """Draw hand landmarks and connections on image."""
        h, w, _ = image.shape
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for landmark in hand_landmarks:
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
            points.append((x, y))
            # Draw landmark point
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.circle(image, (x, y), 7, (255, 255, 255), 2)
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], color, 2)
        
        return image
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Process a single frame to detect hands and estimate 3D pose.
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            Tuple of (annotated_frame, hand_landmarks_list)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the frame
        if self.running_mode == vision.RunningMode.VIDEO:
            self.frame_timestamp_ms += 33  # ~30 fps
            results = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
        else:
            results = self.detector.detect(mp_image)
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        hand_landmarks_list = []
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Extract 3D coordinates
                landmarks_3d = []
                for landmark in hand_landmarks:
                    landmarks_3d.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,  # Depth relative to wrist
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    })
                
                hand_landmarks_list.append(landmarks_3d)
                
                # Draw landmarks
                self.draw_landmarks(annotated_frame, landmarks_3d)
        
        return annotated_frame, hand_landmarks_list if hand_landmarks_list else None
    
    def get_hand_keypoints_3d(self, hand_landmarks: List, 
                              image_width: int, 
                              image_height: int) -> np.ndarray:
        """
        Convert normalized landmarks to 3D keypoints in pixel coordinates.
        
        Args:
            hand_landmarks: List of normalized hand landmarks
            image_width: Width of the image
            image_height: Height of the image
            
        Returns:
            numpy array of shape (21, 3) containing 3D keypoints
        """
        keypoints = np.zeros((21, 3))
        
        for i, landmark in enumerate(hand_landmarks):
            keypoints[i, 0] = landmark['x'] * image_width
            keypoints[i, 1] = landmark['y'] * image_height
            keypoints[i, 2] = landmark['z'] * image_width  # Scale depth to image width
            
        return keypoints
    
    def release(self):
        """Release resources."""
        self.detector.close()


def main():
    """Demo: Real-time hand pose estimation from webcam."""
    print("Starting lightweight 3D hand pose estimation...")
    print("Press 'q' to quit")
    
    # Initialize estimator
    estimator = HandPoseEstimator(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        annotated_frame, hand_landmarks = estimator.process_frame(frame)
        
        # Display info
        frame_count += 1
        if hand_landmarks:
            info_text = f"Hands detected: {len(hand_landmarks)} | Frame: {frame_count}"
        else:
            info_text = f"No hands detected | Frame: {frame_count}"
        
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Lightweight 3D Hand Pose Estimation', annotated_frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    estimator.release()
    print("Demo completed")


if __name__ == "__main__":
    main()
