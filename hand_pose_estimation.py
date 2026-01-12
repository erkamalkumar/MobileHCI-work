"""
Lightweight 3D Hand Pose Estimation Module
Implements real-time hand tracking and 3D pose estimation using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


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
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
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
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract 3D coordinates
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    landmarks_3d.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,  # Depth relative to wrist
                        'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    })
                
                hand_landmarks_list.append(landmarks_3d)
        
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
        self.hands.close()


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
