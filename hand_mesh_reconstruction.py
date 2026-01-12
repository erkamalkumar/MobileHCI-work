"""
Real-time Hand Mesh Reconstruction Module
Implements 3D hand mesh reconstruction using MediaPipe Hands solution
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional, List, Tuple


# Hand landmark connections for drawing
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
])


class HandMeshReconstructor:
    """
    Real-time 3D hand mesh reconstruction using MediaPipe.
    Provides full hand mesh with vertices, connections, and texture.
    """
    
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand mesh reconstructor.
        
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
        
        # Define hand mesh connections for visualization
        self.hand_connections = HAND_CONNECTIONS
        
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
        for connection in self.hand_connections:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], color, 2)
        
        return image
        
    def reconstruct_mesh(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Reconstruct 3D hand mesh from input frame.
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            Tuple of (annotated_frame, mesh_data)
            mesh_data contains: landmarks, world_landmarks, handedness
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
        
        mesh_data = None
        
        if results.hand_landmarks and results.hand_world_landmarks:
            mesh_data = {
                'hands': [],
                'count': len(results.hand_landmarks)
            }
            
            for idx, (hand_landmarks, hand_world_landmarks) in enumerate(
                zip(results.hand_landmarks, results.hand_world_landmarks)
            ):
                # Extract handedness (left or right)
                handedness = "Right"
                if results.handedness and idx < len(results.handedness):
                    handedness = results.handedness[idx][0].category_name
                
                # Extract normalized landmarks (2D + depth)
                screen_landmarks = []
                for landmark in hand_landmarks:
                    screen_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # Extract world landmarks (3D in real-world coordinates)
                world_landmarks = []
                for landmark in hand_world_landmarks:
                    world_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                mesh_data['hands'].append({
                    'handedness': handedness,
                    'screen_landmarks': screen_landmarks,
                    'world_landmarks': world_landmarks,
                    'landmark_count': len(screen_landmarks)
                })

                # Draw the hand mesh with connections
                self.draw_landmarks(annotated_frame, screen_landmarks, 
                                  color=(255, 0, 0) if handedness == "Left" else (0, 0, 255))
        
        return annotated_frame, mesh_data
    
    def get_mesh_vertices(self, world_landmarks: List) -> np.ndarray:
        """
        Extract mesh vertices from world landmarks.
        
        Args:
            world_landmarks: List of world coordinate landmarks
            
        Returns:
            numpy array of shape (21, 3) containing 3D vertices in meters
        """
        vertices = np.zeros((21, 3))
        
        for i, landmark in enumerate(world_landmarks):
            vertices[i, 0] = landmark['x']
            vertices[i, 1] = landmark['y']
            vertices[i, 2] = landmark['z']
        
        return vertices
    
    def draw_3d_mesh(self, frame: np.ndarray, mesh_data: dict) -> np.ndarray:
        """
        Draw hand skeleton from landmarks (simplified visualization).
        
        Args:
            frame: Input frame
            mesh_data: Mesh data from reconstruct_mesh
            
        Returns:
            Frame with hand skeleton overlay
        """
        if not mesh_data or 'hands' not in mesh_data:
            return frame
        
        h, w, _ = frame.shape
        
        for hand_info in mesh_data['hands']:
            handedness = hand_info['handedness']
            base_color = (255, 100, 100) if handedness == "Left" else (100, 100, 255)
            
            # Draw MediaPipe landmarks
            landmarks_2d = np.array([
                [int(lm['x'] * w), int(lm['y'] * h)]
                for lm in hand_info['screen_landmarks']
            ], dtype=np.int32)
            
            for lm in landmarks_2d:
                cv2.circle(frame, tuple(lm), 5, (0, 255, 0), -1)
            
            # Draw connections
            for connection in self.hand_connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d):
                    cv2.line(frame, tuple(landmarks_2d[start_idx]), tuple(landmarks_2d[end_idx]), base_color, 2)
            
            # Label
            wrist_px = tuple(landmarks_2d[0])
            cv2.putText(frame, f"{handedness} Hand (21 landmarks)", (wrist_px[0], wrist_px[1] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)
        
        return frame
    
    def release(self):
        """Release resources."""
        self.detector.close()


def main():
    """Demo: Real-time hand mesh reconstruction from webcam."""
    print("Starting real-time hand mesh reconstruction...")
    print("Press 'q' to quit")
    
    # Initialize reconstructor
    reconstructor = HandMeshReconstructor(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Reconstruct mesh
        annotated_frame, mesh_data = reconstructor.reconstruct_mesh(frame)
        
        # Draw enhanced mesh
        if mesh_data:
            annotated_frame = reconstructor.draw_3d_mesh(annotated_frame, mesh_data)
        
        # Display info
        frame_count += 1
        if mesh_data:
            info_text = f"Meshes: {mesh_data['count']} | Landmarks: 21 per hand | Frame: {frame_count}"
        else:
            info_text = f"No hands detected | Frame: {frame_count}"
        
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show FPS
        cv2.putText(annotated_frame, "Real-time Mesh Reconstruction", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Hand Mesh Reconstruction', annotated_frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    reconstructor.release()
    print("Demo completed")


if __name__ == "__main__":
    main()
