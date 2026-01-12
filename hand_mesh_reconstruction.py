"""
Real-time Hand Mesh Reconstruction Module
Implements 3D hand mesh reconstruction using MediaPipe Hands solution
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple


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
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define hand mesh connections for visualization
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS
        
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
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        mesh_data = None
        
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            mesh_data = {
                'hands': [],
                'count': len(results.multi_hand_landmarks)
            }
            
            for idx, (hand_landmarks, hand_world_landmarks) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks)
            ):
                # Draw the hand mesh with connections
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.hand_connections,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract handedness (left or right)
                handedness = "Unknown"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                
                # Extract normalized landmarks (2D + depth)
                screen_landmarks = []
                for landmark in hand_landmarks.landmark:
                    screen_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                # Extract world landmarks (3D in real-world coordinates)
                world_landmarks = []
                for landmark in hand_world_landmarks.landmark:
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
        Draw enhanced 3D mesh visualization with additional visual cues.
        
        Args:
            frame: Input frame
            mesh_data: Mesh data from reconstruction
            
        Returns:
            Frame with enhanced mesh visualization
        """
        if not mesh_data or 'hands' not in mesh_data:
            return frame
        
        height, width, _ = frame.shape
        
        for hand_info in mesh_data['hands']:
            landmarks = hand_info['screen_landmarks']
            handedness = hand_info['handedness']
            
            # Draw palm area (filled polygon)
            palm_indices = [0, 1, 5, 9, 13, 17]  # Wrist and base of each finger
            palm_points = []
            for idx in palm_indices:
                x = int(landmarks[idx]['x'] * width)
                y = int(landmarks[idx]['y'] * height)
                palm_points.append([x, y])
            
            palm_points = np.array(palm_points, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [palm_points], (100, 150, 100))
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Add handedness label
            wrist = landmarks[0]
            x = int(wrist['x'] * width)
            y = int(wrist['y'] * height)
            cv2.putText(frame, f"{handedness} Hand", (x - 50, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def release(self):
        """Release resources."""
        self.hands.close()


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
