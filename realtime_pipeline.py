"""
Real-time Vision Pipeline
Combines 3D hand pose estimation and mesh reconstruction in a unified pipeline
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict
from hand_pose_estimation import HandPoseEstimator
from hand_mesh_reconstruction import HandMeshReconstructor


class VisionPipeline:
    """
    Unified pipeline for real-time hand tracking, pose estimation, and mesh reconstruction.
    Optimized for low latency and high performance.
    """
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the vision pipeline.
        
        Args:
            max_num_hands: Maximum number of hands to track
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.pose_estimator = HandPoseEstimator(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.mesh_reconstructor = HandMeshReconstructor(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.fps_history = []
        self.max_fps_samples = 30
        
    def process_frame(self, frame: np.ndarray, 
                      show_pose: bool = True,
                      show_mesh: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process a frame through the complete pipeline.
        
        Args:
            frame: Input BGR frame
            show_pose: Whether to show pose landmarks
            show_mesh: Whether to show mesh reconstruction
            
        Returns:
            Tuple of (processed_frame, metadata)
        """
        start_time = time.time()
        
        metadata = {
            'hands_detected': 0,
            'landmarks': None,
            'mesh_data': None,
            'fps': 0,
            'latency_ms': 0
        }
        
        if show_mesh:
            # Use mesh reconstructor (includes landmarks)
            processed_frame, mesh_data = self.mesh_reconstructor.reconstruct_mesh(frame)
            if mesh_data:
                processed_frame = self.mesh_reconstructor.draw_3d_mesh(processed_frame, mesh_data)
                metadata['hands_detected'] = mesh_data['count']
                metadata['mesh_data'] = mesh_data
        elif show_pose:
            # Use pose estimator only
            processed_frame, landmarks = self.pose_estimator.process_frame(frame)
            if landmarks:
                metadata['hands_detected'] = len(landmarks)
                metadata['landmarks'] = landmarks
        else:
            processed_frame = frame.copy()
        
        # Calculate performance metrics
        end_time = time.time()
        elapsed = end_time - start_time
        latency = elapsed * 1000  # Convert to ms
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_fps_samples:
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        metadata['fps'] = avg_fps
        metadata['latency_ms'] = latency
        
        return processed_frame, metadata
    
    def draw_performance_overlay(self, frame: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Draw performance metrics overlay on frame.
        
        Args:
            frame: Input frame
            metadata: Processing metadata
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (5, 5), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw metrics
        y_offset = 25
        cv2.putText(frame, f"FPS: {metadata['fps']:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Latency: {metadata['latency_ms']:.1f} ms", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(frame, f"Hands: {metadata['hands_detected']}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def release(self):
        """Release all resources."""
        self.pose_estimator.release()
        self.mesh_reconstructor.release()


def main():
    """Demo: Real-time vision pipeline."""
    print("=" * 60)
    print("Real-time 3D Hand Pose Estimation & Mesh Reconstruction")
    print("=" * 60)
    print("\nControls:")
    print("  'p' - Toggle pose landmarks")
    print("  'm' - Toggle mesh reconstruction")
    print("  'o' - Toggle performance overlay")
    print("  'q' - Quit")
    print("\n" + "=" * 60)
    
    # Initialize pipeline
    pipeline = VisionPipeline(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # UI state
    show_pose = False
    show_mesh = True
    show_overlay = True
    
    print("\nStarting camera feed...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process through pipeline
        processed_frame, metadata = pipeline.process_frame(
            frame, 
            show_pose=show_pose, 
            show_mesh=show_mesh
        )
        
        # Draw performance overlay
        if show_overlay:
            processed_frame = pipeline.draw_performance_overlay(processed_frame, metadata)
        
        # Draw mode indicators
        mode_text = []
        if show_mesh:
            mode_text.append("MESH")
        if show_pose:
            mode_text.append("POSE")
        
        if mode_text:
            cv2.putText(processed_frame, f"Mode: {' + '.join(mode_text)}", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Vision-KAF: Hand Tracking Pipeline', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_pose = not show_pose
            print(f"Pose landmarks: {'ON' if show_pose else 'OFF'}")
        elif key == ord('m'):
            show_mesh = not show_mesh
            print(f"Mesh reconstruction: {'ON' if show_mesh else 'OFF'}")
        elif key == ord('o'):
            show_overlay = not show_overlay
            print(f"Performance overlay: {'ON' if show_overlay else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pipeline.release()
    print("\nPipeline terminated successfully")


if __name__ == "__main__":
    main()
