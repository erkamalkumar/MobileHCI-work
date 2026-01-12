"""
Quick Start Example for Vision-KAF
Demonstrates basic usage of the hand tracking modules
"""

import cv2
from hand_pose_estimation import HandPoseEstimator
from hand_mesh_reconstruction import HandMeshReconstructor


def example_pose_estimation():
    """Example: Simple hand pose estimation."""
    print("\n=== Example 1: Hand Pose Estimation ===")
    print("This example shows basic 3D hand landmark detection.")
    print("Press 'q' to move to next example\n")
    
    estimator = HandPoseEstimator(max_num_hands=2)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, landmarks = estimator.process_frame(frame)
        
        # Display info
        if landmarks:
            cv2.putText(annotated_frame, f"Detected: {len(landmarks)} hand(s)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Example 1: Hand Pose Estimation', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    estimator.release()


def example_mesh_reconstruction():
    """Example: Hand mesh reconstruction."""
    print("\n=== Example 2: Hand Mesh Reconstruction ===")
    print("This example shows 3D hand mesh with world coordinates.")
    print("Press 'q' to finish\n")
    
    reconstructor = HandMeshReconstructor(max_num_hands=2)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Reconstruct mesh
        annotated_frame, mesh_data = reconstructor.reconstruct_mesh(frame)
        
        # Enhanced visualization
        if mesh_data:
            annotated_frame = reconstructor.draw_3d_mesh(annotated_frame, mesh_data)
            cv2.putText(annotated_frame, f"Meshes: {mesh_data['count']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Example 2: Hand Mesh Reconstruction', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    reconstructor.release()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Vision-KAF: Quick Start Examples")
    print("=" * 60)
    
    # Example 1: Pose estimation
    example_pose_estimation()
    
    # Example 2: Mesh reconstruction
    example_mesh_reconstruction()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("For more features, try: python realtime_pipeline.py")
    print("For web interface, try: python web_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
