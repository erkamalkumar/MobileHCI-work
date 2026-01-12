"""
Flask Web Application for Real-time Hand Tracking
Provides web interface for 3D hand pose estimation and mesh reconstruction
"""

from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import json
import threading
from realtime_pipeline import VisionPipeline

app = Flask(__name__)
CORS(app)

# Thread-safe global state
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pipeline = None
        self.camera = None
        self.latest_metadata = {
            'fps': 0,
            'latency_ms': 0,
            'hands_detected': 0,
            'frame_size': [FRAME_WIDTH, FRAME_HEIGHT]
        }

state = AppState()

# Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def init_camera():
    """Initialize the camera with thread safety."""
    with state.lock:
        if state.camera is None:
            state.camera = cv2.VideoCapture(CAMERA_INDEX)
            state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            state.camera.set(cv2.CAP_PROP_FPS, 30)
        return state.camera


def init_pipeline():
    """Initialize the vision pipeline with thread safety."""
    with state.lock:
        if state.pipeline is None:
            state.pipeline = VisionPipeline(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return state.pipeline


def generate_frames():
    """Generate video frames for streaming."""
    cam = init_camera()
    pipe = init_pipeline()
    
    while True:
        with state.lock:
            if not cam.isOpened():
                break
            success, frame = cam.read()
        
        if not success:
            break
        
        # Process frame through pipeline
        processed_frame, metadata = pipe.process_frame(frame, show_pose=False, show_mesh=True)
        
        # Update cached metadata
        with state.lock:
            state.latest_metadata = metadata
        
        # Draw overlay
        processed_frame = pipe.draw_performance_overlay(processed_frame, metadata)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('demo.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def status():
    """Get system status."""
    with state.lock:
        return jsonify({
            'status': 'running',
            'pipeline': 'initialized' if state.pipeline else 'not initialized',
            'camera': 'active' if state.camera and state.camera.isOpened() else 'inactive'
        })


@app.route('/api/stats')
def stats():
    """Get current processing statistics from cached metadata."""
    with state.lock:
        return jsonify(state.latest_metadata)


def cleanup():
    """Cleanup resources."""
    with state.lock:
        if state.camera:
            state.camera.release()
        if state.pipeline:
            state.pipeline.release()


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Vision-KAF Web Application")
        print("=" * 60)
        print("\nStarting server on http://localhost:5000")
        print("Press Ctrl+C to stop\n")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()
