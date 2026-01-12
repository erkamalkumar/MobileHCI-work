# RehabNet - Vision-KAF

Real-Time Fusion of Kinematic and Affective Digital Biomarkers for Hand Motor Assessment via RGB Camera

## 🚀 Features

This repository implements **lightweight 3D hand pose estimation** and **real-time mesh reconstruction** for rehabilitation and hand motor assessment applications.

### Key Capabilities

- ✅ **Lightweight 3D Hand Pose Estimation**: 21 landmark detection per hand with depth information
- ✅ **Real-time Mesh Reconstruction**: Full 3D hand mesh with world coordinates
- ✅ **Low Latency Performance**: Typically <50ms processing time
- ✅ **Multi-Hand Support**: Simultaneous tracking of up to 2 hands
- ✅ **Web-Based Interface**: Browser-accessible real-time visualization
- ✅ **Standalone Applications**: Command-line demos for testing

## 📋 Requirements

- Python 3.8 or higher
- Webcam or video input device
- Modern web browser (for web demo)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/erkamalkumar/Vision-KAF.git
cd Vision-KAF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Option 1: Real-time Pipeline (Recommended)

Run the unified pipeline with interactive controls:

```bash
python realtime_pipeline.py
```

**Controls:**
- `p` - Toggle pose landmarks
- `m` - Toggle mesh reconstruction
- `o` - Toggle performance overlay
- `q` - Quit

### Option 2: Web Application

Start the web server:

```bash
python web_app.py
```

Then open your browser to: `http://localhost:5000`

### Option 3: Individual Modules

**Hand Pose Estimation only:**
```bash
python hand_pose_estimation.py
```

**Hand Mesh Reconstruction only:**
```bash
python hand_mesh_reconstruction.py
```

## 🏗️ Architecture

The system consists of three main modules:

1. **Hand Pose Estimation** (`hand_pose_estimation.py`)
   - Lightweight 3D keypoint detection
   - 21 landmarks per hand
   - Normalized coordinates + depth

2. **Mesh Reconstruction** (`hand_mesh_reconstruction.py`)
   - Full 3D hand mesh generation
   - World coordinates (meters)
   - Left/right hand classification

3. **Real-time Pipeline** (`realtime_pipeline.py`)
   - Unified processing pipeline
   - Performance monitoring
   - FPS and latency tracking

## 📊 Technical Specifications

- **Model**: MediaPipe Hands (Google Research)
- **Landmarks**: 21 3D keypoints per hand
- **Coordinate Systems**: 
  - Screen coordinates (normalized 0-1)
  - World coordinates (meters, origin at hand center)
- **Detection Pipeline**: Two-stage architecture
  - Stage 1: SSD-based hand detector
  - Stage 2: CNN-based landmark regression
- **Backend**: Flask + OpenCV + MediaPipe
- **Performance**: 30+ FPS on standard hardware

## 🔬 Research

This work is part of research on real-time fusion of kinematic and affective digital biomarkers for hand motor assessment using RGB cameras.

**Institution**: Indian Institute of Technology Guwahati, India

## 📄 License

Please refer to the repository license file.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaboration opportunities, please contact the authors through the repository.