# Stereo Vision Depth Camera

Real-time depth estimation using a dual-camera stereo vision system — from raw frames to 3D scene understanding.


---

##  Overview

This project implements a real-time stereo vision pipeline for depth estimation using a synchronized dual-camera setup. It covers the complete workflow — from camera calibration and stereo rectification to disparity computation and depth visualization — enabling 3D scene understanding.

Depth is computed using:

```
depth = (focal_length × baseline) / disparity
```

---

## Features

* Dual-camera stereo setup for synchronized real-time capture
* Camera calibration (intrinsics and extrinsics)
* Stereo rectification for epipolar alignment
* Disparity computation using StereoSGBM
* Depth estimation from disparity
* Real-time visualization with FPS display

---

##  Tech Stack

| Tool                          | Purpose                               |
| ----------------------------- | ------------------------------------- |
| Python 3.8+                   | Core language                         |
| OpenCV (with contrib modules) | Calibration, rectification, disparity |
| NumPy                         | Matrix operations                     |

---

##  Project Structure

```
stereo_vision/
│
├── calib_left/
├── calib_right/
├── stereo_left/
├── stereo_right/
│
├── camera_index.py
├── step1_capture_left.py
├── step1b_capture_right.py
├── step2_calibrate_individual.py
├── step3_stereo_capture.py
├── step4_stereo_calibrate.py
├── step5_rectify_preview.py
├── step6_depth_full.py
├── step7_depth_map.py
└── README.md
```

---

##  Installation

```bash
pip install opencv-contrib-python numpy
```

---

##  Usage

### Step 0 — Detect Camera Indices

```bash
python camera_index.py
```

---

### Phase 1 — Individual Calibration

```bash
python step1_capture_left.py
python step1b_capture_right.py
python step2_calibrate_individual.py
```

---

### Phase 2 — Stereo Calibration

```bash
python step3_stereo_capture.py
python step4_stereo_calibrate.py
```

---

### Phase 3 — Depth Estimation

```bash
python step5_rectify_preview.py
python step6_depth_full.py
python step7_depth_map.py
```

---

##  Pipeline

```
Dual Camera Capture
        ↓
Individual Calibration (Intrinsics)
        ↓
Stereo Calibration (R, T)
        ↓
Rectification
        ↓
Disparity (StereoSGBM)
        ↓
Depth Estimation
        ↓
Real-time Visualization
```

---

##  Configuration

| Parameter  | Value      |
| ---------- | ---------- |
| Resolution | 640 × 480  |
| Algorithm  | StereoSGBM |

Performance (FPS) depends on system and camera setup.

---

##  Notes

* Good calibration is critical for accurate depth
* Use varied angles and distances during calibration
* Ensure consistent lighting for better results
* Baseline distance affects depth accuracy

---

##  Future Work

* Improve depth in low-texture regions
* Integrate object detection for depth-aware perception
* Optimize for higher FPS and resolution
* Add point cloud generation


