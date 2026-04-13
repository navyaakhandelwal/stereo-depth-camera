# Stereo Vision Project — Run Order & Instructions

## Folder Structure

```
stereo_vision/
├── calib_left/          ← auto-filled by step1_capture_left.py
├── calib_right/         ← auto-filled by step1b_capture_right.py
├── stereo_left/         ← auto-filled by step3_stereo_capture.py
├── stereo_right/        ← auto-filled by step3_stereo_capture.py
│
├── step1_capture_left.py        ← Capture left camera calibration images
├── step1b_capture_right.py      ← Capture right camera calibration images
├── step2_calibrate_individual.py← Compute K and D for both cameras
├── step3_stereo_capture.py      ← Capture synchronized stereo pairs
├── step4_stereo_calibrate.py    ← Compute R and T (stereo extrinsics)
├── step5_rectify_preview.py     ← Live check: are both cameras aligned?
├── step6_depth_full.py          ← 4-panel depth viewer (full quality)
└── step7_depth_map.py           ← Compact depth map with metre labels
```

---

## Install Dependencies (run once)

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## Run Order

### PHASE 1 — Calibrate each camera individually

```bash
python step1_capture_left.py
```
- Point chessboard at LEFT camera (index 4)
- Press S to save ~20 images at different angles/distances
- Press Q when done

```bash
python step1b_capture_right.py
```
- Same thing for RIGHT camera (index 2)

```bash
python step2_calibrate_individual.py
```
- Reads calib_left/ and calib_right/
- Prints K_L, D_L, K_R, D_R
- RMS should be < 1.0 — if higher, recapture more images
- Copy the printed values into step4_stereo_calibrate.py (already filled with example values)

---

### PHASE 2 — Stereo calibration

```bash
python step3_stereo_capture.py
```
- Hold chessboard so BOTH cameras see it simultaneously
- Press S to save a synchronized pair (~20 pairs)
- Press ESC when done

```bash
python step4_stereo_calibrate.py
```
- Prints Stereo RMS, Translation T, Baseline
- Stereo RMS should be < 1.0
- Copy R and T into step5/step6/step7 if they changed from defaults

---

### PHASE 3 — Verify and run depth

```bash
python step5_rectify_preview.py
```
- Shows rectified live view with green horizontal lines
- Lines should pass through the same features in both frames
- If misaligned, redo calibration

```bash
python step6_depth_full.py
```
- 4-panel view: Rectified Left | Rectified Right | Disparity | Depth
- Best for debugging and checking quality

```bash
python step7_depth_map.py
```
- Compact single depth map with distance values in metres
- Best for live use

---

## Tips

- Keep the chessboard flat and fully visible when capturing
- Vary the angle and distance of the chessboard during capture
- A baseline (camera separation) of ~10cm works well for 0.5–5m range
- If depth looks noisy in step6/7, increase images in calibration steps
