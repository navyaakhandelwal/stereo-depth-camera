"""
STEP 2 — Individual camera calibration (LEFT and RIGHT)
Reads images from calib_left/ and calib_right/
Prints camera matrix (K) and distortion (D) for each camera.
Copy the printed values into step4_stereo_calibrate.py
"""
import cv2
import numpy as np
import glob

rows = 6
cols = 9
square_size = 0.025  # 25mm squares

objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objp *= square_size


def calibrate_camera(image_folder, label):
    objpoints = []
    imgpoints = []
    images = glob.glob(f"{image_folder}/*.png")
    print(f"\n[{label}] Found {len(images)} images in {image_folder}/")

    img_shape = None
    for fname in sorted(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
        else:
            print(f"  WARNING: chessboard not found in {fname}")

    if not objpoints:
        print(f"  ERROR: No valid images found for {label}!")
        return None, None

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    print(f"\n  [{label}] RMS error: {ret:.4f}  (good if < 1.0)")
    print(f"  Camera Matrix (K_{label[0]}):\n{mtx}")
    print(f"  Distortion (D_{label[0]}):\n{dist}")
    return mtx, dist


K_L, D_L = calibrate_camera("calib_left", "LEFT")
K_R, D_R = calibrate_camera("calib_right", "RIGHT")

print("\n" + "="*60)
print("Copy these values into step4_stereo_calibrate.py")
print("="*60)
