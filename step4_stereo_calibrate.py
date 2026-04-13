"""
STEP 4 — Stereo calibration
Uses stereo_left/ and stereo_right/ image pairs to compute
R (rotation) and T (translation) between the two cameras.
Prints Stereo RMS, T vector, and Baseline distance.
"""
import cv2
import numpy as np
import glob

rows = 6
cols = 9
square_size = 0.025

objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objp *= square_size

# ==============================
# PASTE YOUR K and D VALUES HERE
# (from step2_calibrate_individual.py output)
# ==============================
K_L = np.array([[937.67702933, 0 , 336.48593563],
 [  0, 940.67413932, 233.68235647],
 [  0,0,1.   ]])
D_L = np.array([[-3.43366130e-02, 1.51190151e+00, -5.85728229e-03, -6.79162812e-03, -8.69198625e+00]])

K_R = np.array([[931.29552118, 0,  306.91008027],
 [  0, 933.80616682, 234.25443327],
 [  0, 0, 1    ]])

D_R = np.array([[ 3.26656164e-02, 5.39165337e-01, -5.62185674e-04,-8.11218007e-03, -5.88842996e+00]])

# ==============================
objpoints = []
imgpointsL = []
imgpointsR = []

left_images  = sorted(glob.glob("stereo_left/*.png"))
right_images = sorted(glob.glob("stereo_right/*.png"))

print(f"Found {len(left_images)} left / {len(right_images)} right images")

gray_shape = None
for fnameL, fnameR in zip(left_images, right_images):
    imgL = cv2.imread(fnameL)
    imgR = cv2.imread(fnameR)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    gray_shape = grayL.shape

    retL, cornersL = cv2.findChessboardCorners(grayL, (cols, rows), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (cols, rows), None)

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
    else:
        print(f"  Skipped pair (corners not found): {fnameL}")

print(f"Valid pairs used: {len(objpoints)}")

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    K_L, D_L, K_R, D_R,
    gray_shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("\n" + "="*60)
print(f"Stereo RMS:  {ret:.4f}  (good if < 1.0)")
print(f"Translation T:\n{T}")
print(f"Baseline:    {np.linalg.norm(T):.4f} m")
print(f"Rotation R:\n{R}")
print("="*60)
print("Copy R and T values into step5_rectify_preview.py,")
print("step6_depth_full.py, and step7_depth_map.py")
