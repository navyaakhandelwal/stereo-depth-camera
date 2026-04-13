"""
STEP 5 — Live rectification preview
Verifies that both cameras are properly aligned after calibration.
Green horizontal lines should pass through the same features
in both the left and right frames.
Press ESC to quit.
"""
import cv2
import numpy as np

image_size = (640, 480)

# ==============================
# CALIBRATION PARAMETERS
# ==============================
K_L = np.array([[937.67702933, 0 , 336.48593563],
 [  0, 940.67413932, 233.68235647],
 [  0,0,1.   ]])
D_L = np.array([[-3.43366130e-02, 1.51190151e+00, -5.85728229e-03, -6.79162812e-03, -8.69198625e+00]])

K_R = np.array([[931.29552118, 0,  306.91008027],
 [  0, 933.80616682, 234.25443327],
 [  0, 0, 1    ]])

D_R = np.array([[ 3.26656164e-02, 5.39165337e-01, -5.62185674e-04,-8.11218007e-03, -5.88842996e+00]])

R = np.array ([[ 0.99922983, -0.00453485, -0.03897667], 
[ 0.00672271, 0.99839768, 0.05618605], 
[ 0.03865942, -0.05640481, 0.99765923]])

T = np.array([[-0.10415867],
 [ 0.00127448],
 [-0.00316938]])

# ==============================
# RECTIFICATION SETUP
# ==============================
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R,
    image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, image_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, image_size, cv2.CV_32FC1)

# ==============================
# LIVE PREVIEW
# ==============================
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Check that green lines align with the same features in both frames.")
print("Press ESC to quit.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    combined = cv2.hconcat([rectL, rectR])

    # Draw horizontal guide lines every 40px
    for i in range(0, 480, 40):
        cv2.line(combined, (0, i), (1280, i), (0, 255, 0), 1)

    cv2.imshow("Rectified Stereo — Check Horizontal Alignment", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
