import cv2
import numpy as np
import time

DISPLAY_RES = (640, 360)
MAX_DEPTH = 5
f = 922
B = 0.1035

# ---- calibration ----
K_L = np.array([[937.67702933, 0 , 336.48593563],
 [  0, 940.67413932, 233.68235647],
 [  0,0,1   ]])
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

image_size = (640, 480)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, image_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, image_size, cv2.CV_32FC1)

# ---- stereo ----
stereoL = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=7,
    P1=8*7*7,
    P2=32*7*7,
    uniquenessRatio=5,
    speckleWindowSize=300,
    speckleRange=32
)

stereoR = cv2.ximgproc.createRightMatcher(stereoL)
wls = cv2.ximgproc.createDisparityWLSFilter(stereoL)
wls.setLambda(12000)
wls.setSigmaColor(1.5)

def disp_to_depth(disp):
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 2
    depth[valid] = (f * B) / disp[valid]
    return np.clip(depth, 0, MAX_DEPTH)

capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)

for cap in [capL, capR]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press ESC to quit.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    # 🔥 STEP 1: STRICT COMMON ROI
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    x = max(x1, x2)
    y = max(y1, y2)
    w = min(w1, w2)
    h = min(h1, h2)

    rectL = rectL[y:y+h, x:x+w]
    rectR = rectR[y:y+h, x:x+w]

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    t0 = time.time()

    dispL = stereoL.compute(grayL, grayR).astype(np.float32) / 16
    dispR = stereoR.compute(grayR, grayL).astype(np.float32) / 16

    disp = wls.filter(dispL, grayL, None, dispR)

    # 🔥 STEP 2: REMOVE INVALID REGION (MAIN FIX)
    disp[:, :50] = 0   # force remove left strip

    disp = cv2.medianBlur(disp, 5)
    disp = cv2.bilateralFilter(disp, 9, 75, 75)

    disp[disp < 1] = 0

    depth = disp_to_depth(disp)
    depth = cv2.medianBlur(depth, 5)

    fps = 1 / (time.time() - t0)

    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

    depth_display = cv2.resize(depth_color, DISPLAY_RES)

    h_disp, w_disp = depth_display.shape[:2]
    h_org, w_org = depth.shape

    for y in range(0, h_disp, 80):
        for x in range(0, w_disp, 80):

            ox = int(x * (w_org / w_disp))
            oy = int(y * (h_org / h_disp))

            window = depth[max(0, oy-3):oy+3, max(0, ox-3):ox+3]
            valid = window[window > 0]

            if len(valid) > 0:
                d = np.mean(valid)
                cv2.putText(depth_display, f"{d:.2f}m", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.putText(depth_display, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Depth Map", depth_display)

    if cv2.waitKey(1) == 27:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()