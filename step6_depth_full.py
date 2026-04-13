import cv2
import numpy as np
import time

DISPLAY_RES = (1280, 720)
COMPUTE_RES = (640, 480)
MAX_DEPTH   = 10


B = 0.1035

K_L = np.array([[921.41524262, 0, 343.05532474],
                [0, 922.83853798, 246.50272108],
                [0, 0, 1]])
D_L = np.array([[0.05890666, 0.09953243, 0.00140495, -0.00240478, -1.72048844]])

K_R = np.array([[922.50156509, 0, 323.14004823],
                [0, 923.51233289, 241.83288167],
                [0, 0, 1]])
D_R = np.array([[0.07078698, -0.65754242, 0.00204820, -0.00097100, 1.92690983]])

R = np.array([[ 0.99203797,  0.02568906,  0.12329131],
              [-0.02484975,  0.9996564 , -0.00834071],
              [-0.12346321,  0.00521054,  0.99233547]])
T = np.array([[0.10313029], [0.00047481], [-0.00917416]])

image_size = (640, 480)

# ================= RECTIFICATION =================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

f = P1[0, 0]

mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, image_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, image_size, cv2.CV_32FC1)

# ================= SGBM =================
WIN      = 5
MAX_DISP = 128  

stereoL = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=MAX_DISP,
    blockSize=WIN,
    P1=8 * 3 * WIN**2,
    P2=32 * 3 * WIN**2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

stereoR = cv2.ximgproc.createRightMatcher(stereoL)

wls = cv2.ximgproc.createDisparityWLSFilter(stereoL)
wls.setLambda(12000)      
wls.setSigmaColor(1.5)


def disp_to_depth(disp):
    depth = np.zeros_like(disp, dtype=np.float32)
    mask = disp > 1
    depth[mask] = (f * B) / disp[mask]
    return np.clip(depth, 0, MAX_DEPTH)

# ================= CAMERA =================
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(2)

for cap in [capL, capR]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Stereo Depth Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Stereo Depth Camera",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

print("Press ESC to quit.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    Lg = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    Rg = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    Lg = cv2.equalizeHist(Lg)
    Rg = cv2.equalizeHist(Rg)

    t0 = time.time()

    dispL = stereoL.compute(Lg, Rg).astype(np.float32) / 16
    dispR = stereoR.compute(Rg, Lg).astype(np.float32) / 16

    disp = wls.filter(dispL, Lg, None, dispR)

    disp = cv2.medianBlur(disp, 5)
    disp[disp < 0] = 0

    depth = disp_to_depth(disp)

    fps = 1 / (time.time() - t0)

    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

    h, w = 480, 640
    rectL = cv2.resize(rectL, (w, h))
    rectR = cv2.resize(rectR, (w, h))
    disp_color = cv2.resize(disp_color, (w, h))
    depth_color = cv2.resize(depth_color, (w, h))

    top = np.hstack((rectL, rectR))
    bottom = np.hstack((disp_color, depth_color))

    frame = np.vstack((top, bottom))

    frame = cv2.resize(frame, DISPLAY_RES)

    cv2.putText(frame, f"FPS: {fps:.2f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Stereo Depth Camera", frame)

    if cv2.waitKey(1) == 27:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
