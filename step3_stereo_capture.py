"""
STEP 3 — Capture synchronized stereo image pairs
Both cameras capture simultaneously.
Press S to save a pair, ESC to quit.
Aim for ~20 pairs showing chessboard from different angles.
"""
import cv2
import os

left_index = 1
right_index = 2

os.makedirs("stereo_left", exist_ok=True)
os.makedirs("stereo_right", exist_ok=True)

capL = cv2.VideoCapture(left_index)
capR = cv2.VideoCapture(right_index)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
print("Press S to save a stereo pair")
print("Press ESC to quit")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    combined = cv2.hconcat([frameL, frameR])
    cv2.putText(combined, f"Pairs saved: {count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Stereo Capture (Left | Right)", combined)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"stereo_left/img_{count}.png", frameL)
        cv2.imwrite(f"stereo_right/img_{count}.png", frameR)
        print(f"Saved pair {count}")
        count += 1
    if key == 27:  # ESC
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
print(f"Total pairs saved: {count}")
