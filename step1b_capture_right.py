"""
STEP 1b — Capture RIGHT camera calibration images
Save ~20 images with the chessboard at different angles.
Press S to save, Q to quit.
"""
import cv2
import os

camera_index = 2   # Right camera index
save_path = "calib_right"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
print("Press S to save image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture RIGHT", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        filename = os.path.join(save_path, f"img_{count}.png")
        cv2.imwrite(filename, frame)
        print("Saved", filename)
        count += 1
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total saved: {count} images")
