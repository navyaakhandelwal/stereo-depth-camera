import cv2

print("Scanning cameras...")
for i in range(8):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera WORKING at index {i}")
        else:
            print(f"~ Camera opens but no frame at index {i}")
        cap.release()
    else:
        print(f"✗ No camera at index {i}")

print("Done.")

#It will print something like:No camera at index 0 Camera found at index Camera found at index No camera at index 3