import cv2
# This script's only job: measure your face width in pixels at a KNOWN
# distance, so we can calculate the camera's focal length (f).
# f is then hardcoded into distance_estimator.py for the real depth/angle math.
print("Step 1: Script started")

# Load the pretrained face detector (same one used in the other scripts)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check the cascade file actually loaded correctly before continuing
if face_cascade.empty():
    print("ERROR: cascade file not loaded!")
else:
    print("Step 2: Cascade loaded fine")

# Open the webcam
cap = cv2.VideoCapture(0)

# Check the camera actually opened correctly before continuing
if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
else:
    print("Step 3: Camera opened fine")

# Force the window to open at a fixed, visible position on screen
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.moveWindow("Calibration", 100, 100)

print("Step 4: Window created, entering loop now...")

# Main loop: keeps reading camera frames until we press 'q'
while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Failed to read frame from camera")
        break
    # Face detection works on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face(s) in this frame
    # minNeighbors=8 and minSize=(80,80) reduce false detections, since we
    # only need ONE clean, reliable reading here (not real-time robustness)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Draw the box so we can visually confirm the detection looks right
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show face width in pixels on screen — this is the number we note
        # down and plug into: f = (w_px * known_distance) / real_face_width
        cv2.putText(frame, f"Width: {w}px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done, camera released.")