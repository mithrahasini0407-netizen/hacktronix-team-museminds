import cv2

print("Step 1: Script started")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("ERROR: cascade file not loaded!")
else:
    print("Step 2: Cascade loaded fine")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
else:
    print("Step 3: Camera opened fine")

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.moveWindow("Calibration", 100, 100)

print("Step 4: Window created, entering loop now...")

while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Failed to read frame from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Width: {w}px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done, camera released.")