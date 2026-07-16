import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("ERROR: XML file not found or failed to load!")
else:
    print("Face detector loaded successfully!")

cap = cv2.VideoCapture(0)

# Force the window to a specific, visible spot on your screen
cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
cv2.moveWindow("Face Detection", 100, 100)
cv2.resizeWindow("Face Detection", 640, 480)

print("Camera starting... Press 'q' while the video window is focused to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Camera released. Done!")