import cv2
import math

# ---- Calibrated values (from your setup) ----
FOCAL_LENGTH = 354.7      # recalculated from careful calibration
REAL_FACE_WIDTH = 15      # cm, average human face width

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

cv2.namedWindow("Distance Estimator", cv2.WINDOW_NORMAL)
cv2.moveWindow("Distance Estimator", 100, 100)

# Get frame width to find image center (c_x)
ret, test_frame = cap.read()
frame_height, frame_width = test_frame.shape[:2]
c_x = frame_width // 2

print(f"Camera resolution: {frame_width}x{frame_height}, image center x = {c_x}")
print("Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_center_x = x + w // 2
        face_center_y = y + h // 2
        w_px = w

        # ---- Depth formula ----
        Z = (FOCAL_LENGTH * REAL_FACE_WIDTH) / w_px

        # ---- Angle formula ----
        theta_rad = math.atan((face_center_x - c_x) / FOCAL_LENGTH)
        theta_deg = math.degrees(theta_rad)

        # Display on screen
        cv2.putText(frame, f"Depth: {Z:.1f} cm", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle: {theta_deg:.1f} deg", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

    # This checks: did we find ANY face at all this frame?
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Distance Estimator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()