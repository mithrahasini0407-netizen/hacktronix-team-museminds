# cv2 lets us use the webcam and detect faces
# math lets us use arctan for the angle calculation
import cv2
import math
# These two values come from our calibration step (measured at 40cm)
FOCAL_LENGTH = 354.7      # recalculated from careful calibration
REAL_FACE_WIDTH = 15      # average human face width in cm

# Load the pretrained face detector (comes from OpenCV, trained by its creators)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Open the webcam (0 = default/built-in camera)
cap = cv2.VideoCapture(0)

# Create a window and force it to open at a fixed, visible position on screen
cv2.namedWindow("Distance Estimator", cv2.WINDOW_NORMAL)
cv2.moveWindow("Distance Estimator", 100, 100)

# Grab one test frame just to measure the camera's resolution
ret, test_frame = cap.read()
frame_height, frame_width = test_frame.shape[:2]
# c_x = the horizontal center of the image in pixels
# We need this for the angle formula: theta = arctan((face_x - c_x) / f)
c_x = frame_width // 2

print(f"Camera resolution: {frame_width}x{frame_height}, image center x = {c_x}")
print("Press 'q' to quit.")

# Main loop: runs once per camera frame, forever, until we press 'q'
while True:
    success, frame = cap.read()
    if not success:
        break

    # Face detection works better on grayscale (black & white) images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find all faces in this frame
    # minNeighbors=8 and minSize=(50,50) reduce false detections (random shadows etc.)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))

    # Loop through every face found (usually just one)
    for (x, y, w, h) in faces:
        # x, y = top-left corner of face box; w, h = width/height in pixels
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the center point of the face (not the corner)
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        w_px = w    # face width in pixels, needed for the depth formula

         # ---- Depth formula: Z = (f * W) / w_px ----
        # Bigger face in pixels = closer to camera = smaller Z
        Z = (FOCAL_LENGTH * REAL_FACE_WIDTH) / w_px

        # ---- Angle formula: theta = arctan((x - c_x) / f) ----
        # Tells us how far left/right the face is from the center of the frame
        theta_rad = math.atan((face_center_x - c_x) / FOCAL_LENGTH)
        theta_deg = math.degrees(theta_rad)

        # Show the calculated depth and angle as text on screen
        cv2.putText(frame, f"Depth: {Z:.1f} cm", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle: {theta_deg:.1f} deg", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw a red dot at the exact center of the face, just for visual confirmation
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)

    # If no face was found this frame, show a message instead of leaving it blank
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Distance Estimator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()