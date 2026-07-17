import cv2

# Open the webcam (0 means "default camera")
cap = cv2.VideoCapture(0)

while True:
    # Read one frame (one photo) from the webcam
    success, frame = cap.read()

    if not success:
        print("Could not access webcam")
        break

    # Show that frame in a window
    cv2.imshow("My Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()