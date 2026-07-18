import cv2
import time
from ultralytics import YOLO

# Initialize target detection model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Execute inference pipeline on a lightweight input resolution grid
    results = model(frame, imgsz=320, conf=0.25, stream=True,verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Target identification filters
            if class_name in ["sports ball", "frisbee", "orange", "apple"]: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Draw target localization structures
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                label = f"Ball: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Frame processing metrics calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display video stream overlay
    cv2.imshow("Ball Detection Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
