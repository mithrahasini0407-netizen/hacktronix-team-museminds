# HackTronix 2.0 — Team Eclipse

**Track B — Artificial Intelligence**
**Team Members:** Mithrahasini T S, Sadhana M, Gayathri Priya K P

## Overview
This repository contains our solutions for the two AI Track qualifier tasks:
1. **Ball Detection** — real-time ball detection using a 2D camera
2. **Monocular Face Distance Estimation** — estimating face depth and angle from a single 2D camera image

## Folder Structure
```
├── task1_ball_detection/      # Ball detection system
├── task2_face_distance/       # Face distance & angle estimation system
└── README.md                  # This file
```
## Task 1: Ball Detection

### Approach
We use Ultralytics' pretrained **YOLOv8** object detection model (`yolov8n.pt`) to identify and track a ball in a live camera stream. The model extracts a bounding box around the detected ball, providing its precise position in the frame (center coordinates, width, and height in pixels). To maximize performance and maintain high frame rates (FPS), the image size is dynamically downscaled during the inference loop, and a custom confidence threshold is set to filter out background clutter:

- **Bounding Box Extraction:** `[x_center, y_center, width, height]`
- **Confidence Filtering:** `conf = 0.25`

Where the tracking script captures video frames via OpenCV, scales them down to reduce computational overhead, passes them through the neural network pipeline, and maps the resulting bounding box coordinates directly back onto the live display window.

### Tracking Optimization
We adjusted the core tracking parameters directly inside the pipeline to balance speed and accuracy:
```text
imgsz = 320         # Reduced input resolution for faster processing
conf = 0.25         # Confidence threshold to filter out background noise
verbose = False     # Disables debug text to save terminal overhead

```
### Performance Results
Tested on a standard laptop webcam under variable lighting conditions:
| Metrics | Value | Notes |
|---|---|---|
| **Model** | YOLOv8 Nano (yolov8n.pt) | Lightweight and optimized for edge devices |
| **Inference Resolution** | 320x320 pixels | Stripped down from default 640x640 for speed |
| **Average Frame Rate** | ~30+ FPS | Smooth real-time performance |
| **Detection Reliability** | High | Stable bounding boxes when ball is in clear view |
All tracking frames remain stable within close to medium proximity (~30-150cm). Precision remains highest when the target object maintains high color contrast against the background, showing minimal jitter during standard velocity changes—consistent with optimized anchors in deep-learning edge models.

### How to Run
 1. Install dependencies:
```bash
pip install opencv-python ultralytics

```
 2. Navigate to the task1 folder:
```bash
cd task1_ball_detection

```
 3. Run the main script:
```bash
python detect.py

```
 4. Press q to quit the video stream.
### Files
 * detect.py – main script: accesses the webcam, runs the YOLOv8 tracking loop, and renders live bounding boxes
 * requirements.txt – dependency list mapping out required packages like ultralytics
 * yolov8n.pt – cached pre-trained neural network weights used for object inference
### Known Limitations
 * Detection performance can drop under extreme motion blur or rapid movements.
 * Low-light environments reduce the model's confidence scores significantly.

## Task 2: Monocular Face Distance Estimation
### Approach
We use OpenCV's pretrained Haar Cascade classifier (haarcascade_frontalface_default.xml) to detect a face in each camera frame and obtain a bounding box (position and width in pixels). We then apply the pinhole camera model formulas provided in the problem statement to estimate depth and horizontal deviation angle:
 * **Depth:** Z = (f x W) / w_px
 * **Angle:** θ = arctan((x - c_x) / f)
Where f is our camera's focal length (found via calibration), W is the average real-world face width (15cm), w_px is the detected face width in pixels, x is the face center's x-coordinate, and c_x is the image center.
### Calibration
We calibrated our focal length by measuring face width in pixels at a known distance of 40cm:
```text
f = (w_px x known_distance) / W
f = (133 x 40) / 15
f ≈ 354.7

```
### Accuracy Results
Tested across a range of real distances using a physical ruler:
| Actual Distance | Measured Distance | Error |
|---|---|---|
| 30cm | 39cm | 9cm |
| 40cm | 47cm | 7cm |
| 60cm | 63cm | 3cm |
| 90cm | ~70cm | ~20cm |
| 120cm | 92.5cm | 27.5cm |
| 150cm | 109cm | 41cm |
All results are within the task's accepted tolerance of ±50-150cm. Accuracy is highest in the 40-60cm range, with increasing deviation at larger distances – consistent with known bounding-box variance in Haar Cascade detectors at range.
### How to Run
 1. Install dependencies:
```bash
pip install opencv-python

```
 2. Navigate to the task2 folder:
```bash
cd task2_face_distance

```
 3. Run the main script:
```bash
python distance_estimator.py

```
 4. Press q to quit.
### Files
 * distance_estimator.py – main script: detects face, calculates and displays depth + angle
 * calibrate.py – helper script used to measure face width in pixels at a known distance, for focal length calculation
 * face_detect.py – early-stage face detection test script
 * test_camera.py – basic webcam test script
 * haarcascade_frontalface_default.xml – pretrained face detection model (from OpenCV)
### Known Limitations
 * Haar Cascade accuracy decreases at longer distances (>90cm) due to bounding box size variance
 * Detection requires reasonably good lighting and a front-facing pose
```
---

## Task 2: Monocular Face Distance Estimation

### Approach
We use OpenCV's pretrained Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to detect a face in each camera frame and extract its bounding box (position and width in pixels). We then apply the pinhole camera model formulas provided in the problem statement to estimate depth and horizontal deviation angle:

- **Depth:** `Z = (f × W) / w_px`
- **Angle:** `θ = arctan((x − c_x) / f)`

Where `f` is our camera's focal length (found via calibration), `W` is the average real-world face width (15cm), `w_px` is the detected face width in pixels, `x` is the face center's x-coordinate, and `c_x` is the image center.

### Calibration
We calibrated our focal length by measuring face width in pixels at a known distance of 40cm:
```
f = (w_px × known_distance) / W
f = (133 × 40) / 15
f ≈ 354.7
```

### Accuracy Results
Tested across a range of real distances using a physical ruler:

| Actual Distance | Measured Distance | Error |
|---|---|---|
| 30cm | 39cm | 9cm |
| 40cm | 47cm | 7cm |
| 60cm | 63cm | 3cm |
| 90cm | ~70cm | ~20cm |
| 120cm | 92.5cm | 27.5cm |
| 150cm | 109cm | 41cm |

All results are within the task's accepted tolerance of ±50–150cm. Accuracy is highest in the 40–60cm range, with increasing deviation at longer distances — consistent with known bounding-box variance in Haar Cascade detectors at range.

### How to Run
1. Install dependencies:
```
pip install opencv-python
```
2. Navigate to the task2 folder:
```
cd task2_face_distance
```
3. Run the main script:
```
python distance_estimator.py
```
4. Press `q` to quit.

### Files
- `distance_estimator.py` — main script: detects face, calculates and displays depth + angle
- `calibrate.py` — helper script used to measure face width in pixels at a known distance, for focal length calibration
- `face_detect.py` — early-stage face detection test script
- `test_camera.py` — basic webcam test script
- `haarcascade_frontalface_default.xml` — pretrained face detection model (from OpenCV)

### Known Limitations
- Haar Cascade accuracy decreases at longer distances (>90cm) due to bounding box size variance
- Detection requires reasonably good lighting and a front-facing pose

---

