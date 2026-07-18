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
*(To be filled in — see task1_ball_detection folder)*
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

