# ADAS Lite: Lane Detection + Object Detection + Warning System

A lightweight Advanced Driver Assistance System (ADAS) perception pipeline that:
- Detects lane lines using classical computer vision (Canny edges + Hough Transform)
- Detects vehicles/pedestrians (YOLOv8 if installed, otherwise a built-in fallback detector)
- Triggers a collision-proximity warning when a detected object is large and centered (i.e. close and in-path)
- Logs every frame's detections to a CSV for downstream analysis
- Includes a FastAPI + MJPEG streaming web app to upload a video and see results interactively live in the browser

## Why I built this
Inspired by real-world ADAS/perception systems used in automotive AI (lane keeping, forward
collision warning). This project demonstrates an end-to-end perception pipeline: input video →
computer vision processing → rule-based decision logic → structured logging → interactive demo.

## Project structure
```
adas_lite/
├── adas_pipeline.py     # Core pipeline: lane detection, object detection, warning logic, logging
├── app.py                # FastAPI streaming backend and demo app
├── make_test_video.py    # Generates a synthetic test video (no external dataset needed to try it)
├── test_drive.mp4        # Pre-generated sample video
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```
> Note: `ultralytics` (YOLOv8) will download its model weights (~6MB) the first time it runs.
> If you don't want that, the pipeline automatically falls back to a lightweight OpenCV-based
> detector, no download required.

## Usage

### Option A: Command line
```bash
python adas_pipeline.py --input test_drive.mp4 --output annotated.mp4 --log detections_log.csv
```
This produces:
- `annotated.mp4` — video with lane lines, bounding boxes, and warnings overlaid
- `detections_log.csv` — per-frame log (lane count, objects detected, warning triggered)

### Option B: Interactive web demo
```bash
uvicorn app:app --reload
```
Open `http://localhost:8000` in your browser. Upload any driving video (or generate your own test clip with `make_test_video.py`) and see:
- Side-by-side original vs. annotated video
- A live table of detection logs
- Summary metrics (total warnings, average objects per frame)
- Charts of warnings/objects over time

## How it works

**1. Lane Detection** (`detect_lanes`)
Grayscale → Gaussian blur → Canny edge detection → region-of-interest mask (focuses on the road
area, ignoring sky/background) → Hough Line Transform to find straight line segments representing
lane markings.

**2. Object Detection** (`detect_objects_yolo` / `detect_objects_fallback`)
If `ultralytics` (YOLOv8) is installed, uses a pretrained model to detect cars, trucks, buses,
pedestrians, motorcycles, and bicycles with real confidence scores. Falls back to an HSV
color-segmentation + contour detector if YOLO isn't available, so the pipeline always runs.

**3. Warning Logic** (`check_warning`)
A detected object triggers a warning if it is (a) large relative to the frame (i.e. close to the
camera) and (b) horizontally centered (i.e. in the vehicle's path) — a simplified stand-in for
real-world forward collision warning heuristics.

**4. Logging**
Every frame's lane count, number of objects, object labels, and warning status is written to a
CSV — the foundation for downstream monitoring/analysis (and a natural extension point for a
fuller MLOps pipeline: tracking detection drift, false-positive rates, etc. over time).

## Testing it right now
A synthetic test video (`test_drive.mp4`) is included so you can verify the pipeline works
without needing to source real driving footage. It simulates a road with lane markings and a
vehicle that grows closer over time, triggering the warning partway through.

To generate a fresh one: `python make_test_video.py`

## Future extensions
- Swap the fallback detector entirely for YOLOv8 fine-tuned on a real driving dataset (BDD100K, KITTI)
- Add a RAG-based explainability layer: query why a specific warning was triggered by retrieving
  from traffic rule documentation and the detection log
- Wrap in a full MLOps pipeline: experiment tracking (MLflow/W&B), containerized inference API
  (FastAPI + Docker), CI tests, and drift monitoring on live detection logs
- Train a real lane-segmentation model (e.g. U-Net) instead of classical Hough-line detection for
  robustness on curved roads and varied lighting

## Tech stack
Python, OpenCV, NumPy, Ultralytics YOLOv8, FastAPI, Jinja2
