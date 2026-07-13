"""
ADAS Lite Pipeline
==================
Reads a driving video, detects lane lines, detects vehicles (using YOLOv8 if
available, otherwise falls back to a simple color/contour detector so the
pipeline ALWAYS runs even without the model downloaded), overlays a warning
when a vehicle looks too close, and logs every frame's results to a CSV.

Usage:
    python adas_pipeline.py --input test_drive.mp4 --output annotated.mp4
"""
import argparse
import csv
import time
import cv2
import numpy as np

# Try to use YOLOv8 for real object detection; fall back gracefully if
# ultralytics isn't installed (keeps the pipeline runnable everywhere).
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_MODEL = YOLO("yolov8n.pt")
    USE_YOLO = True
except Exception:
    YOLO_MODEL = None
    USE_YOLO = False


def detect_lanes(frame):
    """Classical CV lane detection: grayscale -> blur -> edges -> ROI mask -> Hough lines."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest: a triangle covering the lower half of the road
    mask = np.zeros_like(edges)
    roi_pts = np.array([[
        (int(0.1 * w), h),
        (int(0.9 * w), h),
        (int(0.55 * w), int(0.6 * h)),
        (int(0.45 * w), int(0.6 * h)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_pts, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, threshold=40,
        minLineLength=40, maxLineGap=100
    )

    line_img = frame.copy()
    lane_count = 0
    left_lines = 0
    right_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:  # ignore horizontal-ish lines
                    continue
                if slope < 0:
                    left_lines += 1
                else:
                    right_lines += 1
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            lane_count += 1
            
    lane_departure = False
    if lane_count > 0: # Only check if we are tracking lanes
        if left_lines == 0 or right_lines == 0:
            lane_departure = True

    return line_img, lane_count, lane_departure


def detect_objects_fallback(frame, conf_threshold=0.5):
    """
    Lightweight fallback object detector (no model download required).
    Uses color-based segmentation (HSV range) rather than a plain intensity
    threshold, so it doesn't mistake the flat road/background for an object.
    This is only a stand-in for demoing the pipeline logic -- swap in YOLO
    (installed automatically above when ultralytics is available) for real
    accuracy on real footage.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Two color bands likely to correspond to "vehicle-like" saturated colors
    # (reds/oranges and yellows) while ignoring flat gray road/sky.
    lower1, upper1 = np.array([0, 80, 60]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 80, 60]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:  # filter noise
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < w * 0.9 and bh < h * 0.9:
                if 0.5 >= conf_threshold:
                    detections.append({"label": "object", "conf": 0.5, "id": None, "box": (x, y, bw, bh)})
    return detections


def detect_objects_yolo(frame, conf_threshold=0.5):
    results = YOLO_MODEL.track(frame, persist=True, verbose=False)[0]
    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
            
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = YOLO_MODEL.names[int(box.cls[0])]
        obj_id = int(box.id[0]) if box.id is not None else None
        
        if label in ("car", "truck", "bus", "person", "motorcycle", "bicycle"):
            detections.append({
                "label": label, "conf": conf, "id": obj_id,
                "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            })
    return detections


CLASS_COLORS = {
    "car": (255, 150, 0),       # Blueish
    "truck": (200, 100, 50),
    "bus": (200, 50, 50),
    "person": (0, 200, 255),    # Yellowish
    "motorcycle": (150, 50, 200),
    "bicycle": (50, 200, 150),
    "object": (0, 255, 255)     # Fallback
}

def annotate_frame(annotated, detections, warning, fps_value=None, lane_departure=False):
    """Shared annotation logic to draw bounding boxes and warnings on a frame."""
    for d in detections:
        x, y, bw, bh = d["box"]
        color = CLASS_COLORS.get(d["label"], (255, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        
        label_text = f"{d['label']} {d['conf']:.2f}"
        if d.get("id") is not None:
            label_text = f"ID:{d['id']} " + label_text
            
        cv2.putText(annotated, label_text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if warning:
        cv2.putText(annotated, "WARNING: OBJECT CLOSE", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
    if lane_departure:
        cv2.putText(annotated, "LANE DEPARTURE", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3) # Orange
                    
    if fps_value is not None:
        cv2.putText(annotated, f"FPS: {fps_value:.1f}", (annotated.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 150), 2)


def check_warning(detections, frame_shape):
    """Simple rule: if a detected box is large (close) and near horizontal center -> warning."""
    h, w = frame_shape[:2]
    for d in detections:
        x, y, bw, bh = d["box"]
        box_center_x = x + bw / 2
        near_center = abs(box_center_x - w / 2) < w * 0.25
        is_large = (bw * bh) > (0.03 * w * h)  # covers >3% of frame area
        if near_center and is_large:
            return True, d
    return False, None


def run_pipeline(input_path, output_path, log_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    log_rows = []
    frame_idx = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        annotated, lane_count, lane_departure = detect_lanes(frame)

        if USE_YOLO:
            detections = detect_objects_yolo(frame, conf_threshold=0.5)
        else:
            detections = detect_objects_fallback(frame, conf_threshold=0.5)

        warning, warn_obj = check_warning(detections, frame.shape)
        
        fps_value = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        annotate_frame(annotated, detections, warning, fps_value, lane_departure)

        log_rows.append({
            "frame": frame_idx,
            "lane_lines_detected": lane_count,
            "num_objects": len(detections),
            "object_labels": ";".join(d["label"] for d in detections),
            "warning_triggered": warning,
            "lane_departure": lane_departure,
        })

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys() if log_rows else
                                 ["frame", "lane_lines_detected", "num_objects",
                                  "object_labels", "warning_triggered"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Processed {frame_idx} frames.")
    print(f"Annotated video saved to: {output_path}")
    print(f"Detection log saved to: {log_path}")
    print(f"Object detector used: {'YOLOv8' if USE_YOLO else 'fallback contour detector'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_drive.mp4")
    parser.add_argument("--output", default="annotated.mp4")
    parser.add_argument("--log", default="detections_log.csv")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.log)
