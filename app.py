"""
ADAS Lite - FastAPI Streaming Backend
=====================================
Streams processed video frames (lane detection + object detection + warnings)
live to the browser as MJPEG, avoiding the mp4v codec playback issues that
break in VS Code / most browsers with plain saved .mp4 files.

Run with: uvicorn main:app --reload
Then open: http://localhost:8000
"""
import os
import uuid
import shutil
import tempfile

import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from adas_pipeline import detect_lanes, detect_objects_yolo, detect_objects_fallback, check_warning, USE_YOLO

app = FastAPI(title="ADAS Lite")
templates = Jinja2Templates(directory="templates")

# In-memory store: video_id -> {path, log}
SESSIONS = {}
UPLOAD_DIR = tempfile.mkdtemp(prefix="adas_uploads_")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"using_yolo": USE_YOLO})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    SESSIONS[video_id] = {"path": save_path, "log": []}
    return {"video_id": video_id}


def frame_generator(video_id: str):
    session = SESSIONS[video_id]
    cap = cv2.VideoCapture(session["path"])
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, lane_count = detect_lanes(frame)

        if USE_YOLO:
            detections = detect_objects_yolo(frame)
        else:
            detections = detect_objects_fallback(frame)

        for d in detections:
            x, y, bw, bh = d["box"]
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(annotated, f"{d['label']} {d['conf']:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        warning, _ = check_warning(detections, frame.shape)
        if warning:
            cv2.putText(annotated, "WARNING: OBJECT CLOSE", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        session["log"].append({
            "frame": frame_idx,
            "lane_lines_detected": lane_count,
            "num_objects": len(detections),
            "object_labels": ";".join(d["label"] for d in detections),
            "warning_triggered": bool(warning),
        })

        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        frame_idx += 1

    cap.release()
    session["done"] = True


@app.get("/stream/{video_id}")
def stream(video_id: str):
    if video_id not in SESSIONS:
        return JSONResponse({"error": "unknown video_id"}, status_code=404)
    return StreamingResponse(
        frame_generator(video_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/log/{video_id}")
def get_log(video_id: str):
    if video_id not in SESSIONS:
        return JSONResponse({"error": "unknown video_id"}, status_code=404)
    session = SESSIONS[video_id]
    log = session["log"]
    total_warnings = sum(1 for row in log if row["warning_triggered"])
    avg_objects = (sum(row["num_objects"] for row in log) / len(log)) if log else 0
    return {
        "frames_processed": len(log),
        "total_warnings": total_warnings,
        "avg_objects_per_frame": round(avg_objects, 2),
        "done": session.get("done", False),
        "rows": log[-50:],  # last 50 rows to keep payload small
    }


@app.get("/health")
def health():
    return {"status": "ok", "using_yolo": USE_YOLO}